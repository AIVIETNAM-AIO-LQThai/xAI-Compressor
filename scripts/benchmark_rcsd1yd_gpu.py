from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ml.data.rcsd1yd import sha256_file
from ml.energy.gpu_meter import NVMLGpuMeter
from ml.energy.hierarchical_inference import energy_reduction_fraction
from ml.energy.rcsd1yd_evaluation import (
    ROUTE_NAMES,
    build_frozen_routes,
    build_tcn_model,
    build_temporal_batch_from_checkpoint,
    final_external_status,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY_CONFIG_PATH = ROOT / "configs/rcsd1yd_external_evaluation.yaml"
EXEC_CONFIG_PATH = ROOT / "configs/rcsd1yd_one_shot_execution.yaml"
HARNESS_PREFLIGHT_PATH = (
    ROOT / "docs/research/rcsd1yd_one_shot_execution_preflight.json"
)


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return value


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return value


def _run_logical_pass(
    *,
    model,
    inputs: np.ndarray,
    targets: np.ndarray,
    selected: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    if selected.size == 0:
        return 0.0, 0.0

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    started = perf_counter()
    checksum = 0.0

    with torch.inference_mode():
        for start in range(0, selected.size, batch_size):
            idx = selected[start : start + batch_size]
            x = torch.from_numpy(inputs[idx]).to(device)
            y = torch.from_numpy(targets[idx]).to(device)
            prediction = model(x)
            scores = torch.mean((y - prediction).square(), dim=1)
            checksum += float(scores.detach().cpu().sum().item())

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return perf_counter() - started, checksum


def _measure_idle(
    *,
    device_index: int,
    interval_s: float,
    duration_s: float,
) -> dict[str, Any]:
    meter = NVMLGpuMeter(
        device_index=device_index,
        interval_s=interval_s,
    )
    meter.start()
    time.sleep(duration_s)
    return meter.stop().to_dict()


def _measure_condition(
    *,
    model,
    inputs: np.ndarray,
    targets: np.ndarray,
    selected: np.ndarray,
    batch_size: int,
    device: torch.device,
    device_index: int,
    interval_s: float,
    minimum_workload_s: float,
) -> dict[str, Any]:
    if selected.size == 0:
        return {
            "zero_invocation_condition": True,
            "meter": None,
            "logical_passes": 0,
            "invocations_per_logical_pass": 0,
            "total_tcn_invocations": 0,
            "gross_gpu_energy_per_logical_pass_j": 0.0,
            "incremental_gpu_energy_per_logical_pass_j": 0.0,
            "checksum": 0.0,
        }

    meter = NVMLGpuMeter(
        device_index=device_index,
        interval_s=interval_s,
    )
    passes = 0
    checksum = 0.0

    meter.start()
    started = perf_counter()
    while True:
        _, pass_checksum = _run_logical_pass(
            model=model,
            inputs=inputs,
            targets=targets,
            selected=selected,
            batch_size=batch_size,
            device=device,
        )
        checksum += pass_checksum
        passes += 1
        if perf_counter() - started >= minimum_workload_s:
            break

    summary = meter.stop().to_dict()
    return {
        "zero_invocation_condition": False,
        "meter": summary,
        "logical_passes": passes,
        "invocations_per_logical_pass": int(selected.size),
        "total_tcn_invocations": passes * int(selected.size),
        "checksum": checksum,
    }


def main() -> None:
    study = _yaml(STUDY_CONFIG_PATH)
    execution = _yaml(EXEC_CONFIG_PATH)
    preflight = _json(HARNESS_PREFLIGHT_PATH)

    evaluation_result_path = ROOT / execution["evaluation"]["result_file"]
    final_result_path = ROOT / execution["energy_execution"]["final_result_file"]

    if not evaluation_result_path.exists():
        raise RuntimeError("Run the one-shot evidence evaluation first.")
    if final_result_path.exists():
        raise RuntimeError(
            "Final RCSD energy result already exists; refusing to overwrite."
        )

    evaluation_result = _json(evaluation_result_path)
    frozen = execution["frozen_artifacts"]
    freeze_path = ROOT / frozen["adaptation_freeze_file"]
    checkpoint_path = ROOT / frozen["tcn_checkpoint_file"]
    evaluation_path = ROOT / frozen["evaluation_parquet"]

    if sha256_file(freeze_path) != str(
        frozen["expected_adaptation_freeze_sha256"]
    ):
        raise RuntimeError("Adaptation freeze hash changed.")
    if sha256_file(checkpoint_path) != str(
        frozen["expected_tcn_checkpoint_sha256"]
    ):
        raise RuntimeError("TCN checkpoint hash changed.")
    if sha256_file(evaluation_path) != str(
        frozen["expected_evaluation_parquet_sha256"]
    ):
        raise RuntimeError("EVALUATION parquet hash changed.")

    freeze = _json(freeze_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    evaluation = pd.read_parquet(evaluation_path)
    evaluation.index = pd.DatetimeIndex(
        evaluation.index,
        name="timestamp",
    )

    batch = build_temporal_batch_from_checkpoint(
        evaluation,
        checkpoint,
    )
    context = build_frozen_routes(
        frame=evaluation,
        temporal_batch=batch,
        adaptation_freeze=freeze,
    )

    for name in ROUTE_NAMES:
        expected_count = int(
            evaluation_result["systems"][name]["tcn_invocations"]
        )
        observed_count = int(context.routes[name].sum())
        if observed_count != expected_count:
            raise RuntimeError(
                f"{name} route count changed: "
                f"{observed_count} != {expected_count}."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("RCSD energy benchmark requires CUDA.")

    model = build_tcn_model(checkpoint, device=device)

    energy_cfg = study["energy_measurement"]
    exec_energy = execution["energy_execution"]
    conditions = [str(value) for value in exec_energy["conditions"]]

    if tuple(conditions) != ROUTE_NAMES:
        raise RuntimeError("Energy condition list changed.")

    orders = preflight["energy_protocol"]["condition_orders"]
    repetitions = int(energy_cfg["measured_repetitions"])
    if len(orders) != repetitions:
        raise RuntimeError("Frozen condition-order count changed.")

    selected_by_condition = {
        name: np.flatnonzero(context.routes[name]).astype(
            np.int64,
            copy=False,
        )
        for name in conditions
    }

    batch_size = int(exec_energy["inference_batch_size"])
    warmups = int(energy_cfg["warmup_runs_per_condition"])

    for name in conditions:
        selected = selected_by_condition[name]
        if selected.size == 0:
            continue
        for _ in range(warmups):
            _run_logical_pass(
                model=model,
                inputs=batch.inputs,
                targets=batch.targets,
                selected=selected,
                batch_size=batch_size,
                device=device,
            )

    device_index = int(exec_energy["gpu_device_index"])
    interval_s = float(energy_cfg["nvml_sampling_interval_seconds"])
    cooldown_s = float(exec_energy["cooldown_seconds"])
    idle_s = float(energy_cfg["matched_idle_seconds"])
    minimum_s = float(energy_cfg["minimum_metered_workload_seconds"])

    records: list[dict[str, Any]] = []

    print("Measured RCSD repetitions")
    for repetition, order in enumerate(orders, start=1):
        print(f"  repetition {repetition}/{repetitions}: {order}")

        for condition in order:
            selected = selected_by_condition[condition]

            if selected.size == 0:
                record = {
                    "repetition": repetition,
                    "condition": condition,
                    "idle": None,
                    "workload": _measure_condition(
                        model=model,
                        inputs=batch.inputs,
                        targets=batch.targets,
                        selected=selected,
                        batch_size=batch_size,
                        device=device,
                        device_index=device_index,
                        interval_s=interval_s,
                        minimum_workload_s=minimum_s,
                    ),
                    "accounting": {
                        "gross_gpu_energy_per_logical_pass_j": 0.0,
                        "incremental_gpu_energy_per_logical_pass_j": 0.0,
                        "unmetered_zero_invocation_condition": True,
                    },
                }
                records.append(record)
                print(f"    {condition}: zero TCN invocations")
                continue

            time.sleep(cooldown_s)
            idle = _measure_idle(
                device_index=device_index,
                interval_s=interval_s,
                duration_s=idle_s,
            )
            workload = _measure_condition(
                model=model,
                inputs=batch.inputs,
                targets=batch.targets,
                selected=selected,
                batch_size=batch_size,
                device=device,
                device_index=device_index,
                interval_s=interval_s,
                minimum_workload_s=minimum_s,
            )

            meter = workload["meter"]
            passes = int(workload["logical_passes"])
            gross_j = float(meter["instant_energy_j"])
            duration_s = float(meter["duration_s"])
            idle_mean_w = float(idle["mean_instant_w"])
            idle_equivalent_j = idle_mean_w * duration_s
            incremental_j = gross_j - idle_equivalent_j

            accounting = {
                "gross_gpu_energy_j": gross_j,
                "matched_idle_mean_power_w": idle_mean_w,
                "matched_idle_equivalent_energy_j": idle_equivalent_j,
                "incremental_gpu_energy_j": incremental_j,
                "gross_gpu_energy_per_logical_pass_j": gross_j / passes,
                "incremental_gpu_energy_per_logical_pass_j": (
                    incremental_j / passes
                ),
                "unmetered_zero_invocation_condition": False,
            }

            records.append(
                {
                    "repetition": repetition,
                    "condition": condition,
                    "idle": idle,
                    "workload": workload,
                    "accounting": accounting,
                }
            )

            print(
                "    "
                f"{condition}: inv/pass={selected.size}, "
                f"gross/pass={accounting['gross_gpu_energy_per_logical_pass_j']:.6f} J, "
                f"incr/pass={accounting['incremental_gpu_energy_per_logical_pass_j']:.6f} J"
            )

    by_rep: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_rep.setdefault(int(record["repetition"]), {})[
            str(record["condition"])
        ] = record

    reductions = {}
    for condition in conditions[1:]:
        gross_values: list[float] = []
        incremental_values: list[float | None] = []

        for repetition in range(1, repetitions + 1):
            baseline = by_rep[repetition]["always_on_tcn"]
            candidate = by_rep[repetition][condition]

            baseline_gross = float(
                baseline["accounting"][
                    "gross_gpu_energy_per_logical_pass_j"
                ]
            )
            candidate_gross = float(
                candidate["accounting"][
                    "gross_gpu_energy_per_logical_pass_j"
                ]
            )
            gross_values.append(
                energy_reduction_fraction(
                    baseline_gross,
                    candidate_gross,
                )
            )

            baseline_incremental = float(
                baseline["accounting"][
                    "incremental_gpu_energy_per_logical_pass_j"
                ]
            )
            candidate_incremental = float(
                candidate["accounting"][
                    "incremental_gpu_energy_per_logical_pass_j"
                ]
            )
            incremental_values.append(
                energy_reduction_fraction(
                    baseline_incremental,
                    candidate_incremental,
                )
                if baseline_incremental > 0.0
                else None
            )

        valid_incremental = [
            value for value in incremental_values if value is not None
        ]
        reductions[condition] = {
            "gross_reduction_by_repetition": gross_values,
            "mean_gross_reduction_fraction": statistics.fmean(gross_values),
            "gross_reduction_positive_each_repetition": all(
                value > 0.0 for value in gross_values
            ),
            "incremental_reduction_by_repetition": incremental_values,
            "mean_incremental_reduction_fraction": (
                statistics.fmean(valid_incremental)
                if valid_incremental
                else None
            ),
        }

    selected_reduction = reductions["frozen_evidence_aware_router"]
    selected_metrics = evaluation_result["systems"][
        "frozen_evidence_aware_router"
    ]

    final_status = final_external_status(
        evidence_status=str(evaluation_result["evidence_stage_status"]),
        evidence_aware_metrics=selected_metrics,
        mean_gross_energy_reduction=float(
            selected_reduction["mean_gross_reduction_fraction"]
        ),
        gross_reductions=[
            float(value)
            for value in selected_reduction["gross_reduction_by_repetition"]
        ],
        success_criteria=study["success_criteria"],
    )

    payload = {
        "schema_version": "aeroxai.rcsd1yd_external_evaluation_result.v1",
        "study": study["study"]["name"],
        "evidence_role": "genuinely_new_chronological_external_evaluation",
        "evaluation_result_file": str(evaluation_result_path.relative_to(ROOT)),
        "evaluation": evaluation_result,
        "energy": {
            "measurement_scope": "gpu_device_tcn_scoring_only",
            "cpu_router_energy": "UNKNOWN",
            "whole_system_energy": "UNKNOWN",
            "physical_compressor_energy": "NOT_MEASURED",
            "protocol": preflight["energy_protocol"],
            "records": records,
            "reductions_vs_always_on_tcn": reductions,
        },
        "final_external_status": final_status,
        "success_criteria": study["success_criteria"],
        "interpretation": study["interpretation"],
        "post_evaluation_tuning_allowed": False,
    }

    final_result_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "evidence_stage_status": evaluation_result[
                    "evidence_stage_status"
                ],
                "selected_system_metrics": selected_metrics,
                "selected_energy_reduction": selected_reduction,
                "final_external_status": final_status,
                "post_evaluation_tuning_allowed": False,
                "output": str(final_result_path.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
