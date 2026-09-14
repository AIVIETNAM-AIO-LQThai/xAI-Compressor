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

from ml.data.features import model_feature_columns
from ml.detection.alerts import causal_ewma
from ml.detection.common import fit_scaler
from ml.detection.pca_detector import (
    fit_pca_detector,
    score_pca_detector,
)
from ml.energy.gpu_meter import NVMLGpuMeter
from ml.energy.hierarchical_inference import (
    energy_reduction_fraction,
    routing_mask,
    seeded_condition_orders,
    sha256_file,
)
from ml.energy.primary_benchmark import percentile
from ml.temporal.detector import prepare_temporal_batch
from ml.temporal.tcn import (
    TCNForecaster,
    TemporalForecastConfig,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY_CONFIG = ROOT / "configs" / "energy_aware_hierarchical_intelligence.yaml"
BENCHMARK_CONFIG = ROOT / "configs" / "energy_aware_inference_benchmark.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return payload


def _load_partition(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index, name="timestamp")
    return frame.sort_index()


def _build_tcn(
    *,
    train: pd.DataFrame,
    features: list[str],
    checkpoint_path: Path,
    expected_sha: str,
    device: torch.device,
) -> tuple[
    TCNForecaster,
    Any,
    dict[str, Any],
]:
    observed_sha = sha256_file(checkpoint_path)
    if observed_sha != expected_sha:
        raise RuntimeError(
            "TCN checkpoint SHA256 mismatch: "
            f"expected {expected_sha}, got {observed_sha}."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("scaler_name") != "standard":
        raise RuntimeError("Frozen TCN checkpoint must use StandardScaler.")

    scaler = fit_scaler(
        train,
        features,
        method="standard",
    )

    if not np.allclose(
        scaler.mean_,
        np.asarray(checkpoint["scaler_location"], dtype=float),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("StandardScaler mean mismatch.")
    if not np.allclose(
        scaler.scale_,
        np.asarray(checkpoint["scaler_scale"], dtype=float),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("StandardScaler scale mismatch.")

    model_cfg = checkpoint["model_config"]
    model = TCNForecaster(
        TemporalForecastConfig(
            input_dim=int(model_cfg["input_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            kernel_size=int(model_cfg["kernel_size"]),
            dilations=tuple(
                int(value)
                for value in model_cfg["dilations"]
            ),
            dropout=float(model_cfg["dropout"]),
        )
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()

    return model, scaler, checkpoint


def _run_logical_pass(
    *,
    model: TCNForecaster,
    inputs: np.ndarray,
    targets: np.ndarray,
    selected: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float]:
    if selected.size == 0:
        raise RuntimeError("Condition has zero TCN invocations.")

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    started = perf_counter()
    checksum = 0.0

    with torch.inference_mode():
        for start in range(0, selected.size, batch_size):
            batch_indices = selected[start : start + batch_size]

            x = torch.from_numpy(
                inputs[batch_indices]
            ).to(device)
            y = torch.from_numpy(
                targets[batch_indices]
            ).to(device)

            prediction = model(x)
            scores = torch.mean(
                (y - prediction).square(),
                dim=1,
            )

            checksum += float(
                scores.detach().cpu().sum().item()
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return perf_counter() - started, checksum


def _measure_idle(
    *,
    device_index: int,
    interval_s: float,
    duration_s: float,
) -> dict[str, float | int]:
    meter = NVMLGpuMeter(
        device_index=device_index,
        interval_s=interval_s,
    )
    meter.start()
    time.sleep(duration_s)
    summary = meter.stop()
    return summary.to_dict()


def _measure_condition(
    *,
    model: TCNForecaster,
    inputs: np.ndarray,
    targets: np.ndarray,
    selected: np.ndarray,
    batch_size: int,
    device: torch.device,
    device_index: int,
    interval_s: float,
    minimum_workload_s: float,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    meter = NVMLGpuMeter(
        device_index=device_index,
        interval_s=interval_s,
    )

    pass_durations: list[float] = []
    checksum = 0.0
    passes = 0

    meter.start()
    workload_started = perf_counter()

    while True:
        duration, pass_checksum = _run_logical_pass(
            model=model,
            inputs=inputs,
            targets=targets,
            selected=selected,
            batch_size=batch_size,
            device=device,
        )
        pass_durations.append(duration)
        checksum += pass_checksum
        passes += 1

        if (
            perf_counter() - workload_started
            >= minimum_workload_s
        ):
            break

    summary = meter.stop()

    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated(device))
        / (1024.0 * 1024.0)
        if device.type == "cuda"
        else None
    )

    invocations_per_pass = int(selected.size)
    total_invocations = passes * invocations_per_pass
    normalized_ms = [
        1000.0 * value / invocations_per_pass
        for value in pass_durations
    ]

    return {
        "meter": summary.to_dict(),
        "logical_passes": passes,
        "invocations_per_logical_pass": invocations_per_pass,
        "total_tcn_invocations": total_invocations,
        "logical_pass_seconds": pass_durations,
        "throughput_normalized_ms_per_tcn_window": normalized_ms,
        "median_throughput_normalized_ms_per_tcn_window": (
            statistics.median(normalized_ms)
        ),
        "p95_throughput_normalized_ms_per_tcn_window": percentile(
            normalized_ms,
            0.95,
        ),
        "peak_gpu_memory_mb": peak_memory_mb,
        "checksum": checksum,
    }


def main() -> None:
    study = _load_yaml(STUDY_CONFIG)
    benchmark = _load_yaml(BENCHMARK_CONFIG)

    primary = study["datasets"]["primary"]
    tcn_cfg = benchmark["tcn"]
    router_cfg = benchmark["router"]
    measurement = benchmark["energy_measurement"]
    outputs = benchmark["outputs"]

    primary_evidence = _load_json(
        ROOT / outputs["primary_evidence_file"]
    )
    router_calibration = _load_json(
        ROOT / outputs["router_calibration_file"]
    )
    preflight = _load_json(
        ROOT / outputs["pretest_preflight_file"]
    )

    if (
        primary_evidence["frozen_pca_reproduction"]["passed"]
        is not True
    ):
        raise RuntimeError(
            "Frozen PCA reproduction did not pass; energy benchmark aborted."
        )

    expected_sha = str(tcn_cfg["checkpoint_sha256"])
    for observed in (
        primary_evidence["tcn_checkpoint"]["sha256"],
        router_calibration["tcn_checkpoint"]["sha256"],
        preflight["tcn_checkpoint"]["sha256"],
    ):
        if observed != expected_sha:
            raise RuntimeError(
                "Benchmark artifact checkpoint SHA mismatch."
            )

    if measurement["scope"] != "gpu_device":
        raise RuntimeError("Energy scope must remain gpu_device.")

    train = _load_partition(ROOT / primary["train_file"])
    test = _load_partition(ROOT / primary["test_file"])

    features = model_feature_columns(train.columns)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "Measured GPU benchmark requires CUDA."
        )

    model, scaler, checkpoint = _build_tcn(
        train=train,
        features=features,
        checkpoint_path=ROOT / tcn_cfg["checkpoint_file"],
        expected_sha=expected_sha,
        device=device,
    )

    sequence = checkpoint["sequence"]
    temporal_batch = prepare_temporal_batch(
        test,
        features=features,
        scaler=scaler,
        sequence_length=int(sequence["history_bins"]),
        bin_minutes=int(sequence["bin_minutes"]),
    )

    pca = fit_pca_detector(
        train,
        features,
        variance_retained=float(
            router_cfg["variance_retained"]
        ),
        scaler_name="robust",
    )
    pca_test_raw = score_pca_detector(test, pca)
    pca_test_ewma = causal_ewma(
        pca_test_raw,
        alpha=float(router_cfg["ewma_alpha"]),
        reset_gap_minutes=int(
            router_cfg["reset_gap_minutes"]
        ),
    ).reindex(temporal_batch.target_index)

    if pca_test_ewma.isna().any():
        raise RuntimeError(
            "PCA router alignment to TCN windows contains NaN."
        )

    all_indices = np.arange(
        len(temporal_batch.inputs),
        dtype=np.int64,
    )

    selected_by_condition: dict[str, np.ndarray] = {
        "always_on_tcn": all_indices,
    }

    frozen_points = router_calibration["router"]["operating_points"]

    for condition in ("route_q90", "route_q95", "route_q99"):
        routed = routing_mask(
            pca_test_ewma,
            float(frozen_points[condition]["threshold"]),
        )
        selected = np.flatnonzero(
            routed.to_numpy(dtype=bool)
        ).astype(np.int64, copy=False)

        expected_invocations = int(
            primary_evidence["routing"][condition][
                "tcn_invocations"
            ]
        )
        if int(selected.size) != expected_invocations:
            raise RuntimeError(
                f"{condition} invocation-count mismatch: "
                f"{selected.size} != {expected_invocations}."
            )

        selected_by_condition[condition] = selected

    conditions = [
        str(value)
        for value in measurement["condition_order"]["conditions"]
    ]
    repetitions = int(measurement["measured_repetitions"])
    expected_orders = seeded_condition_orders(
        conditions,
        repetitions=repetitions,
        seed=int(measurement["condition_order"]["seed"]),
    )

    if expected_orders != preflight["measurement"]["condition_orders"]:
        raise RuntimeError("Condition order differs from pre-TEST preflight.")

    batch_size = int(tcn_cfg["inference_batch_size"])

    # Frozen two warm-up logical passes per condition.
    warmup_runs = int(
        measurement["warmup_runs_per_condition"]
    )
    for condition in conditions:
        selected = selected_by_condition[condition]
        for _ in range(warmup_runs):
            _run_logical_pass(
                model=model,
                inputs=temporal_batch.inputs,
                targets=temporal_batch.targets,
                selected=selected,
                batch_size=batch_size,
                device=device,
            )

    device_index = int(measurement["gpu_device_index"])
    interval_s = float(measurement["sampling_interval_seconds"])
    paired = measurement["paired_measurement"]

    cooldown_s = float(paired["cooldown_seconds"])
    idle_s = float(paired["matched_idle_seconds"])
    minimum_workload_s = float(
        paired["minimum_workload_seconds"]
    )

    records: list[dict[str, Any]] = []

    print("Measured repetitions")
    for repetition_index, order in enumerate(expected_orders, start=1):
        print(f"  repetition {repetition_index}/{repetitions}: {order}")

        for condition in order:
            time.sleep(cooldown_s)

            idle = _measure_idle(
                device_index=device_index,
                interval_s=interval_s,
                duration_s=idle_s,
            )

            workload = _measure_condition(
                model=model,
                inputs=temporal_batch.inputs,
                targets=temporal_batch.targets,
                selected=selected_by_condition[condition],
                batch_size=batch_size,
                device=device,
                device_index=device_index,
                interval_s=interval_s,
                minimum_workload_s=minimum_workload_s,
            )

            meter = workload["meter"]
            passes = int(workload["logical_passes"])
            gross_j = float(meter["instant_energy_j"])
            duration_s = float(meter["duration_s"])
            idle_mean_w = float(idle["mean_instant_w"])
            baseline_j = idle_mean_w * duration_s
            incremental_j = gross_j - baseline_j

            gross_per_pass = gross_j / passes
            incremental_per_pass = incremental_j / passes

            record = {
                "repetition": repetition_index,
                "condition": condition,
                "idle": idle,
                "workload": workload,
                "accounting": {
                    "gross_gpu_energy_j": gross_j,
                    "matched_idle_mean_power_w": idle_mean_w,
                    "matched_idle_equivalent_energy_j": baseline_j,
                    "incremental_gpu_energy_j": incremental_j,
                    "gross_gpu_energy_per_logical_pass_j": gross_per_pass,
                    "incremental_gpu_energy_per_logical_pass_j": (
                        incremental_per_pass
                    ),
                    "gross_gpu_energy_per_tcn_invocation_j": (
                        gross_j / int(workload["total_tcn_invocations"])
                    ),
                    "incremental_gpu_energy_per_tcn_invocation_j": (
                        incremental_j
                        / int(workload["total_tcn_invocations"])
                    ),
                },
            }
            records.append(record)

            print(
                "    "
                f"{condition}: passes={passes}, "
                f"inv/pass={workload['invocations_per_logical_pass']}, "
                f"gross/pass={gross_per_pass:.6f} J, "
                f"incr/pass={incremental_per_pass:.6f} J"
            )

    by_rep: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_rep.setdefault(
            int(record["repetition"]),
            {},
        )[str(record["condition"])] = record

    reductions: dict[str, dict[str, Any]] = {}

    for condition in ("route_q90", "route_q95", "route_q99"):
        gross_values: list[float] = []
        incremental_values: list[float | None] = []

        for repetition_index in range(1, repetitions + 1):
            baseline = by_rep[repetition_index]["always_on_tcn"]
            candidate = by_rep[repetition_index][condition]

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
            value
            for value in incremental_values
            if value is not None
        ]
        mean_gross = statistics.fmean(gross_values)

        reductions[condition] = {
            "gross_reduction_by_repetition": gross_values,
            "mean_gross_reduction_fraction": mean_gross,
            "gross_reduction_directionally_consistent": all(
                value > 0.0
                for value in gross_values
            ),
            "incremental_reduction_by_repetition": incremental_values,
            "mean_incremental_reduction_fraction": (
                statistics.fmean(valid_incremental)
                if valid_incremental
                else None
            ),
            "meets_3_percent_materiality": (
                mean_gross
                >= float(
                    measurement["materiality"][
                        "minimum_mean_gpu_energy_reduction_fraction"
                    ]
                )
            ),
            "meets_directional_consistency": all(
                value > 0.0
                for value in gross_values
            ),
        }

    report = {
        "schema_version": "aeroxai.energy_aware_hierarchical_routing.v1",
        "study": study["study"]["name"],
        "evidence_class": "EXPLORATORY_REUSED_TEST_BENCHMARK",
        "causal_claim": False,
        "measurement_scope": "gpu_device",
        "cpu_energy": "UNKNOWN",
        "whole_system_energy": "UNKNOWN",
        "tcn_checkpoint_sha256": expected_sha,
        "primary_evidence_file": outputs["primary_evidence_file"],
        "energy_protocol": {
            "sampling_interval_seconds": interval_s,
            "warmup_runs_per_condition": warmup_runs,
            "measured_repetitions": repetitions,
            "cooldown_seconds": cooldown_s,
            "matched_idle_seconds": idle_s,
            "minimum_workload_seconds": minimum_workload_s,
            "condition_orders": expected_orders,
            "primary_materiality_metric": measurement[
                "materiality"
            ]["primary_metric"],
            "materiality_fraction": float(
                measurement["materiality"][
                    "minimum_mean_gpu_energy_reduction_fraction"
                ]
            ),
        },
        "records": records,
        "energy_reduction_vs_always_on_tcn": reductions,
        "routing_evidence_coverage": primary_evidence["routing"],
        "operational_metrics_source": "frozen_robust_pca",
        "operational_metrics": primary_evidence[
            "frozen_pca_reproduction"
        ]["metrics"],
    }

    output_path = ROOT / outputs["primary_result_file"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "measurement_scope": "gpu_device",
                "cpu_energy": "UNKNOWN",
                "whole_system_energy": "UNKNOWN",
                "energy_reduction_vs_always_on_tcn": reductions,
                "output": str(output_path.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
