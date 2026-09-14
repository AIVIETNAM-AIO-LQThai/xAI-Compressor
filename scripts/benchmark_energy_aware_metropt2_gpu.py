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

from ml.data.metropt2_features import (
    model_feature_columns as metropt2_feature_columns,
)
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
from ml.temporal.tcn import TCNForecaster, TemporalForecastConfig

ROOT = Path(__file__).resolve().parents[1]
TRANSPORT_CONFIG = (
    ROOT / "configs" / "energy_aware_metropt2_transport.yaml"
)
BENCHMARK_CONFIG = (
    ROOT / "configs" / "energy_aware_inference_benchmark.yaml"
)


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


def _build_model(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
) -> TCNForecaster:
    config = checkpoint["model_config"]
    model = TCNForecaster(
        TemporalForecastConfig(
            input_dim=int(config["input_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            kernel_size=int(config["kernel_size"]),
            dilations=tuple(int(v) for v in config["dilations"]),
            dropout=float(config["dropout"]),
        )
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


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
            indices = selected[start : start + batch_size]
            x = torch.from_numpy(inputs[indices]).to(device)
            y = torch.from_numpy(targets[indices]).to(device)
            prediction = model(x)
            score = torch.mean((y - prediction).square(), dim=1)
            checksum += float(score.detach().cpu().sum().item())

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
    return meter.stop().to_dict()


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
    started = perf_counter()

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

        if perf_counter() - started >= minimum_workload_s:
            break

    summary = meter.stop()

    normalized_ms = [
        1000.0 * value / int(selected.size)
        for value in pass_durations
    ]

    return {
        "meter": summary.to_dict(),
        "logical_passes": passes,
        "invocations_per_logical_pass": int(selected.size),
        "total_tcn_invocations": int(passes * selected.size),
        "logical_pass_seconds_summary": {
            "minimum": float(min(pass_durations)),
            "mean": float(statistics.fmean(pass_durations)),
            "median": float(statistics.median(pass_durations)),
            "p95": percentile(pass_durations, 0.95),
            "maximum": float(max(pass_durations)),
        },
        "throughput_normalized_ms_per_tcn_window_summary": {
            "mean": float(statistics.fmean(normalized_ms)),
            "median": float(statistics.median(normalized_ms)),
            "p95": percentile(normalized_ms, 0.95),
        },
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device))
            / (1024.0 * 1024.0)
            if device.type == "cuda"
            else None
        ),
        "checksum": checksum,
    }


def main() -> None:
    transport = _load_yaml(TRANSPORT_CONFIG)
    benchmark = _load_yaml(BENCHMARK_CONFIG)

    data_cfg = transport["data"]
    feature_cfg = transport["feature_schema"]
    outputs = transport["outputs"]
    measurement = benchmark["energy_measurement"]
    tcn_primary = benchmark["tcn"]

    result = _load_json(ROOT / outputs["transport_result"])

    if result["test_tuning_performed"] is not False:
        raise RuntimeError("Transport result indicates TEST tuning.")
    if result["prior_pca_reproduction"]["passed"] is not True:
        raise RuntimeError("Prior PCA reproduction guard did not pass.")

    checkpoint_path = ROOT / outputs["checkpoint"]
    observed_sha = sha256_file(checkpoint_path)
    expected_sha = str(result["tcn"]["checkpoint_sha256"])
    if observed_sha != expected_sha:
        raise RuntimeError(
            "MetroPT2 TCN checkpoint SHA mismatch: "
            f"{observed_sha} != {expected_sha}."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint.get("scaler_name") != "standard":
        raise RuntimeError("MetroPT2 transport TCN must use StandardScaler.")

    train = _load_partition(ROOT / data_cfg["train_file"])
    test = _load_partition(ROOT / data_cfg["test_file"])

    features = metropt2_feature_columns(
        train.columns,
        excluded_groups=feature_cfg["excluded_groups"],
    )
    if features != checkpoint["features"]:
        raise RuntimeError("Transport checkpoint feature identity mismatch.")

    scaler = fit_scaler(train, features, method="standard")
    if not np.allclose(
        scaler.mean_,
        np.asarray(checkpoint["scaler_location"], dtype=float),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("Transport StandardScaler mean mismatch.")
    if not np.allclose(
        scaler.scale_,
        np.asarray(checkpoint["scaler_scale"], dtype=float),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError("Transport StandardScaler scale mismatch.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.type != "cuda":
        raise RuntimeError("Measured GPU benchmark requires CUDA.")

    model = _build_model(checkpoint, device=device)

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
        variance_retained=0.95,
        scaler_name="robust",
    )
    pca_test_raw = score_pca_detector(test, pca)
    pca_test_ewma = causal_ewma(
        pca_test_raw,
        alpha=0.20,
        reset_gap_minutes=10,
    ).reindex(temporal_batch.target_index)

    if pca_test_ewma.isna().any():
        raise RuntimeError("Transport PCA router alignment contains NaN.")

    all_indices = np.arange(
        len(temporal_batch.inputs),
        dtype=np.int64,
    )
    selected_by_condition: dict[str, np.ndarray] = {
        "always_on_tcn": all_indices
    }

    for condition in ("route_q90", "route_q95", "route_q99"):
        threshold = float(
            result["router_calibration"][condition]["threshold"]
        )
        routed = routing_mask(pca_test_ewma, threshold)
        selected = np.flatnonzero(
            routed.to_numpy(dtype=bool)
        ).astype(np.int64, copy=False)

        expected_invocations = int(
            result["routing"][condition]["tcn_invocations"]
        )
        if int(selected.size) != expected_invocations:
            raise RuntimeError(
                f"{condition} invocation mismatch: "
                f"{selected.size} != {expected_invocations}."
            )
        selected_by_condition[condition] = selected

    conditions = [
        str(value)
        for value in measurement["condition_order"]["conditions"]
    ]
    repetitions = int(measurement["measured_repetitions"])
    orders = seeded_condition_orders(
        conditions,
        repetitions=repetitions,
        seed=int(measurement["condition_order"]["seed"]),
    )

    batch_size = int(tcn_primary["inference_batch_size"])
    warmups = int(measurement["warmup_runs_per_condition"])

    for condition in conditions:
        for _ in range(warmups):
            _run_logical_pass(
                model=model,
                inputs=temporal_batch.inputs,
                targets=temporal_batch.targets,
                selected=selected_by_condition[condition],
                batch_size=batch_size,
                device=device,
            )

    paired = measurement["paired_measurement"]
    cooldown_s = float(paired["cooldown_seconds"])
    idle_s = float(paired["matched_idle_seconds"])
    minimum_workload_s = float(paired["minimum_workload_seconds"])
    interval_s = float(measurement["sampling_interval_seconds"])
    device_index = int(measurement["gpu_device_index"])

    records: list[dict[str, Any]] = []

    print("MetroPT2 measured repetitions")
    for repetition, order in enumerate(orders, start=1):
        print(f"  repetition {repetition}/{repetitions}: {order}")

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
            idle_equivalent_j = idle_mean_w * duration_s
            incremental_j = gross_j - idle_equivalent_j

            record = {
                "repetition": repetition,
                "condition": condition,
                "idle": idle,
                "workload": workload,
                "accounting": {
                    "gross_gpu_energy_j": gross_j,
                    "matched_idle_mean_power_w": idle_mean_w,
                    "matched_idle_equivalent_energy_j": idle_equivalent_j,
                    "incremental_gpu_energy_j": incremental_j,
                    "gross_gpu_energy_per_logical_pass_j": (
                        gross_j / passes
                    ),
                    "incremental_gpu_energy_per_logical_pass_j": (
                        incremental_j / passes
                    ),
                    "gross_gpu_energy_per_tcn_invocation_j": (
                        gross_j
                        / int(workload["total_tcn_invocations"])
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
                "gross/pass="
                f"{record['accounting']['gross_gpu_energy_per_logical_pass_j']:.6f} J, "
                "incr/pass="
                f"{record['accounting']['incremental_gpu_energy_per_logical_pass_j']:.6f} J"
            )

    by_rep: dict[int, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_rep.setdefault(int(record["repetition"]), {})[
            str(record["condition"])
        ] = record

    reductions: dict[str, Any] = {}
    materiality = float(
        measurement["materiality"][
            "minimum_mean_gpu_energy_reduction_fraction"
        ]
    )

    for condition in ("route_q90", "route_q95", "route_q99"):
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
        mean_gross = float(statistics.fmean(gross_values))

        reductions[condition] = {
            "gross_reduction_by_repetition": gross_values,
            "mean_gross_reduction_fraction": mean_gross,
            "sample_sd_gross_reduction_fraction": (
                float(statistics.stdev(gross_values))
                if len(gross_values) > 1
                else None
            ),
            "gross_reduction_directionally_consistent": all(
                value > 0.0 for value in gross_values
            ),
            "incremental_reduction_by_repetition": incremental_values,
            "mean_incremental_reduction_fraction": (
                float(statistics.fmean(valid_incremental))
                if valid_incremental
                else None
            ),
            "meets_3_percent_materiality": mean_gross >= materiality,
        }

    report = {
        "schema_version": "aeroxai.energy_aware_metropt2_energy.v1",
        "study": transport["transport"]["name"],
        "dataset": "MetroPT2",
        "evidence_class": "PREVIOUSLY_INSPECTED_CROSS_DATASET_TRANSPORT",
        "independent_confirmation": False,
        "causal_claim": False,
        "test_tuning_performed": False,
        "measurement_scope": "gpu_device_tcn_scoring_only",
        "cpu_router_energy": "UNKNOWN",
        "whole_system_energy": "UNKNOWN",
        "tcn_checkpoint_sha256": expected_sha,
        "energy_protocol": {
            "sampling_interval_seconds": interval_s,
            "warmup_runs_per_condition": warmups,
            "measured_repetitions": repetitions,
            "cooldown_seconds": cooldown_s,
            "matched_idle_seconds": idle_s,
            "minimum_workload_seconds": minimum_workload_s,
            "condition_orders": orders,
            "materiality_fraction": materiality,
            "primary_metric": measurement["materiality"]["primary_metric"],
        },
        "records": records,
        "energy_reduction_vs_always_on_tcn": reductions,
        "routing_evidence_coverage": result["routing"],
        "guardrails": transport["guardrails"],
    }

    output_path = ROOT / outputs["energy_result"]
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "dataset": "MetroPT2",
                "measurement_scope": report["measurement_scope"],
                "cpu_router_energy": "UNKNOWN",
                "whole_system_energy": "UNKNOWN",
                "energy_reduction_vs_always_on_tcn": reductions,
                "output": str(output_path.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
