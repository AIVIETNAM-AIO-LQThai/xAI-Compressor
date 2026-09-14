from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ml.data.metropt2_features import (
    model_feature_columns as metropt2_feature_columns,
)
from ml.detection.alerts import (
    causal_ewma,
    extract_alert_episodes,
    persistent_alerts,
)
from ml.detection.common import fit_scaler, transform_frame
from ml.detection.evaluate import evaluate_detection
from ml.detection.pca_detector import (
    fit_pca_detector,
    score_pca_detector,
)
from ml.energy.hierarchical_inference import (
    higher_quantile,
    routing_mask,
    sha256_file,
)
from ml.energy.primary_benchmark import (
    bin_coverage,
    episode_coverage,
    incident_related_mask,
    serializable_episodes,
    verify_frozen_metrics,
)
from ml.temporal.detector import (
    TemporalDetector,
    prepare_temporal_batch,
    score_temporal_detector,
)
from ml.temporal.energy_aware_training import (
    EnergyAwareTrainingConfig,
    train_energy_aware_tcn,
)
from ml.temporal.tcn import TemporalForecastConfig

ROOT = Path(__file__).resolve().parents[1]
TRANSPORT_CONFIG = (
    ROOT / "configs" / "energy_aware_metropt2_transport.yaml"
)
STUDY_CONFIG = (
    ROOT / "configs" / "energy_aware_hierarchical_intelligence.yaml"
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


def _scaled_magnitude_summary(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    return {
        "p50_abs": float(np.quantile(absolute, 0.50)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "p999_abs": float(np.quantile(absolute, 0.999)),
        "max_abs": float(absolute.max()),
    }


def _feature_error_concentration(
    frame: pd.DataFrame,
    *,
    detector: TemporalDetector,
    device: str,
    batch_size: int = 1024,
) -> dict[str, Any]:
    batch = prepare_temporal_batch(
        frame,
        features=detector.features,
        scaler=detector.scaler,
        sequence_length=detector.sequence_length,
        bin_minutes=detector.bin_minutes,
    )

    torch_device = torch.device(device)
    model = detector.model.to(torch_device).eval()

    feature_sse = np.zeros(
        len(detector.features),
        dtype=np.float64,
    )
    rows = 0

    with torch.inference_mode():
        for start in range(0, len(batch.inputs), batch_size):
            stop = min(len(batch.inputs), start + batch_size)
            x = torch.from_numpy(batch.inputs[start:stop]).to(torch_device)
            y = torch.from_numpy(batch.targets[start:stop]).to(torch_device)
            prediction = model(x)
            squared = (
                (y - prediction)
                .square()
                .detach()
                .cpu()
                .numpy()
            )
            feature_sse += squared.sum(axis=0, dtype=np.float64)
            rows += squared.shape[0]

    if rows == 0:
        raise RuntimeError("No calibration sequences available.")

    feature_mse = feature_sse / rows
    total = float(feature_mse.sum())
    order = np.argsort(feature_mse)[::-1]

    return {
        "top_1_share": float(feature_mse[order[:1]].sum() / total),
        "top_3_share": float(feature_mse[order[:3]].sum() / total),
        "top_5_share": float(feature_mse[order[:5]].sum() / total),
        "top_features": [
            {
                "feature": detector.features[int(index)],
                "mse": float(feature_mse[int(index)]),
                "share": float(feature_mse[int(index)] / total),
            }
            for index in order[:10]
        ],
    }


def _require_equal(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise RuntimeError(
            f"Frozen transport mismatch for {label}: "
            f"observed={value!r}, expected={expected!r}."
        )


def main() -> None:
    transport = _load_yaml(TRANSPORT_CONFIG)
    study = _load_yaml(STUDY_CONFIG)
    benchmark = _load_yaml(BENCHMARK_CONFIG)

    transport_root = transport["transport"]
    feature_cfg = transport["feature_schema"]
    data_cfg = transport["data"]
    method = transport["method"]
    integrity = transport["integrity"]
    outputs = transport["outputs"]

    _require_equal(
        transport_root["retuning_allowed"],
        False,
        "retuning_allowed",
    )
    _require_equal(
        feature_cfg["variant"],
        "flowmeter_excluded",
        "feature variant",
    )
    _require_equal(
        int(feature_cfg["expected_feature_count"]),
        63,
        "feature count",
    )

    metropt2 = _load_yaml(ROOT / data_cfg["config"])
    preprocessing_report = _load_json(
        ROOT / data_cfg["preprocessing_report"]
    )
    prior_pca = _load_json(
        ROOT / data_cfg["prior_pca_benchmark"]
    )
    primary_tcn_calibration = _load_json(
        ROOT / method["primary_feature_reference"]
    )

    temporal = study["temporal_model"]
    sequence = temporal["sequence"]
    architecture = temporal["architecture"]
    training = temporal["training"]
    temporal_calibration = temporal["calibration"]
    router_primary = benchmark["router"]
    tcn_evidence_primary = benchmark["tcn_evidence"]

    # Freeze the transport to the already-frozen primary definitions.
    _require_equal(temporal["scaler"], "standard", "TCN scaler")
    _require_equal(
        method["tcn"]["scaler"],
        temporal["scaler"],
        "transport TCN scaler",
    )
    _require_equal(
        float(method["tcn"]["calibration_threshold_quantile"]),
        float(temporal_calibration["threshold_quantile"]),
        "TCN calibration quantile",
    )
    _require_equal(
        method["tcn"]["calibration_quantile_interpolation"],
        temporal_calibration["quantile_interpolation"],
        "TCN interpolation",
    )
    _require_equal(
        method["evidence"]["tcn_score_smoothing"],
        tcn_evidence_primary["score_smoothing"],
        "TCN score smoothing",
    )
    _require_equal(
        method["evidence"]["tcn_persistence"],
        tcn_evidence_primary["persistence"],
        "TCN persistence",
    )
    _require_equal(
        int(method["evidence"]["episode_merge_minutes"]),
        int(tcn_evidence_primary["episode_merge_minutes"]),
        "episode merge",
    )
    _require_equal(
        int(method["evidence"]["episode_reset_gap_minutes"]),
        int(tcn_evidence_primary["episode_reset_gap_minutes"]),
        "episode reset gap",
    )

    for operating_id, quantile in method["router"][
        "operating_quantiles"
    ].items():
        _require_equal(
            float(quantile),
            float(router_primary["operating_points"][operating_id]),
            f"{operating_id} quantile",
        )

    # TRAIN + CALIBRATION are opened first; TEST is deliberately delayed until
    # training, TCN calibration, and router calibration are complete.
    train = _load_partition(ROOT / data_cfg["train_file"])
    calibration = _load_partition(ROOT / data_cfg["calibration_file"])

    expected_rows = data_cfg["expected_rows"]
    if len(train) != int(expected_rows["train"]):
        raise RuntimeError("MetroPT2 TRAIN row-count identity failed.")
    if len(calibration) != int(expected_rows["calibration"]):
        raise RuntimeError("MetroPT2 CALIBRATION row-count identity failed.")

    if len(train) != int(preprocessing_report["splits"]["train"]["valid_bins"]):
        raise RuntimeError("TRAIN no longer matches preprocessing report.")
    if len(calibration) != int(
        preprocessing_report["splits"]["calibration"]["valid_bins"]
    ):
        raise RuntimeError("CALIBRATION no longer matches preprocessing report.")

    features = metropt2_feature_columns(
        train.columns,
        excluded_groups=feature_cfg["excluded_groups"],
    )

    primary_features = [
        str(value)
        for value in primary_tcn_calibration["features"]["names"]
    ]

    if features != primary_features:
        raise RuntimeError(
            "MetroPT2 flowmeter_excluded feature names/order do not "
            "exactly match the primary 63-feature representation."
        )

    if len(features) != 63:
        raise RuntimeError(
            f"Expected 63 transport features, got {len(features)}."
        )

    # --- TCN TRAINING: same architecture/hyperparameters, MetroPT2 TRAIN only.
    scaler = fit_scaler(
        train,
        features,
        method="standard",
    )

    train_scaled = transform_frame(train, features, scaler)
    calibration_scaled = transform_frame(calibration, features, scaler)

    train_batch = prepare_temporal_batch(
        train,
        features=features,
        scaler=scaler,
        sequence_length=int(sequence["history_bins"]),
        bin_minutes=int(sequence["bin_minutes"]),
    )

    channels = [int(value) for value in architecture["channels"]]
    if channels != [32, 32, 32]:
        raise RuntimeError("Primary TCN channels are no longer [32,32,32].")
    if len(set(channels)) != 1:
        raise RuntimeError("TCNForecaster requires equal hidden channels.")

    model_config = TemporalForecastConfig(
        input_dim=len(features),
        hidden_dim=channels[0],
        kernel_size=int(architecture["kernel_size"]),
        dilations=tuple(int(value) for value in architecture["dilations"]),
        dropout=float(architecture["dropout"]),
    )

    training_config = EnergyAwareTrainingConfig(
        seed=int(training["seed"]),
        max_epochs=int(training["max_epochs"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        validation_fraction=float(training["validation_fraction_of_train"]),
        patience=int(training["early_stopping_patience"]),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    result = train_energy_aware_tcn(
        train_batch,
        model_config=model_config,
        training_config=training_config,
    )

    detector = TemporalDetector(
        features=features,
        scaler=scaler,
        model=result.model,
        sequence_length=int(sequence["history_bins"]),
        bin_minutes=int(sequence["bin_minutes"]),
    )

    calibration_scores = score_temporal_detector(
        calibration,
        detector,
        device=result.device,
        batch_size=int(benchmark["tcn"]["inference_batch_size"]),
    )

    tcn_threshold = float(
        calibration_scores.quantile(
            float(temporal_calibration["threshold_quantile"]),
            interpolation=str(
                temporal_calibration["quantile_interpolation"]
            ),
        )
    )

    concentration = _feature_error_concentration(
        calibration,
        detector=detector,
        device=result.device,
    )

    # --- ROUTER CALIBRATION: Robust-PCA + causal EWMA on MetroPT2 CAL only.
    pca = fit_pca_detector(
        train,
        features,
        variance_retained=float(method["router"]["variance_retained"]),
        scaler_name="robust",
    )

    pca_cal_raw = score_pca_detector(calibration, pca)
    pca_cal_ewma = causal_ewma(
        pca_cal_raw,
        alpha=float(method["router"]["ewma_alpha"]),
        reset_gap_minutes=int(method["router"]["reset_gap_minutes"]),
    )

    router_points: dict[str, dict[str, float | int]] = {}
    for operating_id, quantile_value in method["router"][
        "operating_quantiles"
    ].items():
        threshold = higher_quantile(
            pca_cal_ewma,
            float(quantile_value),
        )
        mask = routing_mask(pca_cal_ewma, threshold)
        router_points[str(operating_id)] = {
            "calibration_quantile": float(quantile_value),
            "threshold": threshold,
            "calibration_rows": len(mask),
            "routed_rows": int(mask.sum()),
            "routed_fraction": float(mask.mean()),
        }

    pca_alerting = study["frozen_pca"]["alerting"]
    pca_alert_threshold = float(
        pca_cal_ewma.quantile(
            float(pca_alerting["threshold_quantile"]),
            interpolation=str(pca_alerting["quantile_interpolation"]),
        )
    )

    # Save the transport checkpoint before TEST is opened.
    checkpoint = {
        "study": transport_root["name"],
        "dataset": "MetroPT2",
        "evidence_role": transport_root["evidence_role"],
        "state_dict": {
            key: value.detach().cpu()
            for key, value in result.model.state_dict().items()
        },
        "features": features,
        "feature_variant": feature_cfg["variant"],
        "sequence": {
            "history_bins": int(sequence["history_bins"]),
            "bin_minutes": int(sequence["bin_minutes"]),
            "history_minutes": int(sequence["history_minutes"]),
            "exact_contiguity_required": True,
        },
        "model_config": {
            "input_dim": model_config.input_dim,
            "hidden_dim": model_config.hidden_dim,
            "kernel_size": model_config.kernel_size,
            "dilations": list(model_config.dilations),
            "dropout": model_config.dropout,
        },
        "training_config": {
            "objective": training["objective"],
            "optimizer": training["optimizer"],
            "seed": training_config.seed,
            "max_epochs": training_config.max_epochs,
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
            "validation_fraction": training_config.validation_fraction,
            "patience": training_config.patience,
        },
        "scaler_name": "standard",
        "scaler_location": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "calibration_threshold": tcn_threshold,
        "router_points": router_points,
        "pca_alert_threshold": pca_alert_threshold,
    }

    checkpoint_path = ROOT / outputs["checkpoint"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = sha256_file(checkpoint_path)

    # --- TEST boundary. No frozen model/calibration quantity changes below.
    test = _load_partition(ROOT / data_cfg["test_file"])

    if len(test) != int(expected_rows["test"]):
        raise RuntimeError("MetroPT2 TEST row-count identity failed.")
    if len(test) != int(preprocessing_report["splits"]["test"]["valid_bins"]):
        raise RuntimeError("TEST no longer matches preprocessing report.")

    incidents = metropt2["dataset"]["reported_incidents"]
    detection = _load_yaml(ROOT / "configs" / "detection.yaml")
    detection_eval = detection["evaluation"]

    # Reproduce the already-committed MetroPT2 flowmeter_excluded Robust-PCA.
    pca_test_raw = score_pca_detector(test, pca)
    pca_test_ewma = causal_ewma(
        pca_test_raw,
        alpha=float(method["router"]["ewma_alpha"]),
        reset_gap_minutes=int(method["router"]["reset_gap_minutes"]),
    )

    _, pca_alerts = persistent_alerts(
        pca_test_ewma,
        threshold=pca_alert_threshold,
        required_hits=int(pca_alerting["persistence_hits"]),
        window_bins=int(pca_alerting["persistence_window"]),
        reset_gap_minutes=int(pca_alerting["reset_gap_minutes"]),
    )
    pca_episodes = extract_alert_episodes(
        pca_alerts,
        merge_minutes=int(pca_alerting["merge_minutes"]),
        reset_gap_minutes=int(pca_alerting["reset_gap_minutes"]),
    )
    pca_metrics = evaluate_detection(
        scores=pca_test_ewma,
        alerts=pca_alerts,
        episodes=pca_episodes,
        incidents=incidents,
        early_warning_hours=int(detection_eval["early_warning_hours"]),
        late_tolerance_hours=int(detection_eval["late_tolerance_hours"]),
        bin_minutes=int(detection_eval["bin_minutes"]),
    )

    prior_variant = prior_pca["variants"]["flowmeter_excluded"]
    reproduction_deltas = verify_frozen_metrics(
        pca_metrics,
        prior_variant["metrics"],
        tolerance=float(integrity["reproduction_tolerance"]),
    )

    threshold_delta = abs(
        pca_alert_threshold
        - float(prior_variant["model"]["threshold"])
    )
    if threshold_delta > float(integrity["reproduction_tolerance"]):
        raise RuntimeError(
            "Prior MetroPT2 PCA threshold reproduction failed: "
            f"delta={threshold_delta}."
        )

    # Always-on TCN evidence.
    tcn_scores = score_temporal_detector(
        test,
        detector,
        device=result.device,
        batch_size=int(benchmark["tcn"]["inference_batch_size"]),
    )
    tcn_alerts = (tcn_scores >= tcn_threshold).rename("alert")
    tcn_episodes = extract_alert_episodes(
        tcn_alerts,
        merge_minutes=int(method["evidence"]["episode_merge_minutes"]),
        reset_gap_minutes=int(
            method["evidence"]["episode_reset_gap_minutes"]
        ),
    )

    related = incident_related_mask(
        pd.DatetimeIndex(tcn_scores.index),
        incidents,
        early_warning_hours=int(
            method["evidence"]["incident_related_early_warning_hours"]
        ),
    )

    always_on_route = pd.Series(
        True,
        index=tcn_alerts.index,
        dtype=bool,
    )

    always_bins = bin_coverage(
        alerts=tcn_alerts,
        routed=always_on_route,
        incident_related=related,
    )
    always_episodes = episode_coverage(
        tcn_episodes,
        alerts=tcn_alerts,
        routed=always_on_route,
        incidents=incidents,
        early_warning_hours=int(
            method["evidence"]["incident_related_early_warning_hours"]
        ),
    )

    pca_router_scores = pca_test_ewma.reindex(tcn_scores.index)
    if pca_router_scores.isna().any():
        raise RuntimeError("PCA router alignment contains NaN.")

    routing: dict[str, Any] = {}
    for operating_id, point in router_points.items():
        routed = routing_mask(
            pca_router_scores,
            float(point["threshold"]),
        )
        routing[operating_id] = {
            "threshold": float(point["threshold"]),
            "calibration_quantile": float(point["calibration_quantile"]),
            "valid_tcn_windows": len(routed),
            "tcn_invocations": int(routed.sum()),
            "tcn_invocation_fraction": float(routed.mean()),
            **bin_coverage(
                alerts=tcn_alerts,
                routed=routed,
                incident_related=related,
            ),
            **episode_coverage(
                tcn_episodes,
                alerts=tcn_alerts,
                routed=routed,
                incidents=incidents,
                early_warning_hours=int(
                    method["evidence"][
                        "incident_related_early_warning_hours"
                    ]
                ),
            ),
        }

    report = {
        "schema_version": "aeroxai.energy_aware_metropt2_transport.v1",
        "study": transport_root["name"],
        "dataset": "MetroPT2",
        "evidence_class": "PREVIOUSLY_INSPECTED_CROSS_DATASET_TRANSPORT",
        "independent_confirmation": False,
        "causal_claim": False,
        "primary_result_commit": transport_root["primary_result_commit"],
        "test_opened": True,
        "test_tuning_performed": False,
        "feature_schema": {
            "variant": feature_cfg["variant"],
            "feature_count": len(features),
            "primary_feature_identity_passed": True,
            "names": features,
        },
        "training": {
            "device": result.device,
            "fit_samples": result.fit_samples,
            "validation_samples": result.validation_samples,
            "best_epoch": result.best_epoch,
            "epochs_ran": result.epochs_ran,
            "best_validation_mse": float(min(result.validation_losses)),
            "final_train_mse": float(result.train_losses[-1]),
            "train_losses": list(result.train_losses),
            "validation_losses": list(result.validation_losses),
        },
        "conditioning": {
            "train_scaled_magnitude": _scaled_magnitude_summary(train_scaled),
            "calibration_scaled_magnitude": _scaled_magnitude_summary(
                calibration_scaled
            ),
            "calibration_feature_mse_concentration": concentration,
        },
        "tcn": {
            "checkpoint": outputs["checkpoint"],
            "checkpoint_sha256": checkpoint_sha,
            "calibration_threshold": tcn_threshold,
            "calibration_score_rows": len(calibration_scores),
        },
        "router_calibration": router_points,
        "prior_pca_reproduction": {
            "passed": True,
            "threshold": pca_alert_threshold,
            "threshold_absolute_delta": threshold_delta,
            "metrics": pca_metrics,
            "absolute_deltas": reproduction_deltas,
        },
        "always_on_tcn": {
            "valid_tcn_windows": len(tcn_scores),
            "score_min": float(tcn_scores.min()),
            "score_median": float(tcn_scores.median()),
            "score_mean": float(tcn_scores.mean()),
            "score_max": float(tcn_scores.max()),
            **always_bins,
            **always_episodes,
            "episodes": serializable_episodes(tcn_episodes),
        },
        "routing": routing,
        "definitions": {
            "tcn_alert_rule": "raw_score >= MetroPT2 CAL q0.995 threshold",
            "score_smoothing": "none",
            "persistence": "none",
            "episode_merge_minutes": int(
                method["evidence"]["episode_merge_minutes"]
            ),
            "incident_related_window": (
                "incident_start_minus_24_hours through incident_end"
            ),
            "router_thresholds": (
                "q90/q95/q99 of MetroPT2 CAL Robust-PCA causal-EWMA"
            ),
        },
        "guardrails": transport["guardrails"],
    }

    output_path = ROOT / outputs["transport_result"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "dataset": "MetroPT2",
                "feature_variant": feature_cfg["variant"],
                "feature_count": len(features),
                "primary_feature_identity_passed": True,
                "prior_pca_reproduction_passed": True,
                "tcn_checkpoint_sha256": checkpoint_sha,
                "training": {
                    "best_epoch": result.best_epoch,
                    "epochs_ran": result.epochs_ran,
                    "best_validation_mse": float(
                        min(result.validation_losses)
                    ),
                    "calibration_threshold": tcn_threshold,
                    "calibration_top_3_mse_share": concentration[
                        "top_3_share"
                    ],
                },
                "router_calibration": router_points,
                "always_on_tcn": {
                    "valid_tcn_windows": len(tcn_scores),
                    "tcn_alert_bins": always_bins["tcn_alert_bins"],
                    "tcn_incident_related_alert_bins": always_bins[
                        "tcn_incident_related_alert_bins"
                    ],
                    "alert_episodes": always_episodes["alert_episodes"],
                    "incident_related_alert_episodes": always_episodes[
                        "incident_related_alert_episodes"
                    ],
                },
                "routing": routing,
                "test_tuning_performed": False,
                "output": str(output_path.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
