from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ml.data.rcsd1yd import sha256_file
from ml.detection.alerts import causal_ewma
from ml.detection.common import fit_scaler
from ml.detection.pca_detector import (
    fit_pca_detector,
    score_pca_detector,
)
from ml.energy.evidence_aware_routing import (
    ROUTER_FEATURE_NAMES,
    build_router_features,
)
from ml.energy.support_aware_routing import (
    feature_support_statistic,
    pca_latent_support_statistic,
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
from ml.temporal.sequences import build_causal_sequences
from ml.temporal.tcn import TemporalForecastConfig

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rcsd1yd_external_evaluation.yaml"
MANIFEST_PATH = ROOT / "docs/research/rcsd1yd_ingestion_manifest.json"
PARENT_PREFLIGHT_PATH = (
    ROOT / "docs/research/evidence_aware_router_preflight.json"
)
CHECKPOINT_PATH = ROOT / "artifacts/research/rcsd1yd_tcn.pt"
FREEZE_PATH = ROOT / "docs/research/rcsd1yd_adaptation_freeze.json"

PREREGISTRATION_COMMIT = "7dc7bd2d9a44758075dab286cc3d674949a876bd"
FROZEN_ROUTER_COMMIT = "aa3863080493c19483d536a2f9d2da1613a651b7"


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
    return frame


def _higher(series: pd.Series, quantile: float) -> float:
    value = float(
        series.quantile(
            quantile,
            interpolation="higher",
        )
    )
    if not np.isfinite(value):
        raise RuntimeError("Calibration quantile is non-finite.")
    return value


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _serialize_scaler(scaler: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "class": type(scaler).__name__,
        "scale": np.asarray(scaler.scale_, dtype=float).tolist(),
    }

    if hasattr(scaler, "center_"):
        result["center"] = np.asarray(
            scaler.center_,
            dtype=float,
        ).tolist()

    if hasattr(scaler, "mean_"):
        result["mean"] = np.asarray(
            scaler.mean_,
            dtype=float,
        ).tolist()

    return result


def _serialize_pca(detector: Any) -> dict[str, Any]:
    model = detector.model
    payload = {
        "features": list(detector.features),
        "scaler_name": detector.scaler_name,
        "scaler": _serialize_scaler(detector.scaler),
        "n_components": int(model.n_components_),
        "components": np.asarray(model.components_, dtype=float).tolist(),
        "mean": np.asarray(model.mean_, dtype=float).tolist(),
        "explained_variance": np.asarray(
            model.explained_variance_,
            dtype=float,
        ).tolist(),
        "explained_variance_ratio": np.asarray(
            model.explained_variance_ratio_,
            dtype=float,
        ).tolist(),
        "singular_values": np.asarray(
            model.singular_values_,
            dtype=float,
        ).tolist(),
    }
    payload["parameter_sha256"] = _json_sha256(payload)
    return payload


def _resolve_tcn_fit_scaler_frame(
    train: pd.DataFrame,
    *,
    features: list[str],
    sequence_length: int,
    bin_minutes: int,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.Timestamp, int, int]:
    raw_batch = build_causal_sequences(
        train.loc[:, features].to_numpy(dtype=float),
        pd.DatetimeIndex(train.index),
        sequence_length=sequence_length,
        expected_step=pd.Timedelta(minutes=bin_minutes),
    )

    sample_count = len(raw_batch.inputs)
    validation_count = max(
        1,
        round(sample_count * validation_fraction),
    )
    fit_count = sample_count - validation_count

    if fit_count < 2:
        raise RuntimeError("Insufficient TRAIN temporal samples.")

    validation_start = raw_batch.target_index[fit_count]

    scaler_frame = train.loc[
        train.index < validation_start,
        features,
    ]

    if scaler_frame.empty:
        raise RuntimeError("TCN TRAIN-fit scaler frame is empty.")

    return scaler_frame, validation_start, fit_count, validation_count


def main() -> None:
    config = _load_yaml(CONFIG_PATH)
    manifest = _load_json(MANIFEST_PATH)
    parent = _load_json(PARENT_PREFLIGHT_PATH)

    if manifest["preregistration_commit"] != PREREGISTRATION_COMMIT:
        raise RuntimeError("RCSD-1YD ingestion was not produced under freeze.")

    if parent["preregistration_commit"] != (
        "0b3a3135223e764ca0f3a855b1f18d3788db3d5e"
    ):
        raise RuntimeError("Frozen parent router preregistration identity changed.")

    selected_model = parent["selected_model"]
    selected_policy = parent["selected_policy"]

    if float(selected_policy["C"]) != float(config["frozen_router"]["C"]):
        raise RuntimeError("Frozen parent C mismatch.")
    if selected_policy["class_weight"] != config["frozen_router"]["class_weight"]:
        raise RuntimeError("Frozen parent class_weight mismatch.")
    if float(selected_policy["probability_threshold"]) != float(
        config["frozen_router"]["probability_threshold"]
    ):
        raise RuntimeError("Frozen parent probability threshold mismatch.")
    if selected_model["feature_names"] != config["cheap_router_features"]["features"]:
        raise RuntimeError("Frozen eight-feature router identity mismatch.")
    if selected_model["feature_names"] != ROUTER_FEATURE_NAMES:
        raise RuntimeError("Repository router-feature order changed.")

    train_path = ROOT / manifest["train"]["path"]
    calibration_path = ROOT / manifest["calibration"]["path"]

    if sha256_file(train_path) != manifest["train"]["sha256"]:
        raise RuntimeError("TRAIN parquet changed after ingestion.")
    if sha256_file(calibration_path) != manifest["calibration"]["sha256"]:
        raise RuntimeError("CALIBRATION parquet changed after ingestion.")

    # EVALUATION parquet is intentionally not opened by this script.
    train = _load_partition(train_path)
    calibration = _load_partition(calibration_path)

    features = [str(value) for value in config["sensor_columns"]]

    if train.columns.tolist() != features:
        raise RuntimeError("TRAIN schema differs from frozen 25-sensor schema.")
    if calibration.columns.tolist() != features:
        raise RuntimeError(
            "CALIBRATION schema differs from frozen 25-sensor schema."
        )

    temporal = config["dataset_specific_tcn"]
    sequence_length = int(temporal["sequence_length_bins"])
    bin_minutes = int(temporal["native_bin_minutes"])
    validation_fraction = float(
        temporal["train_validation_split"]["last_fraction_for_validation"]
    )

    scaler_fit_frame, validation_start, expected_fit, expected_validation = (
        _resolve_tcn_fit_scaler_frame(
            train,
            features=features,
            sequence_length=sequence_length,
            bin_minutes=bin_minutes,
            validation_fraction=validation_fraction,
        )
    )

    tcn_scaler = fit_scaler(
        scaler_fit_frame,
        features,
        method="standard",
    )

    train_batch = prepare_temporal_batch(
        train,
        features=features,
        scaler=tcn_scaler,
        sequence_length=sequence_length,
        bin_minutes=bin_minutes,
    )

    hidden_dim = int(temporal["hidden_dim"])
    model_config = TemporalForecastConfig(
        input_dim=len(features),
        hidden_dim=hidden_dim,
        kernel_size=int(temporal["kernel_size"]),
        dilations=tuple(int(value) for value in temporal["dilations"]),
        dropout=float(temporal["dropout"]),
    )

    training_config = EnergyAwareTrainingConfig(
        seed=int(temporal["seed"]),
        max_epochs=int(temporal["max_epochs"]),
        batch_size=int(temporal["batch_size"]),
        learning_rate=float(temporal["learning_rate"]),
        weight_decay=float(temporal["weight_decay"]),
        validation_fraction=validation_fraction,
        patience=int(temporal["early_stopping_patience"]),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    training_result = train_energy_aware_tcn(
        train_batch,
        model_config=model_config,
        training_config=training_config,
    )

    if training_result.fit_samples != expected_fit:
        raise RuntimeError("TCN chronological fit sample count changed.")
    if training_result.validation_samples != expected_validation:
        raise RuntimeError("TCN chronological validation sample count changed.")

    detector = TemporalDetector(
        features=features,
        scaler=tcn_scaler,
        model=training_result.model,
        sequence_length=sequence_length,
        bin_minutes=bin_minutes,
    )

    calibration_scores = score_temporal_detector(
        calibration,
        detector,
        device=training_result.device,
        batch_size=1024,
    )

    high_threshold = _higher(calibration_scores, 0.90)
    alert_threshold = _higher(calibration_scores, 0.995)

    pca_cfg = config["dataset_specific_pca"]
    pca = fit_pca_detector(
        train,
        features,
        variance_retained=float(pca_cfg["variance_retained"]),
        scaler_name="robust",
    )

    support_scaler = fit_scaler(
        train,
        features,
        method="standard",
    )

    pca_raw = score_pca_detector(
        calibration,
        pca,
    )

    pca_ewma = causal_ewma(
        pca_raw,
        alpha=float(pca_cfg["causal_ewma"]["alpha"]),
        reset_gap_minutes=int(
            pca_cfg["causal_ewma"]["reset_gap_minutes"]
        ),
    )

    feature_support_raw = feature_support_statistic(
        calibration,
        features=features,
        scaler=support_scaler,
        quantile=0.95,
    )

    latent_support_raw = pca_latent_support_statistic(
        calibration,
        detector=pca,
    )

    target_index = pd.DatetimeIndex(calibration_scores.index)

    router_cal_features, references = build_router_features(
        pca_ewma=pca_ewma,
        feature_support_raw=feature_support_raw,
        latent_support_raw=latent_support_raw,
        target_index=target_index,
        fit_index=target_index,
        rolling_history_bins=int(
            config["cheap_router_features"]["rolling_history_bins"]
        ),
    )

    if router_cal_features.columns.tolist() != ROUTER_FEATURE_NAMES:
        raise RuntimeError("RCSD router feature order changed.")

    pca_target = pca_ewma.reindex(target_index)

    pca_thresholds = {
        "q90": _higher(pca_target, 0.90),
        "q95": _higher(pca_target, 0.95),
        "q99": _higher(pca_target, 0.99),
    }

    checkpoint = {
        "study": config["study"]["name"],
        "role": "rcsd1yd_dataset_specific_tcn_temporal_evidence",
        "state_dict": {
            key: value.detach().cpu()
            for key, value in training_result.model.state_dict().items()
        },
        "features": features,
        "sequence": {
            "history_bins": sequence_length,
            "bin_minutes": bin_minutes,
            "history_minutes": int(temporal["physical_history_minutes"]),
            "exact_native_contiguity_required": True,
        },
        "model_config": {
            "input_dim": model_config.input_dim,
            "hidden_dim": model_config.hidden_dim,
            "kernel_size": model_config.kernel_size,
            "dilations": list(model_config.dilations),
            "dropout": model_config.dropout,
        },
        "training_config": {
            "seed": training_config.seed,
            "max_epochs": training_config.max_epochs,
            "batch_size": training_config.batch_size,
            "learning_rate": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
            "validation_fraction": training_config.validation_fraction,
            "patience": training_config.patience,
        },
        "scaler_name": "standard",
        "scaler_fit_end_exclusive": str(validation_start),
        "scaler_location": np.asarray(tcn_scaler.mean_, dtype=float),
        "scaler_scale": np.asarray(tcn_scaler.scale_, dtype=float),
        "calibration_high_evidence_threshold_q90": high_threshold,
        "calibration_alert_evidence_threshold_q995": alert_threshold,
    }

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, CHECKPOINT_PATH)
    checkpoint_sha256 = sha256_file(CHECKPOINT_PATH)

    pca_payload = _serialize_pca(pca)

    frozen_router_identity = {
        "source_commit": FROZEN_ROUTER_COMMIT,
        "preflight_file": str(PARENT_PREFLIGHT_PATH.relative_to(ROOT)),
        "preflight_sha256": sha256_file(PARENT_PREFLIGHT_PATH),
        "feature_names": list(selected_model["feature_names"]),
        "feature_scaler": selected_model["feature_scaler"],
        "logistic": selected_model["logistic"],
        "C": selected_model["C"],
        "class_weight": selected_model["class_weight"],
        "probability_threshold": selected_model["probability_threshold"],
    }
    frozen_router_identity["identity_sha256"] = _json_sha256(
        frozen_router_identity
    )

    freeze = {
        "schema_version": "aeroxai.rcsd1yd_adaptation_freeze.v1",
        "study": config["study"]["name"],
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "parent_simulation_commit": config["study"]["parent_simulation_commit"],
        "data_use": {
            "train_opened": True,
            "calibration_opened": True,
            "evaluation_parquet_opened_by_adaptation": False,
            "evaluation_sensor_summary_computed": False,
            "evaluation_tcn_metrics_computed": False,
            "evaluation_router_metrics_computed": False,
            "evaluation_energy_metrics_computed": False,
        },
        "source": {
            "md5": manifest["source"]["md5"],
            "schema_validation_passed": manifest["source"][
                "schema_validation_passed"
            ],
            "sensor_count": manifest["source"]["sensor_count"],
            "sensor_columns": manifest["source"]["sensor_columns"],
        },
        "splits": {
            "train": {
                "rows": manifest["train"]["rows"],
                "bounds": manifest["train"]["bounds"],
                "dropped_nonfinite_rows": manifest["train"][
                    "dropped_nonfinite_rows"
                ],
                "gaps_greater_than_30_minutes": manifest["train"][
                    "gaps_greater_than_30_minutes"
                ],
                "sha256": manifest["train"]["sha256"],
            },
            "calibration": {
                "rows": manifest["calibration"]["rows"],
                "bounds": manifest["calibration"]["bounds"],
                "dropped_nonfinite_rows": manifest["calibration"][
                    "dropped_nonfinite_rows"
                ],
                "gaps_greater_than_30_minutes": manifest["calibration"][
                    "gaps_greater_than_30_minutes"
                ],
                "sha256": manifest["calibration"]["sha256"],
            },
            "evaluation": {
                "rows": manifest["evaluation"]["rows"],
                "frozen_calendar_bounds": manifest["evaluation"][
                    "frozen_calendar_bounds"
                ],
                "sha256": manifest["evaluation"]["sha256"],
            },
        },
        "tcn": {
            "checkpoint_path": str(CHECKPOINT_PATH.relative_to(ROOT)),
            "checkpoint_sha256": checkpoint_sha256,
            "features": features,
            "sequence_length_bins": sequence_length,
            "bin_minutes": bin_minutes,
            "physical_history_minutes": int(temporal["physical_history_minutes"]),
            "train_scaler_fit_end_exclusive": str(validation_start),
            "fit_samples": training_result.fit_samples,
            "validation_samples": training_result.validation_samples,
            "best_epoch": training_result.best_epoch,
            "epochs_ran": training_result.epochs_ran,
            "best_validation_mse": float(
                min(training_result.validation_losses)
            ),
            "final_train_mse": float(training_result.train_losses[-1]),
            "device": training_result.device,
            "calibration_score_rows": len(calibration_scores),
            "calibration_high_evidence_threshold_q90": high_threshold,
            "calibration_alert_evidence_threshold_q995": alert_threshold,
        },
        "pca": pca_payload,
        "support_scaler": _serialize_scaler(support_scaler),
        "calibration_references": {
            **references,
            "pca_q99_threshold": pca_thresholds["q99"],
            "pca_q90_threshold": pca_thresholds["q90"],
            "pca_q95_threshold": pca_thresholds["q95"],
            "target_rows": len(target_index),
            "target_start": str(target_index[0]),
            "target_end": str(target_index[-1]),
        },
        "frozen_router": frozen_router_identity,
        "ready_for_evaluation_preflight": True,
        "evaluation_metrics_computed": False,
    }

    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(
        json.dumps(freeze, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "evaluation_parquet_opened_by_adaptation": False,
                "evaluation_metrics_computed": False,
                "tcn_checkpoint_sha256": checkpoint_sha256,
                "tcn_fit_samples": training_result.fit_samples,
                "tcn_validation_samples": training_result.validation_samples,
                "tcn_best_epoch": training_result.best_epoch,
                "tcn_best_validation_mse": float(
                    min(training_result.validation_losses)
                ),
                "calibration_tcn_score_rows": len(calibration_scores),
                "calibration_q90": high_threshold,
                "calibration_q995": alert_threshold,
                "pca_components": pca_payload["n_components"],
                "calibration_router_reference_rows": len(target_index),
                "ready_for_evaluation_preflight": True,
                "output": str(FREEZE_PATH.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
