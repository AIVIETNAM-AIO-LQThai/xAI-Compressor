from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ml.data.features import model_feature_columns
from ml.detection.common import fit_scaler, transform_frame
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
STUDY_CONFIG_PATH = (
    ROOT / "configs" / "energy_aware_hierarchical_intelligence.yaml"
)
CHECKPOINT_PATH = ROOT / "artifacts" / "research" / "energy_aware_tcn.pt"
REPORT_PATH = (
    ROOT / "docs" / "research" / "energy_aware_tcn_calibration.json"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}.")

    return payload


def _load_partition(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index, name="timestamp")
    return frame.sort_index()


def validate_study_config(config: dict[str, Any]) -> None:
    temporal = config["temporal_model"]
    sequence = temporal["sequence"]
    architecture = temporal["architecture"]
    training = temporal["training"]
    calibration = temporal["calibration"]

    expected = {
        "family": "causal_tcn_next_step_prediction",
        "scaler": "standard",
        "objective": "mean_squared_next_step_prediction_error",
        "optimizer": "AdamW",
    }

    if temporal["family"] != expected["family"]:
        raise ValueError("Unexpected temporal_model.family.")
    if temporal["scaler"] != expected["scaler"]:
        raise ValueError("Study is frozen to StandardScaler for the TCN.")
    if training["objective"] != expected["objective"]:
        raise ValueError("Study is frozen to next-step MSE.")
    if training["optimizer"] != expected["optimizer"]:
        raise ValueError("Study is frozen to AdamW.")

    if int(sequence["bin_minutes"]) != 5:
        raise ValueError("Study is frozen to 5-minute bins.")
    if int(sequence["history_bins"]) != 12:
        raise ValueError("Study is frozen to 12 history bins.")
    if int(sequence["history_minutes"]) != 60:
        raise ValueError("Study is frozen to a 60-minute history.")
    if not bool(sequence["prohibit_cross_gap_sequences"]):
        raise ValueError("Cross-gap sequences must remain prohibited.")
    if int(sequence["max_gap_minutes"]) != 10:
        raise ValueError("Frozen max_gap_minutes must remain 10.")

    if int(architecture["residual_blocks"]) != 3:
        raise ValueError("Study is frozen to three residual blocks.")
    if [int(v) for v in architecture["channels"]] != [32, 32, 32]:
        raise ValueError("Study is frozen to channels [32, 32, 32].")
    if int(architecture["kernel_size"]) != 3:
        raise ValueError("Study is frozen to kernel size 3.")
    if [int(v) for v in architecture["dilations"]] != [1, 2, 4]:
        raise ValueError("Study is frozen to dilations [1, 2, 4].")
    if float(architecture["dropout"]) != 0.10:
        raise ValueError("Study is frozen to dropout 0.10.")

    frozen_training = {
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 256,
        "max_epochs": 50,
        "early_stopping_patience": 5,
        "seed": 20260915,
        "fit_fraction_of_train": 0.80,
        "validation_fraction_of_train": 0.20,
    }

    for key, value in frozen_training.items():
        observed = training[key]
        if isinstance(value, float):
            if float(observed) != value:
                raise ValueError(f"Unexpected frozen training value: {key}.")
        elif int(observed) != value:
            raise ValueError(f"Unexpected frozen training value: {key}.")

    if training["validation_split"] != "chronological_tail":
        raise ValueError("Validation must remain the chronological tail.")
    if bool(training["calibration_used_for_training"]):
        raise ValueError("CALIBRATION must not be used for training.")
    if bool(training["test_used_for_training"]):
        raise ValueError("TEST must not be used for training.")
    if bool(training["hyperparameter_search_allowed"]):
        raise ValueError("Hyperparameter search must remain disabled.")

    if calibration["threshold_source"] != "official_calibration_split_only":
        raise ValueError("Threshold source must remain CALIBRATION only.")
    if float(calibration["threshold_quantile"]) != 0.995:
        raise ValueError("TCN threshold quantile must remain 0.995.")
    if calibration["quantile_interpolation"] != "higher":
        raise ValueError("TCN quantile interpolation must remain 'higher'.")



def _scaled_magnitude_summary(values: np.ndarray) -> dict[str, float]:
    absolute = np.abs(values)
    return {
        "p50_abs": float(np.quantile(absolute, 0.50)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "p99_abs": float(np.quantile(absolute, 0.99)),
        "p999_abs": float(np.quantile(absolute, 0.999)),
        "max_abs": float(absolute.max()),
    }


def _calibration_error_concentration(
    frame: pd.DataFrame,
    *,
    detector: TemporalDetector,
    device: str,
    batch_size: int = 1024,
) -> dict[str, object]:
    batch = prepare_temporal_batch(
        frame,
        features=detector.features,
        scaler=detector.scaler,
        sequence_length=detector.sequence_length,
        bin_minutes=detector.bin_minutes,
    )

    torch_device = torch.device(device)
    model = detector.model.to(torch_device)
    model.eval()

    squared_errors: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(batch.inputs), batch_size):
            stop = min(len(batch.inputs), start + batch_size)

            inputs = torch.from_numpy(
                batch.inputs[start:stop]
            ).to(torch_device)
            targets = torch.from_numpy(
                batch.targets[start:stop]
            ).to(torch_device)

            predictions = model(inputs)
            squared_errors.append(
                (targets - predictions)
                .square()
                .detach()
                .cpu()
                .numpy()
            )

    feature_mse = np.concatenate(squared_errors, axis=0).mean(axis=0)
    total = float(feature_mse.sum())
    order = np.argsort(feature_mse)[::-1]

    top_features = []
    for index in order[:10]:
        top_features.append(
            {
                "feature": detector.features[int(index)],
                "mse": float(feature_mse[int(index)]),
                "share": float(feature_mse[int(index)] / total),
            }
        )

    return {
        "top_1_share": float(feature_mse[order[:1]].sum() / total),
        "top_3_share": float(feature_mse[order[:3]].sum() / total),
        "top_5_share": float(feature_mse[order[:5]].sum() / total),
        "top_features": top_features,
    }


def main() -> None:
    config = _load_yaml(STUDY_CONFIG_PATH)
    validate_study_config(config)

    primary = config["datasets"]["primary"]
    temporal = config["temporal_model"]
    sequence = temporal["sequence"]
    architecture = temporal["architecture"]
    training = temporal["training"]
    calibration_config = temporal["calibration"]

    # G3 intentionally opens TRAIN and CALIBRATION only.
    train = _load_partition(ROOT / primary["train_file"])
    calibration = _load_partition(ROOT / primary["calibration_file"])

    features = model_feature_columns(train.columns)

    if len(features) != 63:
        raise RuntimeError(
            "Expected the frozen 63-feature representation, "
            f"got {len(features)}."
        )

    missing_in_calibration = [
        feature for feature in features if feature not in calibration.columns
    ]
    if missing_in_calibration:
        raise RuntimeError(
            "CALIBRATION is missing frozen features: "
            f"{missing_in_calibration}"
        )

    scaler_name = str(temporal["scaler"])

    scaler = fit_scaler(
        train,
        features,
        method=scaler_name,
    )

    train_scaled = transform_frame(train, features, scaler)
    calibration_scaled = transform_frame(calibration, features, scaler)

    # Exact 5-minute contiguity preserves the preregistered 60-minute
    # history and is stricter than the 10-minute maximum-gap ceiling.
    train_batch = prepare_temporal_batch(
        train,
        features=features,
        scaler=scaler,
        sequence_length=int(sequence["history_bins"]),
        bin_minutes=int(sequence["bin_minutes"]),
    )

    channels = [int(value) for value in architecture["channels"]]
    if len(set(channels)) != 1:
        raise ValueError(
            "Existing TCNForecaster requires equal hidden channels "
            "across residual blocks."
        )

    model_config = TemporalForecastConfig(
        input_dim=len(features),
        hidden_dim=channels[0],
        kernel_size=int(architecture["kernel_size"]),
        dilations=tuple(
            int(value) for value in architecture["dilations"]
        ),
        dropout=float(architecture["dropout"]),
    )

    training_config = EnergyAwareTrainingConfig(
        seed=int(training["seed"]),
        max_epochs=int(training["max_epochs"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        validation_fraction=float(
            training["validation_fraction_of_train"]
        ),
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
        batch_size=1024,
    )

    threshold = float(
        calibration_scores.quantile(
            float(calibration_config["threshold_quantile"]),
            interpolation=str(
                calibration_config["quantile_interpolation"]
            ),
        )
    )

    error_concentration = _calibration_error_concentration(
        calibration,
        detector=detector,
        device=result.device,
    )

    checkpoint = {
        "study": config["study"]["name"],
        "evidence_role": "research_second_stage_evidence",
        "operational_alert_source": False,
        "state_dict": {
            key: value.detach().cpu()
            for key, value in result.model.state_dict().items()
        },
        "features": features,
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
        "scaler_name": scaler_name,
        "scaler_location": (
            scaler.center_
            if hasattr(scaler, "center_")
            else scaler.mean_
        ),
        "scaler_scale": scaler.scale_,
        "calibration_threshold": threshold,
    }

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, CHECKPOINT_PATH)

    report = {
        "schema_version": "aeroxai.energy_aware_tcn_calibration.v1",
        "study": config["study"]["name"],
        "evidence_class": config["study"]["evidence_class"],
        "causal_claim": False,
        "purpose": (
            "G3 TRAIN/CALIBRATION-only TCN training and calibration; "
            "MetroPT-3 TEST not opened"
        ),
        "data_use": {
            "train_file": primary["train_file"],
            "calibration_file": primary["calibration_file"],
            "test_file_opened": False,
            "calibration_used_for_training": False,
        },
        "features": {
            "count": len(features),
            "names": features,
            "scaler": scaler_name,
            "scaler_fit": "TRAIN only",
        },
        "sequence": {
            "history_bins": int(sequence["history_bins"]),
            "bin_minutes": int(sequence["bin_minutes"]),
            "history_minutes": int(sequence["history_minutes"]),
            "exact_5_minute_contiguity": True,
            "max_gap_minutes_ceiling": int(sequence["max_gap_minutes"]),
        },
        "model": {
            "family": temporal["family"],
            "residual_blocks": int(architecture["residual_blocks"]),
            "channels": channels,
            "kernel_size": model_config.kernel_size,
            "dilations": list(model_config.dilations),
            "dropout": model_config.dropout,
            "output_features": len(features),
        },
        "training": {
            "objective": training["objective"],
            "optimizer": training["optimizer"],
            "device": result.device,
            "fit_samples": result.fit_samples,
            "validation_samples": result.validation_samples,
            "best_epoch": result.best_epoch,
            "epochs_ran": result.epochs_ran,
            "best_validation_mse": min(result.validation_losses),
            "final_train_mse": result.train_losses[-1],
            "train_losses": list(result.train_losses),
            "validation_losses": list(result.validation_losses),
        },
        "conditioning": {
            "train_scaled_magnitude": _scaled_magnitude_summary(
                train_scaled
            ),
            "calibration_scaled_magnitude": _scaled_magnitude_summary(
                calibration_scaled
            ),
            "calibration_feature_mse_concentration": error_concentration,
        },
        "calibration": {
            "score_rows": len(calibration_scores),
            "threshold_quantile": float(
                calibration_config["threshold_quantile"]
            ),
            "quantile_interpolation": calibration_config[
                "quantile_interpolation"
            ],
            "threshold": threshold,
            "score_min": float(calibration_scores.min()),
            "score_median": float(calibration_scores.median()),
            "score_mean": float(calibration_scores.mean()),
            "score_max": float(calibration_scores.max()),
        },
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "device": result.device,
                "fit_samples": result.fit_samples,
                "validation_samples": result.validation_samples,
                "best_epoch": result.best_epoch,
                "epochs_ran": result.epochs_ran,
                "best_validation_mse": min(result.validation_losses),
                "calibration_score_rows": len(calibration_scores),
                "calibration_threshold": threshold,
                "scaler": scaler_name,
                "train_scaled_p99_abs": (
                    _scaled_magnitude_summary(train_scaled)["p99_abs"]
                ),
                "calibration_top_3_mse_share": (
                    error_concentration["top_3_share"]
                ),
                "test_file_opened": False,
                "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
                "report": str(REPORT_PATH.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
