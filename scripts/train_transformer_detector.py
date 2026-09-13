from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.data.features import model_feature_columns
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
from ml.temporal.sequences import build_causal_sequences
from ml.temporal.transformer import (
    TemporalTransformerConfig,
    TemporalTransformerForecaster,
)

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return payload


def load_partition(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index, name="timestamp")
    return frame.sort_index()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_scores(
    calibration_scores: pd.Series,
    test_scores: pd.Series,
    *,
    alert_config: dict[str, Any],
    evaluation_config: dict[str, Any],
    incidents: list[dict[str, Any]],
) -> tuple[dict[str, Any], float, pd.Series, pd.Series]:
    calibration_smoothed = causal_ewma(
        calibration_scores,
        alpha=float(alert_config["ewma_alpha"]),
        reset_gap_minutes=int(alert_config["reset_gap_minutes"]),
    )
    threshold = float(
        calibration_smoothed.quantile(
            float(alert_config["threshold_quantile"]),
            interpolation="higher",
        )
    )
    test_smoothed = causal_ewma(
        test_scores,
        alpha=float(alert_config["ewma_alpha"]),
        reset_gap_minutes=int(alert_config["reset_gap_minutes"]),
    )
    threshold_hits, alerts = persistent_alerts(
        test_smoothed,
        threshold=threshold,
        required_hits=int(alert_config["persistence_hits"]),
        window_bins=int(alert_config["persistence_window"]),
        reset_gap_minutes=int(alert_config["reset_gap_minutes"]),
    )
    episodes = extract_alert_episodes(
        alerts,
        merge_minutes=int(alert_config["merge_minutes"]),
        reset_gap_minutes=int(alert_config["reset_gap_minutes"]),
    )
    metrics = evaluate_detection(
        scores=test_smoothed,
        alerts=alerts,
        episodes=episodes,
        incidents=incidents,
        early_warning_hours=int(evaluation_config["early_warning_hours"]),
        late_tolerance_hours=int(evaluation_config["late_tolerance_hours"]),
        bin_minutes=int(evaluation_config["bin_minutes"]),
    )
    return metrics, threshold, threshold_hits, test_smoothed


def score_model(
    frame: pd.DataFrame,
    *,
    features: list[str],
    scaler,
    model: nn.Module,
    sequence_length: int,
    bin_minutes: int,
    device: torch.device,
    batch_size: int = 1024,
) -> pd.Series:
    values = transform_frame(frame, features, scaler)
    batch = build_causal_sequences(
        values,
        pd.DatetimeIndex(frame.index),
        sequence_length=sequence_length,
        expected_step=pd.Timedelta(minutes=bin_minutes),
    )

    model = model.to(device)
    model.eval()
    scores: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(batch.inputs), batch_size):
            stop = min(len(batch.inputs), start + batch_size)
            inputs = torch.from_numpy(batch.inputs[start:stop]).to(device)
            targets = torch.from_numpy(batch.targets[start:stop]).to(device)
            predictions = model(inputs)
            batch_scores = torch.mean(
                (targets - predictions).square(),
                dim=1,
            )
            scores.append(batch_scores.cpu().numpy())

    return pd.Series(
        np.concatenate(scores),
        index=batch.target_index,
        name="raw_score",
        dtype=float,
    )


def main() -> None:
    metropt = load_yaml(ROOT / "configs" / "metropt.yaml")
    detection = load_yaml(ROOT / "configs" / "detection.yaml")
    cfg = load_yaml(ROOT / "configs" / "transformer.yaml")

    preprocessing = metropt["preprocessing"]
    incidents = metropt["dataset"]["reported_incidents"]
    alert_config = detection["alerting"]
    evaluation_config = detection["evaluation"]

    train = load_partition(ROOT / preprocessing["train_file"])
    calibration = load_partition(ROOT / preprocessing["calibration_file"])
    test = load_partition(ROOT / preprocessing["test_file"])

    features = model_feature_columns(train.columns)
    if len(features) != 63:
        raise RuntimeError(f"Expected 63 model features, got {len(features)}.")

    scaler = fit_scaler(train, features, method="robust")
    sequence_length = int(cfg["temporal"]["sequence_length"])
    bin_minutes = int(evaluation_config["bin_minutes"])

    train_values = transform_frame(train, features, scaler)
    train_batch = build_causal_sequences(
        train_values,
        pd.DatetimeIndex(train.index),
        sequence_length=sequence_length,
        expected_step=pd.Timedelta(minutes=bin_minutes),
    )

    model_cfg = TemporalTransformerConfig(
        input_dim=len(features),
        d_model=int(cfg["model"]["d_model"]),
        nhead=int(cfg["model"]["nhead"]),
        num_layers=int(cfg["model"]["num_layers"]),
        dim_feedforward=int(cfg["model"]["dim_feedforward"]),
        dropout=float(cfg["model"]["dropout"]),
        max_sequence_length=int(cfg["model"]["max_sequence_length"]),
    )

    training_cfg = cfg["training"]
    seed = int(training_cfg["seed"])
    set_seed(seed)
    device = resolve_device(str(training_cfg["device"]))

    model = TemporalTransformerForecaster(model_cfg).to(device)
    sample_count = len(train_batch.inputs)
    validation_count = max(
        1,
        round(sample_count * float(training_cfg["validation_fraction"])),
    )
    fit_count = sample_count - validation_count

    fit_dataset = TensorDataset(
        torch.from_numpy(train_batch.inputs[:fit_count]),
        torch.from_numpy(train_batch.targets[:fit_count]),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(train_batch.inputs[fit_count:]),
        torch.from_numpy(train_batch.targets[fit_count:]),
    )

    generator = torch.Generator().manual_seed(seed)
    fit_loader = DataLoader(
        fit_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
    )

    criterion = nn.HuberLoss(delta=float(training_cfg["huber_delta"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    best_validation = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    train_losses: list[float] = []
    validation_losses: list[float] = []

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        model.train()
        total_loss = 0.0
        seen = 0

        for inputs, targets in fit_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            count = int(inputs.shape[0])
            total_loss += float(loss.item()) * count
            seen += count

        train_loss = total_loss / seen
        train_losses.append(train_loss)

        model.eval()
        val_total = 0.0
        val_seen = 0

        with torch.inference_mode():
            for inputs, targets in validation_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                loss = criterion(model(inputs), targets)
                count = int(inputs.shape[0])
                val_total += float(loss.item()) * count
                val_seen += count

        validation_loss = val_total / val_seen
        validation_losses.append(validation_loss)

        if validation_loss < (
            best_validation - float(training_cfg["min_delta"])
        ):
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                }
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(training_cfg["patience"]):
            break

    if best_state is None:
        raise RuntimeError("Transformer training produced no valid state.")

    model.load_state_dict(best_state)
    model = model.cpu()

    calibration_scores = score_model(
        calibration,
        features=features,
        scaler=scaler,
        model=model,
        sequence_length=sequence_length,
        bin_minutes=bin_minutes,
        device=device,
    )
    test_scores = score_model(
        test,
        features=features,
        scaler=scaler,
        model=model,
        sequence_length=sequence_length,
        bin_minutes=bin_minutes,
        device=device,
    )

    transformer_metrics, threshold, threshold_hits, test_smoothed = (
        evaluate_scores(
            calibration_scores,
            test_scores,
            alert_config=alert_config,
            evaluation_config=evaluation_config,
            incidents=incidents,
        )
    )

    pca = fit_pca_detector(
        train,
        features,
        variance_retained=float(detection["pca"]["variance_retained"]),
        scaler_name="robust",
    )
    pca_calibration = score_pca_detector(calibration, pca).reindex(
        calibration_scores.index
    )
    pca_test = score_pca_detector(test, pca).reindex(test_scores.index)

    pca_metrics, pca_threshold, _, _ = evaluate_scores(
        pca_calibration,
        pca_test,
        alert_config=alert_config,
        evaluation_config=evaluation_config,
        incidents=incidents,
    )

    tcn_path = ROOT / "docs" / "tcn_detector_benchmark.json"
    tcn_reference = None
    if tcn_path.exists():
        tcn_reference = json.loads(
            tcn_path.read_text(encoding="utf-8")
        ).get("tcn_metrics")

    report = {
        "evidence_class": "REAL",
        "causal_claim": False,
        "evaluation_protocol": {
            "training_partition_only": True,
            "calibration_threshold_only": True,
            "test_final_evaluation_only": True,
            "comparison_basis": "same_causal_target_timestamps",
            "sequence_length_bins": sequence_length,
            "sequence_minutes": sequence_length * bin_minutes,
            "feature_count": len(features),
            "train_sequence_targets": len(train_batch.targets),
            "calibration_score_rows": len(calibration_scores),
            "test_score_rows": len(test_scores),
        },
        "training": {
            "device": str(device),
            "fit_samples": fit_count,
            "validation_samples": validation_count,
            "best_epoch": best_epoch,
            "epochs_ran": len(train_losses),
            "best_validation_loss": min(validation_losses),
            "final_train_loss": train_losses[-1],
        },
        "model": {
            "name": "causal_transformer_next_state_forecaster",
            "d_model": model_cfg.d_model,
            "nhead": model_cfg.nhead,
            "num_layers": model_cfg.num_layers,
            "dim_feedforward": model_cfg.dim_feedforward,
            "dropout": model_cfg.dropout,
            "threshold": threshold,
        },
        "transformer_metrics": transformer_metrics,
        "aligned_pca_robust": pca_metrics,
        "aligned_pca_threshold": pca_threshold,
        "tcn_reference": tcn_reference,
        "comparison_vs_pca": {
            "false_alerts_per_24h_delta": (
                transformer_metrics["false_alerts_per_24h"]
                - pca_metrics["false_alerts_per_24h"]
            ),
            "pr_auc_delta": (
                transformer_metrics["pr_auc"] - pca_metrics["pr_auc"]
            ),
            "timely_incident_recall_delta": (
                transformer_metrics["timely_incident_recall"]
                - pca_metrics["timely_incident_recall"]
            ),
            "pre_onset_incident_recall_delta": (
                transformer_metrics["pre_onset_incident_recall"]
                - pca_metrics["pre_onset_incident_recall"]
            ),
        },
    }

    if tcn_reference is not None:
        report["comparison_vs_tcn"] = {
            "false_alerts_per_24h_delta": (
                transformer_metrics["false_alerts_per_24h"]
                - tcn_reference["false_alerts_per_24h"]
            ),
            "pr_auc_delta": (
                transformer_metrics["pr_auc"] - tcn_reference["pr_auc"]
            ),
            "pre_onset_incident_recall_delta": (
                transformer_metrics["pre_onset_incident_recall"]
                - tcn_reference["pre_onset_incident_recall"]
            ),
        }

    outputs = cfg["outputs"]
    checkpoint_path = ROOT / outputs["checkpoint_file"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "features": features,
            "sequence_length": sequence_length,
            "bin_minutes": bin_minutes,
            "model_config": {
                "input_dim": model_cfg.input_dim,
                "d_model": model_cfg.d_model,
                "nhead": model_cfg.nhead,
                "num_layers": model_cfg.num_layers,
                "dim_feedforward": model_cfg.dim_feedforward,
                "dropout": model_cfg.dropout,
                "max_sequence_length": model_cfg.max_sequence_length,
            },
            "scaler_center": scaler.center_,
            "scaler_scale": scaler.scale_,
            "threshold": threshold,
        },
        checkpoint_path,
    )

    scores_path = ROOT / outputs["scores_file"]
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "raw_score": test_scores,
            "smoothed_score": test_smoothed,
            "threshold_hit": threshold_hits,
        }
    ).to_parquet(scores_path)

    report_path = ROOT / outputs["benchmark_file"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "device": str(device),
                "best_epoch": best_epoch,
                "epochs_ran": len(train_losses),
                "transformer": {
                    key: transformer_metrics[key]
                    for key in (
                        "timely_incident_recall",
                        "pre_onset_incident_recall",
                        "false_alerts_per_24h",
                        "time_in_alert_fraction",
                        "pr_auc",
                    )
                },
                "aligned_pca_robust": {
                    key: pca_metrics[key]
                    for key in (
                        "timely_incident_recall",
                        "pre_onset_incident_recall",
                        "false_alerts_per_24h",
                        "time_in_alert_fraction",
                        "pr_auc",
                    )
                },
                "tcn_reference": None
                if tcn_reference is None
                else {
                    key: tcn_reference[key]
                    for key in (
                        "timely_incident_recall",
                        "pre_onset_incident_recall",
                        "false_alerts_per_24h",
                        "time_in_alert_fraction",
                        "pr_auc",
                    )
                },
                "comparison_vs_pca": report["comparison_vs_pca"],
                "comparison_vs_tcn": report.get("comparison_vs_tcn"),
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
