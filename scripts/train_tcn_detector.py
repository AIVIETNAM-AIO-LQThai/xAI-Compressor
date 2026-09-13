from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from ml.data.features import (
    model_feature_columns,
)
from ml.detection.alerts import (
    causal_ewma,
    extract_alert_episodes,
    persistent_alerts,
)
from ml.detection.common import fit_scaler
from ml.detection.evaluate import (
    evaluate_detection,
)
from ml.detection.pca_detector import (
    fit_pca_detector,
    score_pca_detector,
)
from ml.temporal.detector import (
    TemporalDetector,
    prepare_temporal_batch,
    score_temporal_detector,
)
from ml.temporal.tcn import (
    TemporalForecastConfig,
)
from ml.temporal.training import (
    TemporalTrainingConfig,
    train_tcn_forecaster,
)

ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = yaml.safe_load(
            handle
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            f"Expected mapping in {path}."
        )

    return payload


def _load_partition(
    path: Path,
) -> pd.DataFrame:
    frame = pd.read_parquet(
        path
    )
    frame.index = pd.DatetimeIndex(
        frame.index,
        name="timestamp",
    )
    return frame.sort_index()


def _serializable_metrics(
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return json.loads(
        json.dumps(
            metrics,
            default=str,
        )
    )


def main() -> None:
    metropt = _load_yaml(
        ROOT
        / "configs"
        / "metropt.yaml"
    )

    detection = _load_yaml(
        ROOT
        / "configs"
        / "detection.yaml"
    )

    temporal_config = _load_yaml(
        ROOT
        / "configs"
        / "temporal.yaml"
    )

    preprocessing = metropt[
        "preprocessing"
    ]
    incidents = metropt[
        "dataset"
    ][
        "reported_incidents"
    ]

    alert_config = detection[
        "alerting"
    ]
    evaluation_config = detection[
        "evaluation"
    ]

    temporal_section = temporal_config[
        "temporal"
    ]
    model_section = temporal_config[
        "model"
    ]
    training_section = temporal_config[
        "training"
    ]
    outputs = temporal_config[
        "outputs"
    ]

    train = _load_partition(
        ROOT
        / preprocessing[
            "train_file"
        ]
    )

    calibration = _load_partition(
        ROOT
        / preprocessing[
            "calibration_file"
        ]
    )

    test = _load_partition(
        ROOT
        / preprocessing[
            "test_file"
        ]
    )

    features = model_feature_columns(
        train.columns
    )

    if len(features) != 63:
        raise RuntimeError(
            "Expected the frozen 63-feature "
            f"representation, got {len(features)}."
        )

    scaler_name = str(
        temporal_section[
            "scaler"
        ]
    )

    if scaler_name != "robust":
        raise ValueError(
            "Initial TCN benchmark is frozen "
            "to the train-only robust scaler."
        )

    scaler = fit_scaler(
        train,
        features,
        method="robust",
    )

    sequence_length = int(
        temporal_section[
            "sequence_length"
        ]
    )

    bin_minutes = int(
        evaluation_config[
            "bin_minutes"
        ]
    )

    train_batch = (
        prepare_temporal_batch(
            train,
            features=features,
            scaler=scaler,
            sequence_length=(
                sequence_length
            ),
            bin_minutes=bin_minutes,
        )
    )

    model_config = (
        TemporalForecastConfig(
            input_dim=len(features),
            hidden_dim=int(
                model_section[
                    "hidden_dim"
                ]
            ),
            kernel_size=int(
                model_section[
                    "kernel_size"
                ]
            ),
            dilations=tuple(
                int(value)
                for value
                in model_section[
                    "dilations"
                ]
            ),
            dropout=float(
                model_section[
                    "dropout"
                ]
            ),
        )
    )

    training_config = (
        TemporalTrainingConfig(
            seed=int(
                training_section[
                    "seed"
                ]
            ),
            epochs=int(
                training_section[
                    "epochs"
                ]
            ),
            batch_size=int(
                training_section[
                    "batch_size"
                ]
            ),
            learning_rate=float(
                training_section[
                    "learning_rate"
                ]
            ),
            weight_decay=float(
                training_section[
                    "weight_decay"
                ]
            ),
            huber_delta=float(
                training_section[
                    "huber_delta"
                ]
            ),
            validation_fraction=float(
                training_section[
                    "validation_fraction"
                ]
            ),
            patience=int(
                training_section[
                    "patience"
                ]
            ),
            min_delta=float(
                training_section[
                    "min_delta"
                ]
            ),
            device=str(
                training_section[
                    "device"
                ]
            ),
        )
    )

    result = train_tcn_forecaster(
        train_batch,
        model_config=model_config,
        training_config=(
            training_config
        ),
    )

    detector = TemporalDetector(
        features=features,
        scaler=scaler,
        model=result.model,
        sequence_length=(
            sequence_length
        ),
        bin_minutes=bin_minutes,
    )

    calibration_scores = (
        score_temporal_detector(
            calibration,
            detector,
            device=result.device,
            batch_size=1024,
        )
    )

    test_scores = (
        score_temporal_detector(
            test,
            detector,
            device=result.device,
            batch_size=1024,
        )
    )

    calibration_smoothed = causal_ewma(
        calibration_scores,
        alpha=float(
            alert_config[
                "ewma_alpha"
            ]
        ),
        reset_gap_minutes=int(
            alert_config[
                "reset_gap_minutes"
            ]
        ),
    )

    threshold = float(
        calibration_smoothed.quantile(
            float(
                alert_config[
                    "threshold_quantile"
                ]
            ),
            interpolation="higher",
        )
    )

    test_smoothed = causal_ewma(
        test_scores,
        alpha=float(
            alert_config[
                "ewma_alpha"
            ]
        ),
        reset_gap_minutes=int(
            alert_config[
                "reset_gap_minutes"
            ]
        ),
    )

    threshold_hits, alerts = (
        persistent_alerts(
            test_smoothed,
            threshold=threshold,
            required_hits=int(
                alert_config[
                    "persistence_hits"
                ]
            ),
            window_bins=int(
                alert_config[
                    "persistence_window"
                ]
            ),
            reset_gap_minutes=int(
                alert_config[
                    "reset_gap_minutes"
                ]
            ),
        )
    )

    episodes = extract_alert_episodes(
        alerts,
        merge_minutes=int(
            alert_config[
                "merge_minutes"
            ]
        ),
        reset_gap_minutes=int(
            alert_config[
                "reset_gap_minutes"
            ]
        ),
    )

    metrics = evaluate_detection(
        scores=test_smoothed,
        alerts=alerts,
        episodes=episodes,
        incidents=incidents,
        early_warning_hours=int(
            evaluation_config[
                "early_warning_hours"
            ]
        ),
        late_tolerance_hours=int(
            evaluation_config[
                "late_tolerance_hours"
            ]
        ),
        bin_minutes=bin_minutes,
    )

    pca_config = detection[
        "pca"
    ]

    aligned_pca = fit_pca_detector(
        train,
        features,
        variance_retained=float(
            pca_config[
                "variance_retained"
            ]
        ),
        scaler_name="robust",
    )

    aligned_pca_calibration = (
        score_pca_detector(
            calibration,
            aligned_pca,
        )
        .reindex(
            calibration_scores.index
        )
    )

    aligned_pca_test = (
        score_pca_detector(
            test,
            aligned_pca,
        )
        .reindex(
            test_scores.index
        )
    )

    if (
        aligned_pca_calibration
        .isna()
        .any()
        or aligned_pca_test
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Aligned PCA comparison contains "
            "missing scores."
        )

    aligned_pca_calibration_smoothed = (
        causal_ewma(
            aligned_pca_calibration,
            alpha=float(
                alert_config[
                    "ewma_alpha"
                ]
            ),
            reset_gap_minutes=int(
                alert_config[
                    "reset_gap_minutes"
                ]
            ),
        )
    )

    aligned_pca_threshold = float(
        aligned_pca_calibration_smoothed
        .quantile(
            float(
                alert_config[
                    "threshold_quantile"
                ]
            ),
            interpolation="higher",
        )
    )

    aligned_pca_test_smoothed = (
        causal_ewma(
            aligned_pca_test,
            alpha=float(
                alert_config[
                    "ewma_alpha"
                ]
            ),
            reset_gap_minutes=int(
                alert_config[
                    "reset_gap_minutes"
                ]
            ),
        )
    )

    (
        _,
        aligned_pca_alerts,
    ) = persistent_alerts(
        aligned_pca_test_smoothed,
        threshold=(
            aligned_pca_threshold
        ),
        required_hits=int(
            alert_config[
                "persistence_hits"
            ]
        ),
        window_bins=int(
            alert_config[
                "persistence_window"
            ]
        ),
        reset_gap_minutes=int(
            alert_config[
                "reset_gap_minutes"
            ]
        ),
    )

    aligned_pca_episodes = (
        extract_alert_episodes(
            aligned_pca_alerts,
            merge_minutes=int(
                alert_config[
                    "merge_minutes"
                ]
            ),
            reset_gap_minutes=int(
                alert_config[
                    "reset_gap_minutes"
                ]
            ),
        )
    )

    aligned_pca_metrics = (
        evaluate_detection(
            scores=(
                aligned_pca_test_smoothed
            ),
            alerts=(
                aligned_pca_alerts
            ),
            episodes=(
                aligned_pca_episodes
            ),
            incidents=incidents,
            early_warning_hours=int(
                evaluation_config[
                    "early_warning_hours"
                ]
            ),
            late_tolerance_hours=int(
                evaluation_config[
                    "late_tolerance_hours"
                ]
            ),
            bin_minutes=bin_minutes,
        )
    )

    benchmark_path = (
        ROOT
        / "docs"
        / "detector_benchmark.json"
    )

    with benchmark_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        frozen_benchmark = json.load(
            handle
        )

    frozen_pca_metrics = frozen_benchmark[
        "summary"
    ][
        "pca_robust"
    ]

    report = {
        "evidence_class": "REAL",
        "causal_claim": False,
        "evaluation_protocol": {
            "training_partition_only": True,
            "calibration_threshold_only": True,
            "test_final_evaluation_only": True,
            "sequence_length_bins": (
                sequence_length
            ),
            "sequence_minutes": (
                sequence_length
                * bin_minutes
            ),
            "feature_count": len(
                features
            ),
            "train_sequence_targets": (
                len(train_batch.targets)
            ),
            "calibration_score_rows": (
                len(calibration_scores)
            ),
            "test_score_rows": (
                len(test_scores)
            ),
            "scaler": "robust",
            "ewma_alpha": float(
                alert_config[
                    "ewma_alpha"
                ]
            ),
            "threshold_quantile": float(
                alert_config[
                    "threshold_quantile"
                ]
            ),
            "persistence_hits": int(
                alert_config[
                    "persistence_hits"
                ]
            ),
            "persistence_window": int(
                alert_config[
                    "persistence_window"
                ]
            ),
        },
        "training": {
            "device": result.device,
            "fit_samples": (
                result.fit_samples
            ),
            "validation_samples": (
                result.validation_samples
            ),
            "best_epoch": (
                result.best_epoch
            ),
            "epochs_ran": (
                result.epochs_ran
            ),
            "best_validation_loss": (
                min(
                    result
                    .validation_losses
                )
            ),
            "final_train_loss": (
                result
                .train_losses[-1]
            ),
        },
        "model": {
            "name": (
                "tcn_next_state_forecaster"
            ),
            "hidden_dim": (
                model_config.hidden_dim
            ),
            "kernel_size": (
                model_config.kernel_size
            ),
            "dilations": list(
                model_config.dilations
            ),
            "dropout": (
                model_config.dropout
            ),
            "threshold": threshold,
        },
        "tcn_metrics": (
            _serializable_metrics(
                metrics
            )
        ),
        "aligned_pca_robust": (
            _serializable_metrics(
                aligned_pca_metrics
            )
        ),
        "aligned_pca_threshold": (
            aligned_pca_threshold
        ),
        "frozen_pca_reference": (
            frozen_pca_metrics
        ),
        "comparison": {
            "comparison_basis": (
                "same_causal_target_timestamps"
            ),
            "false_alerts_per_24h_delta": (
                metrics[
                    "false_alerts_per_24h"
                ]
                - aligned_pca_metrics[
                    "false_alerts_per_24h"
                ]
            ),
            "pr_auc_delta": (
                metrics[
                    "pr_auc"
                ]
                - aligned_pca_metrics[
                    "pr_auc"
                ]
            ),
            "timely_incident_recall_delta": (
                metrics[
                    "timely_incident_recall"
                ]
                - aligned_pca_metrics[
                    "timely_incident_recall"
                ]
            ),
            "pre_onset_incident_recall_delta": (
                metrics[
                    "pre_onset_incident_recall"
                ]
                - aligned_pca_metrics[
                    "pre_onset_incident_recall"
                ]
            ),
        },
    }

    checkpoint_path = (
        ROOT
        / outputs[
            "checkpoint_file"
        ]
    )
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "state_dict": {
                key: value
                .detach()
                .cpu()
                for key, value
                in result.model
                .state_dict()
                .items()
            },
            "features": features,
            "sequence_length": (
                sequence_length
            ),
            "bin_minutes": (
                bin_minutes
            ),
            "model_config": {
                "input_dim": (
                    model_config.input_dim
                ),
                "hidden_dim": (
                    model_config.hidden_dim
                ),
                "kernel_size": (
                    model_config.kernel_size
                ),
                "dilations": list(
                    model_config.dilations
                ),
                "dropout": (
                    model_config.dropout
                ),
            },
            "scaler_center": (
                scaler.center_
            ),
            "scaler_scale": (
                scaler.scale_
            ),
            "threshold": threshold,
        },
        checkpoint_path,
    )

    scores_path = (
        ROOT
        / outputs[
            "scores_file"
        ]
    )
    scores_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        {
            "raw_score": (
                test_scores
            ),
            "smoothed_score": (
                test_smoothed
            ),
            "threshold_hit": (
                threshold_hits
            ),
            "alert": alerts,
        }
    ).to_parquet(
        scores_path
    )

    report_path = (
        ROOT
        / outputs[
            "benchmark_file"
        ]
    )
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "device": result.device,
                "best_epoch": (
                    result.best_epoch
                ),
                "epochs_ran": (
                    result.epochs_ran
                ),
                "tcn": {
                    key: metrics[key]
                    for key in (
                        "timely_incident_recall",
                        "pre_onset_incident_recall",
                        "false_alerts_per_24h",
                        "time_in_alert_fraction",
                        "pr_auc",
                    )
                },
                "aligned_pca_robust": {
                    key: aligned_pca_metrics[
                        key
                    ]
                    for key in (
                        "timely_incident_recall",
                        "pre_onset_incident_recall",
                        "false_alerts_per_24h",
                        "time_in_alert_fraction",
                        "pr_auc",
                    )
                },
                "comparison": (
                    report[
                        "comparison"
                    ]
                ),
                "report": str(
                    report_path
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
