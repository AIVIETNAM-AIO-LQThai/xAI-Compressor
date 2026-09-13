from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from ml.data.features import (
    model_feature_columns,
)
from ml.data.schema import (
    SENSOR_COLUMNS,
    normalize_raw_frame,
)
from ml.detection.alerts import (
    causal_ewma,
    extract_alert_episodes,
    persistent_alerts,
)
from ml.detection.common import (
    fit_scaler,
    transform_frame,
)
from ml.detection.evaluate import (
    evaluate_detection,
)
from ml.detection.pca_detector import (
    fit_pca_detector,
    score_pca_detector,
)
from ml.temporal.raw_context import (
    RawContextBatch,
    build_raw_context_batch,
)
from ml.temporal.raw_context_tcn import (
    RawContextTCNConfig,
    RawContextTCNForecaster,
)

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(
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


def load_processed(
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


def select_raw_split(
    frame: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    start_time = pd.Timestamp(
        start
    )
    end_time = pd.Timestamp(
        end
    )

    return frame.loc[
        (frame.index >= start_time)
        & (frame.index <= end_time)
    ].copy()


def resample_raw_context_frame(
    frame: pd.DataFrame,
    *,
    step_seconds: int,
) -> pd.DataFrame:
    """Downsample raw telemetry onto the causal grid expected by the TCN.

    Each output timestamp is the right edge of a fixed-width bin, and the
    value is the latest observation available at or before that timestamp.
    Empty/incomplete bins are dropped rather than interpolated, so real gaps
    remain gaps and build_raw_context_batch() can reject windows that cross
    them.
    """
    if step_seconds <= 0:
        raise ValueError(
            "step_seconds must be positive."
        )

    if not isinstance(
        frame.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "raw frame index must be a DatetimeIndex."
        )

    if frame.empty:
        raise ValueError(
            "Cannot resample an empty raw frame."
        )

    frame = frame.sort_index()

    if frame.index.has_duplicates:
        frame = frame.loc[
            ~frame.index.duplicated(
                keep="last"
            )
        ]

    step = pd.Timedelta(
        seconds=step_seconds
    )

    sampled = (
        frame.resample(
            step,
            origin="epoch",
            label="right",
            closed="right",
        )
        .last()
        .dropna(
            subset=list(
                SENSOR_COLUMNS
            ),
            how="any",
        )
    )

    if sampled.empty:
        raise ValueError(
            "Raw resampling produced no usable rows."
        )

    sampled.index = pd.DatetimeIndex(
        sampled.index,
        name="timestamp",
    )

    return sampled


def resolve_device(
    requested: str,
) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(
        requested
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is "
            "not available."
        )

    return device


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def build_batch(
    raw_frame: pd.DataFrame,
    target_frame: pd.DataFrame,
    *,
    raw_scaler,
    target_scaler,
    target_features: list[str],
    context_samples: int,
    raw_step_seconds: int,
    forecast_gap_minutes: int,
) -> RawContextBatch:
    raw_values = transform_frame(
        raw_frame,
        SENSOR_COLUMNS,
        raw_scaler,
    )

    target_values = transform_frame(
        target_frame,
        target_features,
        target_scaler,
    )

    return build_raw_context_batch(
        raw_values,
        pd.DatetimeIndex(
            raw_frame.index
        ),
        target_values,
        pd.DatetimeIndex(
            target_frame.index
        ),
        context_samples=(
            context_samples
        ),
        raw_step=pd.Timedelta(
            seconds=raw_step_seconds
        ),
        forecast_gap=pd.Timedelta(
            minutes=forecast_gap_minutes
        ),
    )


def train_model(
    batch: RawContextBatch,
    *,
    model: RawContextTCNForecaster,
    training: dict[str, Any],
) -> tuple[
    RawContextTCNForecaster,
    dict[str, Any],
]:
    seed = int(
        training[
            "seed"
        ]
    )
    set_seed(
        seed
    )

    device = resolve_device(
        str(
            training[
                "device"
            ]
        )
    )

    sample_count = len(
        batch.inputs
    )

    validation_count = max(
        1,
        round(
            sample_count
            * float(
                training[
                    "validation_fraction"
                ]
            )
        ),
    )

    fit_count = (
        sample_count
        - validation_count
    )

    if fit_count < 2:
        raise ValueError(
            "Not enough raw-context samples "
            "for train/validation."
        )

    fit_dataset = TensorDataset(
        torch.from_numpy(
            batch.inputs[:fit_count]
        ),
        torch.from_numpy(
            batch.targets[:fit_count]
        ),
    )

    validation_dataset = TensorDataset(
        torch.from_numpy(
            batch.inputs[fit_count:]
        ),
        torch.from_numpy(
            batch.targets[fit_count:]
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(
        seed
    )

    fit_loader = DataLoader(
        fit_dataset,
        batch_size=int(
            training[
                "batch_size"
            ]
        ),
        shuffle=True,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=int(
            training[
                "batch_size"
            ]
        ),
        shuffle=False,
    )

    model = model.to(
        device
    )

    criterion = nn.HuberLoss(
        delta=float(
            training[
                "huber_delta"
            ]
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            training[
                "learning_rate"
            ]
        ),
        weight_decay=float(
            training[
                "weight_decay"
            ]
        ),
    )

    best_validation = float(
        "inf"
    )
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    train_losses: list[
        float
    ] = []
    validation_losses: list[
        float
    ] = []

    epochs = int(
        training[
            "epochs"
        ]
    )
    patience = int(
        training[
            "patience"
        ]
    )
    min_delta = float(
        training[
            "min_delta"
        ]
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()
        weighted_loss = 0.0
        seen = 0

        for inputs, targets in (
            fit_loader
        ):
            inputs = inputs.to(
                device
            )
            targets = targets.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss = criterion(
                model(
                    inputs
                ),
                targets,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            count = int(
                inputs.shape[0]
            )

            weighted_loss += (
                float(
                    loss.item()
                )
                * count
            )
            seen += count

        train_loss = (
            weighted_loss
            / seen
        )

        model.eval()
        validation_total = 0.0
        validation_seen = 0

        with torch.inference_mode():
            for inputs, targets in (
                validation_loader
            ):
                inputs = inputs.to(
                    device
                )
                targets = targets.to(
                    device
                )

                loss = criterion(
                    model(
                        inputs
                    ),
                    targets,
                )

                count = int(
                    inputs.shape[0]
                )

                validation_total += (
                    float(
                        loss.item()
                    )
                    * count
                )
                validation_seen += (
                    count
                )

        validation_loss = (
            validation_total
            / validation_seen
        )

        train_losses.append(
            float(
                train_loss
            )
        )

        validation_losses.append(
            float(
                validation_loss
            )
        )

        if (
            validation_loss
            < best_validation
            - min_delta
        ):
            best_validation = (
                validation_loss
            )
            best_epoch = epoch
            best_state = {
                key: value
                .detach()
                .cpu()
                .clone()
                for key, value
                in model
                .state_dict()
                .items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= patience
        ):
            break

    if best_state is None:
        raise RuntimeError(
            "Raw-context training produced "
            "no valid state."
        )

    model.load_state_dict(
        best_state
    )
    model = model.cpu()

    return (
        model,
        {
            "device": str(
                device
            ),
            "fit_samples": (
                fit_count
            ),
            "validation_samples": (
                validation_count
            ),
            "best_epoch": (
                best_epoch
            ),
            "epochs_ran": len(
                train_losses
            ),
            "best_validation_loss": (
                min(
                    validation_losses
                )
            ),
            "final_train_loss": (
                train_losses[-1]
            ),
            "best_state": (
                best_state
            ),
        },
    )


def score_batch(
    batch: RawContextBatch,
    *,
    model: RawContextTCNForecaster,
    device: str,
    batch_size: int = 256,
) -> pd.Series:
    torch_device = torch.device(
        device
    )

    model = model.to(
        torch_device
    )
    model.eval()

    scores: list[
        np.ndarray
    ] = []

    with torch.inference_mode():
        for start in range(
            0,
            len(batch.inputs),
            batch_size,
        ):
            stop = min(
                len(batch.inputs),
                start + batch_size,
            )

            inputs = torch.from_numpy(
                batch.inputs[start:stop]
            ).to(
                torch_device
            )

            targets = torch.from_numpy(
                batch.targets[start:stop]
            ).to(
                torch_device
            )

            prediction = model(
                inputs
            )

            residual = (
                targets
                - prediction
            )

            score = torch.mean(
                residual.square(),
                dim=1,
            )

            scores.append(
                score.cpu().numpy()
            )

    return pd.Series(
        np.concatenate(
            scores
        ),
        index=batch.target_index,
        name="raw_score",
        dtype=float,
    )


def evaluate_scores(
    calibration_scores: pd.Series,
    test_scores: pd.Series,
    *,
    alert_config: dict[str, Any],
    evaluation_config: dict[str, Any],
    incidents: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    float,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    calibration_smoothed = (
        causal_ewma(
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
        bin_minutes=int(
            evaluation_config[
                "bin_minutes"
            ]
        ),
    )

    return (
        metrics,
        threshold,
        threshold_hits,
        alerts,
        test_smoothed,
    )


def main() -> None:
    metropt = load_yaml(
        ROOT
        / "configs"
        / "metropt.yaml"
    )

    detection = load_yaml(
        ROOT
        / "configs"
        / "detection.yaml"
    )

    raw_config = load_yaml(
        ROOT
        / "configs"
        / "raw_context_tcn.yaml"
    )

    preprocessing = metropt[
        "preprocessing"
    ]
    split = metropt[
        "split"
    ]
    dataset = metropt[
        "dataset"
    ]

    train_target = load_processed(
        ROOT
        / preprocessing[
            "train_file"
        ]
    )

    calibration_target = (
        load_processed(
            ROOT
            / preprocessing[
                "calibration_file"
            ]
        )
    )

    test_target = load_processed(
        ROOT
        / preprocessing[
            "test_file"
        ]
    )

    target_features = (
        model_feature_columns(
            train_target.columns
        )
    )

    if len(
        target_features
    ) != 63:
        raise RuntimeError(
            "Expected 63 target features, "
            f"got {len(target_features)}."
        )

    raw_path = (
        ROOT
        / dataset[
            "raw_csv"
        ]
    )

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw MetroPT CSV not found: "
            f"{raw_path}"
        )

    raw = normalize_raw_frame(
        pd.read_csv(
            raw_path
        )
    ).set_index(
        "timestamp"
    )

    raw_section = raw_config[
        "raw_context"
    ]

    raw_step_seconds = int(
        raw_section[
            "raw_step_seconds"
        ]
    )

    train_raw = resample_raw_context_frame(
        select_raw_split(
            raw,
            **split[
                "train"
            ],
        ),
        step_seconds=raw_step_seconds,
    )

    calibration_raw = (
        resample_raw_context_frame(
            select_raw_split(
                raw,
                **split[
                    "calibration"
                ],
            ),
            step_seconds=raw_step_seconds,
        )
    )

    test_raw = resample_raw_context_frame(
        select_raw_split(
            raw,
            **split[
                "test"
            ],
        ),
        step_seconds=raw_step_seconds,
    )

    raw_scaler = fit_scaler(
        train_raw,
        SENSOR_COLUMNS,
        method="robust",
    )

    target_scaler = fit_scaler(
        train_target,
        target_features,
        method="robust",
    )

    context_minutes = int(
        raw_section[
            "context_minutes"
        ]
    )

    forecast_gap_minutes = int(
        raw_section[
            "forecast_gap_minutes"
        ]
    )

    context_samples = (
        context_minutes
        * 60
        // raw_step_seconds
    )

    build_kwargs = {
        "raw_scaler": raw_scaler,
        "target_scaler": (
            target_scaler
        ),
        "target_features": (
            target_features
        ),
        "context_samples": (
            context_samples
        ),
        "raw_step_seconds": (
            raw_step_seconds
        ),
        "forecast_gap_minutes": (
            forecast_gap_minutes
        ),
    }

    train_batch = build_batch(
        train_raw,
        train_target,
        **build_kwargs,
    )

    calibration_batch = (
        build_batch(
            calibration_raw,
            calibration_target,
            **build_kwargs,
        )
    )

    test_batch = build_batch(
        test_raw,
        test_target,
        **build_kwargs,
    )

    model_section = raw_config[
        "model"
    ]

    model_config = (
        RawContextTCNConfig(
            input_dim=len(
                SENSOR_COLUMNS
            ),
            target_dim=len(
                target_features
            ),
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

    model = (
        RawContextTCNForecaster(
            model_config
        )
    )

    model, training_summary = (
        train_model(
            train_batch,
            model=model,
            training=raw_config[
                "training"
            ],
        )
    )

    calibration_scores = (
        score_batch(
            calibration_batch,
            model=model,
            device=(
                training_summary[
                    "device"
                ]
            ),
        )
    )

    test_scores = score_batch(
        test_batch,
        model=model,
        device=(
            training_summary[
                "device"
            ]
        ),
    )

    alert_config = detection[
        "alerting"
    ]

    evaluation_config = detection[
        "evaluation"
    ]

    (
        raw_metrics,
        threshold,
        threshold_hits,
        alerts,
        test_smoothed,
    ) = evaluate_scores(
        calibration_scores,
        test_scores,
        alert_config=(
            alert_config
        ),
        evaluation_config=(
            evaluation_config
        ),
        incidents=dataset[
            "reported_incidents"
        ],
    )

    pca = fit_pca_detector(
        train_target,
        target_features,
        variance_retained=float(
            detection[
                "pca"
            ][
                "variance_retained"
            ]
        ),
        scaler_name="robust",
    )

    pca_calibration = (
        score_pca_detector(
            calibration_target,
            pca,
        ).reindex(
            calibration_scores.index
        )
    )

    pca_test = (
        score_pca_detector(
            test_target,
            pca,
        ).reindex(
            test_scores.index
        )
    )

    if (
        pca_calibration
        .isna()
        .any()
        or pca_test
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Aligned PCA contains missing "
            "scores."
        )

    (
        pca_metrics,
        pca_threshold,
        _,
        _,
        _,
    ) = evaluate_scores(
        pca_calibration,
        pca_test,
        alert_config=(
            alert_config
        ),
        evaluation_config=(
            evaluation_config
        ),
        incidents=dataset[
            "reported_incidents"
        ],
    )

    report = {
        "evidence_class": "REAL",
        "causal_claim": False,
        "experiment_question": (
            "Does raw 10-second causal context "
            "improve anomaly ranking compared "
            "with PCA on the same 5-minute "
            "target timestamps?"
        ),
        "evaluation_protocol": {
            "raw_input_features": (
                list(
                    SENSOR_COLUMNS
                )
            ),
            "raw_input_feature_count": (
                len(
                    SENSOR_COLUMNS
                )
            ),
            "target_feature_count": (
                len(
                    target_features
                )
            ),
            "raw_step_seconds": (
                raw_step_seconds
            ),
            "context_minutes": (
                context_minutes
            ),
            "context_samples": (
                context_samples
            ),
            "forecast_gap_minutes": (
                forecast_gap_minutes
            ),
            "train_examples": (
                len(
                    train_batch.inputs
                )
            ),
            "calibration_examples": (
                len(
                    calibration_batch.inputs
                )
            ),
            "test_examples": (
                len(
                    test_batch.inputs
                )
            ),
            "comparison_basis": (
                "same_raw_context_valid_"
                "target_timestamps"
            ),
            "input_scaler": "robust",
            "target_scaler": "robust",
            "threshold_quantile": (
                float(
                    alert_config[
                        "threshold_quantile"
                    ]
                )
            ),
            "ewma_alpha": float(
                alert_config[
                    "ewma_alpha"
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
            key: value
            for key, value
            in training_summary.items()
            if key != "best_state"
        },
        "model": {
            "name": (
                "raw_context_causal_tcn_"
                "to_5min_feature_forecast"
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
            "threshold": (
                threshold
            ),
        },
        "raw_context_tcn_metrics": (
            raw_metrics
        ),
        "aligned_pca_robust": (
            pca_metrics
        ),
        "aligned_pca_threshold": (
            pca_threshold
        ),
        "comparison_vs_pca": {
            "false_alerts_per_24h_delta": (
                raw_metrics[
                    "false_alerts_per_24h"
                ]
                - pca_metrics[
                    "false_alerts_per_24h"
                ]
            ),
            "pr_auc_delta": (
                raw_metrics[
                    "pr_auc"
                ]
                - pca_metrics[
                    "pr_auc"
                ]
            ),
            "timely_incident_recall_delta": (
                raw_metrics[
                    "timely_incident_recall"
                ]
                - pca_metrics[
                    "timely_incident_recall"
                ]
            ),
            "pre_onset_incident_recall_delta": (
                raw_metrics[
                    "pre_onset_incident_recall"
                ]
                - pca_metrics[
                    "pre_onset_incident_recall"
                ]
            ),
        },
    }

    outputs = raw_config[
        "outputs"
    ]

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
            "state_dict": (
                training_summary[
                    "best_state"
                ]
            ),
            "raw_features": list(
                SENSOR_COLUMNS
            ),
            "target_features": (
                target_features
            ),
            "context_samples": (
                context_samples
            ),
            "raw_step_seconds": (
                raw_step_seconds
            ),
            "forecast_gap_minutes": (
                forecast_gap_minutes
            ),
            "model_config": {
                "input_dim": (
                    model_config.input_dim
                ),
                "target_dim": (
                    model_config.target_dim
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
            "raw_scaler_center": (
                raw_scaler.center_
            ),
            "raw_scaler_scale": (
                raw_scaler.scale_
            ),
            "target_scaler_center": (
                target_scaler.center_
            ),
            "target_scaler_scale": (
                target_scaler.scale_
            ),
            "threshold": (
                threshold
            ),
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
                "device": (
                    training_summary[
                        "device"
                    ]
                ),
                "best_epoch": (
                    training_summary[
                        "best_epoch"
                    ]
                ),
                "epochs_ran": (
                    training_summary[
                        "epochs_ran"
                    ]
                ),
                "examples": {
                    "train": len(
                        train_batch.inputs
                    ),
                    "calibration": len(
                        calibration_batch.inputs
                    ),
                    "test": len(
                        test_batch.inputs
                    ),
                },
                "raw_context_tcn": {
                    key: (
                        raw_metrics[
                            key
                        ]
                    )
                    for key in (
                        "timely_incident_recall",
                        "pre_onset_incident_recall",
                        "false_alerts_per_24h",
                        "time_in_alert_fraction",
                        "pr_auc",
                    )
                },
                "aligned_pca_robust": {
                    key: (
                        pca_metrics[
                            key
                        ]
                    )
                    for key in (
                        "timely_incident_recall",
                        "pre_onset_incident_recall",
                        "false_alerts_per_24h",
                        "time_in_alert_fraction",
                        "pr_auc",
                    )
                },
                "comparison_vs_pca": (
                    report[
                        "comparison_vs_pca"
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
