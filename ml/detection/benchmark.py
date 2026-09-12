from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.data.features import model_feature_columns
from ml.detection.alerts import (
    causal_ewma,
    extract_alert_episodes,
    persistent_alerts,
)
from ml.detection.evaluate import evaluate_detection
from ml.detection.isolation_forest import (
    fit_isolation_forest,
    score_isolation_forest,
)
from ml.detection.pca_detector import (
    fit_pca_detector,
    score_pca_detector,
)
from ml.detection.robust_z import (
    fit_robust_z,
    score_robust_z,
)

ROOT = Path(__file__).resolve().parents[2]

METROPT_CONFIG_PATH = (
    ROOT
    / "configs"
    / "metropt.yaml"
)

DETECTION_CONFIG_PATH = (
    ROOT
    / "configs"
    / "detection.yaml"
)

OUTPUT_PATH = (
    ROOT
    / "docs"
    / "detector_benchmark.json"
)


def load_yaml(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected mapping in {path}"
        )

    return data


def load_partition(
    path: Path,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)

    frame.index = pd.DatetimeIndex(
        frame.index,
        name="timestamp",
    )

    return frame.sort_index()


def evaluate_scores(
    *,
    calibration_scores: pd.Series,
    test_scores: pd.Series,
    incidents: list[dict[str, Any]],
    alert_config: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    calibration_smoothed = causal_ewma(
        calibration_scores,
        alpha=float(
            alert_config["ewma_alpha"]
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
            alert_config["ewma_alpha"]
        ),
        reset_gap_minutes=int(
            alert_config[
                "reset_gap_minutes"
            ]
        ),
    )

    _, alerts = persistent_alerts(
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

    return metrics, threshold


def run_robust_z(
    *,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    incidents: list[dict[str, Any]],
    detector_config: dict[str, Any],
    alert_config: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    model = fit_robust_z(
        train,
        feature_columns,
        top_k=int(
            detector_config["top_k"]
        ),
        min_scale=float(
            detector_config["min_scale"]
        ),
    )

    calibration_scores, _ = (
        score_robust_z(
            calibration,
            model,
        )
    )

    test_scores, _ = score_robust_z(
        test,
        model,
    )

    metrics, threshold = (
        evaluate_scores(
            calibration_scores=(
                calibration_scores
            ),
            test_scores=test_scores,
            incidents=incidents,
            alert_config=alert_config,
            evaluation_config=(
                evaluation_config
            ),
        )
    )

    metrics["model"] = {
        "name": "robust_z_topk",
        "input_feature_count": len(feature_columns),
        "active_feature_count": len(model.features),
        "dropped_feature_count": len(model.dropped_features),
        "dropped_features": model.dropped_features,
        "top_k": model.top_k,
        "threshold": threshold,
    }

    return metrics


def run_pca(
    *,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    incidents: list[dict[str, Any]],
    pca_config: dict[str, Any],
    alert_config: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    variance_retained = float(pca_config["variance_retained"])
    detector = fit_pca_detector(
        train,
        feature_columns,
        variance_retained=variance_retained,
    )

    calibration_scores = (
        score_pca_detector(
            calibration,
            detector,
        )
    )

    test_scores = score_pca_detector(
        test,
        detector,
    )

    metrics, threshold = (
        evaluate_scores(
            calibration_scores=(
                calibration_scores
            ),
            test_scores=test_scores,
            incidents=incidents,
            alert_config=alert_config,
            evaluation_config=(
                evaluation_config
            ),
        )
    )

    metrics["model"] = {
        "name": "pca_reconstruction",
        "input_feature_count": len(feature_columns),
        "variance_retained": variance_retained,
        "components":
            int(detector.model.n_components_),
        "explained_variance_ratio":
            float(
                detector.model
                .explained_variance_ratio_
                .sum()
            ),
        "threshold": threshold,
    }

    return metrics


def run_isolation_forest(
    *,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    incidents: list[dict[str, Any]],
    alert_config: dict[str, Any],
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    n_estimators = 300
    random_state = 42
    detector = fit_isolation_forest(
        train,
        feature_columns,
        n_estimators=n_estimators,
        random_state=random_state,
    )

    calibration_scores = (
        score_isolation_forest(
            calibration,
            detector,
        )
    )

    test_scores = (
        score_isolation_forest(
            test,
            detector,
        )
    )

    metrics, threshold = (
        evaluate_scores(
            calibration_scores=(
                calibration_scores
            ),
            test_scores=test_scores,
            incidents=incidents,
            alert_config=alert_config,
            evaluation_config=(
                evaluation_config
            ),
        )
    )

    metrics["model"] = {
        "name": "isolation_forest",
        "input_feature_count":
            len(feature_columns),
        "n_estimators": n_estimators,
        "random_state": random_state,
        "threshold":
            threshold,
    }

    return metrics


def compact_summary(
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "timely_incident_recall":
            metrics[
                "timely_incident_recall"
            ],
        "anytime_incident_recall":
            metrics[
                "anytime_incident_recall"
            ],
        "false_alerts_per_24h":
            metrics[
                "false_alerts_per_24h"
            ],
        "time_in_alert_fraction":
            metrics[
                "time_in_alert_fraction"
            ],
        "pr_auc":
            metrics["pr_auc"],
        "episodes_total":
            metrics["episodes_total"],
        "false_episodes":
            metrics["false_episodes"],
        "pre_onset_incident_recall": 
            metrics["pre_onset_incident_recall"],
        "incident_overlap_recall":
            metrics["incident_overlap_recall"],
        "relevant_episode_precision":
            metrics["relevant_episode_precision"],
    }


def main() -> None:
    metropt = load_yaml(
        METROPT_CONFIG_PATH
    )
    detection = load_yaml(
        DETECTION_CONFIG_PATH
    )
    preprocessing = metropt[
        "preprocessing"
    ]

    detector_config = detection["detector"]
    alert_config = detection["alerting"]
    evaluation_config = detection["evaluation"]
    pca_config = detection["pca"]
    incidents = metropt["dataset"][
        "reported_incidents"
    ]

    train = load_partition(
        ROOT
        / preprocessing["train_file"]
    )
    calibration = load_partition(
        ROOT
        / preprocessing[
            "calibration_file"
        ]
    )
    test = load_partition(
        ROOT
        / preprocessing["test_file"]
    )

    feature_columns = (
        model_feature_columns(
            train.columns
        )
    )

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Calibration rows: "
        f"{len(calibration):,}"
    )

    print(
        f"Test rows: "
        f"{len(test):,}"
    )

    print(
        f"Model features: "
        f"{len(feature_columns)}"
    )

    print()
    print("Running Robust-Z...")

    robust_z = run_robust_z(
        train=train,
        calibration=calibration,
        test=test,
        feature_columns=feature_columns,
        incidents=incidents,
        detector_config=detector_config,
        alert_config=alert_config,
        evaluation_config=(
            evaluation_config
        ),
    )

    print("Running PCA...")

    pca = run_pca(
        train=train,
        calibration=calibration,
        test=test,
        feature_columns=feature_columns,
        incidents=incidents,
        pca_config=pca_config,
        alert_config=alert_config,
        evaluation_config=(
            evaluation_config
        ),
    )

    print("Running Isolation Forest...")

    isolation_forest = (
        run_isolation_forest(
            train=train,
            calibration=calibration,
            test=test,
            feature_columns=(
                feature_columns
            ),
            incidents=incidents,
            alert_config=alert_config,
            evaluation_config=(
                evaluation_config
            ),
        )
    )

    report = {
        "evaluation_protocol": {
            "train_rows": len(train),
            "calibration_rows":
                len(calibration),
            "test_rows": len(test),
            "model_feature_count":
                len(feature_columns),
            "threshold_quantile":
                alert_config[
                    "threshold_quantile"
                ],
            "ewma_alpha":
                alert_config[
                    "ewma_alpha"
                ],
            "persistence_hits":
                alert_config[
                    "persistence_hits"
                ],
            "persistence_window":
                alert_config[
                    "persistence_window"
                ],
            "early_warning_hours":
                evaluation_config[
                    "early_warning_hours"
                ],
            "late_tolerance_hours":
                evaluation_config[
                    "late_tolerance_hours"
                ],
        },
        "summary": {
            "robust_z":
                compact_summary(
                    robust_z
                ),
            "pca":
                compact_summary(
                    pca
                ),
            "isolation_forest":
                compact_summary(
                    isolation_forest
                ),
        },
        "details": {
            "robust_z":
                robust_z,
            "pca":
                pca,
            "isolation_forest":
                isolation_forest,
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=== Detector Benchmark ===")

    for name, metrics in report[
        "summary"
    ].items():
        print()
        print(name)
        print(
            "  timely recall: "
            f"{metrics['timely_incident_recall']}"
        )
        print(
            "  false alerts / 24h: "
            f"{metrics['false_alerts_per_24h']}"
        )
        print(
            "  time in alert: "
            f"{metrics['time_in_alert_fraction']}"
        )
        print(
            "  PR-AUC: "
            f"{metrics['pr_auc']}"
        )

    print()
    print(
        f"Benchmark written to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()