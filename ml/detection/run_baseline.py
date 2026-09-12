from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.data.features import (
    model_feature_columns,
)
from ml.detection.alerts import (
    causal_ewma,
    extract_alert_episodes,
    persistent_alerts,
)
from ml.detection.evaluate import (
    evaluate_detection,
)
from ml.detection.robust_z import (
    fit_robust_z,
    save_model,
    score_robust_z,
)

ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return yaml.safe_load(handle)


def load_partition(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)

    frame.index = pd.DatetimeIndex(
        frame.index,
        name="timestamp",
    )

    return frame.sort_index()


def main() -> None:
    metropt = load_yaml(
        ROOT / "configs" / "metropt.yaml"
    )

    detection = load_yaml(
        ROOT / "configs" / "detection.yaml"
    )

    preprocessing = metropt[
        "preprocessing"
    ]

    detector_config = detection[
        "detector"
    ]

    alert_config = detection[
        "alerting"
    ]

    evaluation_config = detection[
        "evaluation"
    ]

    output_config = detection[
        "outputs"
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

    model = fit_robust_z(
        train,
        feature_columns,
        top_k=detector_config[
            "top_k"
        ],
        min_scale=detector_config[
            "min_scale"
        ],
    )

    calibration_raw, _ = (
        score_robust_z(
            calibration,
            model,
        )
    )

    calibration_smoothed = causal_ewma(
        calibration_raw,
        alpha=alert_config[
            "ewma_alpha"
        ],
        reset_gap_minutes=alert_config[
            "reset_gap_minutes"
        ],
    )

    threshold = float(
        calibration_smoothed.quantile(
            alert_config[
                "threshold_quantile"
            ],
            interpolation="higher",
        )
    )

    test_raw, _ = score_robust_z(
        test,
        model,
    )

    test_smoothed = causal_ewma(
        test_raw,
        alpha=alert_config[
            "ewma_alpha"
        ],
        reset_gap_minutes=alert_config[
            "reset_gap_minutes"
        ],
    )

    threshold_hits, alerts = (
        persistent_alerts(
            test_smoothed,
            threshold=threshold,
            required_hits=alert_config[
                "persistence_hits"
            ],
            window_bins=alert_config[
                "persistence_window"
            ],
            reset_gap_minutes=alert_config[
                "reset_gap_minutes"
            ],
        )
    )

    episodes = extract_alert_episodes(
        alerts,
        merge_minutes=alert_config[
            "merge_minutes"
        ],
        reset_gap_minutes=alert_config[
            "reset_gap_minutes"
        ],
    )

    metrics = evaluate_detection(
        scores=test_smoothed,
        alerts=alerts,
        episodes=episodes,
        incidents=metropt["dataset"][
            "reported_incidents"
        ],
        early_warning_hours=(
            evaluation_config[
                "early_warning_hours"
            ]
        ),
        late_tolerance_hours=(
            evaluation_config[
                "late_tolerance_hours"
            ]
        ),
        bin_minutes=evaluation_config[
            "bin_minutes"
        ],
    )

    metrics["detector"] = {
        "method":
            detector_config["method"],
        "active_feature_count":
            len(model.features),
        "dropped_feature_count":
            len(model.dropped_features),
        "dropped_features":
            model.dropped_features,
        "top_k":
            model.top_k,
        "threshold":
            threshold,
        "threshold_quantile":
            alert_config[
                "threshold_quantile"
            ],
        "calibration_rows":
            len(calibration),
        "test_rows":
            len(test),
    }

    score_table = pd.DataFrame(
        {
            "raw_score":
                test_raw,
            "smoothed_score":
                test_smoothed,
            "threshold":
                threshold,
            "threshold_hit":
                threshold_hits,
            "alert":
                alerts,
        }
    )

    scores_path = (
        ROOT
        / output_config["scores_file"]
    )

    scores_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    score_table.to_parquet(
        scores_path
    )

    model_path = (
        ROOT
        / output_config["model_file"]
    )

    save_model(
        model,
        model_path,
    )

    metrics_path = (
        ROOT
        / output_config["metrics_file"]
    )

    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Active features: "
        f"{len(model.features)}"
    )

    print(
        f"Dropped features: "
        f"{len(model.dropped_features)}"
    )

    print(
        f"Threshold: "
        f"{threshold:.6f}"
    )

    print(
        f"Alert episodes: "
        f"{len(episodes)}"
    )

    print(
        "Timely incident recall: "
        f"{metrics['timely_incident_recall']}"
    )

    print(
        "False alerts / 24h: "
        f"{metrics['false_alerts_per_24h']}"
    )

    print(
        f"Metrics: {metrics_path}"
    )


if __name__ == "__main__":
    main()