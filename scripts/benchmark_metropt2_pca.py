from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.data.metropt2_features import (
    model_feature_columns,
)
from ml.detection.pca_research_benchmark import (
    AlertProtocol,
    benchmark_pca_variant,
)

ROOT = Path(__file__).resolve().parents[1]
METROPT2_CONFIG_PATH = (
    ROOT
    / "configs"
    / "metropt2.yaml"
)


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
            f"{path} must contain a mapping."
        )

    return payload


def _load_json(
    path: Path,
) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(
            handle
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            f"{path} must contain a JSON object."
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


def _assert_preprocessing_identity(
    *,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    preprocessing_report: dict[str, Any],
) -> None:
    expected = (
        preprocessing_report[
            "splits"
        ]
    )

    observed = {
        "train": len(train),
        "calibration": len(
            calibration
        ),
        "test": len(test),
    }

    for split_name, row_count in (
        observed.items()
    ):
        expected_count = int(
            expected[
                split_name
            ][
                "valid_bins"
            ]
        )

        if row_count != expected_count:
            raise RuntimeError(
                "Processed partition does "
                "not match frozen "
                "preprocessing report: "
                f"{split_name} expected "
                f"{expected_count}, "
                f"observed {row_count}."
            )


def main() -> None:
    metropt2 = _load_yaml(
        METROPT2_CONFIG_PATH
    )

    preprocessing = (
        metropt2[
            "preprocessing"
        ]
    )

    benchmark_config = (
        metropt2[
            "benchmark"
        ]
    )

    detection = _load_yaml(
        ROOT
        / benchmark_config[
            "detection_config"
        ]
    )

    preprocessing_report = (
        _load_json(
            ROOT
            / benchmark_config[
                "preprocessing_report"
            ]
        )
    )

    baseline = _load_json(
        ROOT
        / benchmark_config[
            "metropt3_concentration_report"
        ]
    )

    train = _load_partition(
        ROOT
        / preprocessing[
            "train_file"
        ]
    )

    calibration = (
        _load_partition(
            ROOT
            / preprocessing[
                "calibration_file"
            ]
        )
    )

    test = _load_partition(
        ROOT
        / preprocessing[
            "test_file"
        ]
    )

    _assert_preprocessing_identity(
        train=train,
        calibration=calibration,
        test=test,
        preprocessing_report=(
            preprocessing_report
        ),
    )

    alerting = detection[
        "alerting"
    ]

    evaluation = detection[
        "evaluation"
    ]

    pca = detection[
        "pca"
    ]

    protocol = AlertProtocol(
        ewma_alpha=float(
            alerting[
                "ewma_alpha"
            ]
        ),
        threshold_quantile=float(
            alerting[
                "threshold_quantile"
            ]
        ),
        persistence_hits=int(
            alerting[
                "persistence_hits"
            ]
        ),
        persistence_window=int(
            alerting[
                "persistence_window"
            ]
        ),
        reset_gap_minutes=int(
            alerting[
                "reset_gap_minutes"
            ]
        ),
        merge_minutes=int(
            alerting[
                "merge_minutes"
            ]
        ),
        early_warning_hours=int(
            evaluation[
                "early_warning_hours"
            ]
        ),
        late_tolerance_hours=int(
            evaluation[
                "late_tolerance_hours"
            ]
        ),
        bin_minutes=int(
            evaluation[
                "bin_minutes"
            ]
        ),
    )

    incidents = metropt2[
        "dataset"
    ][
        "reported_incidents"
    ]

    variant_reports = {}

    for variant_name, variant in (
        preprocessing[
            "feature_variants"
        ].items()
    ):
        features = (
            model_feature_columns(
                train.columns,
                excluded_groups=(
                    variant[
                        "excluded_groups"
                    ]
                ),
            )
        )

        expected_feature_count = (
            preprocessing_report[
                "feature_variants"
            ][
                variant_name
            ][
                "model_feature_count"
            ]
        )

        if (
            len(features)
            != expected_feature_count
        ):
            raise RuntimeError(
                "Feature variant no longer "
                "matches frozen "
                "preprocessing report: "
                f"{variant_name} expected "
                f"{expected_feature_count}, "
                f"observed {len(features)}."
            )

        print(
            "Running "
            f"{variant_name} "
            f"({len(features)} features)..."
        )

        variant_reports[
            variant_name
        ] = (
            benchmark_pca_variant(
                train=train,
                calibration=calibration,
                test=test,
                features=features,
                incidents=incidents,
                protocol=protocol,
                variance_retained=float(
                    pca[
                        "variance_retained"
                    ]
                ),
                scaler_name=str(
                    pca[
                        "scaler"
                    ]
                ),
            )
        )

    report = {
        "schema_version": (
            "aeroxai.metropt2_pca_benchmark.v1"
        ),
        "dataset": "MetroPT2",
        "evidence_class": "REAL",
        "causal_claim": False,
        "protocol": {
            "train_rows": len(
                train
            ),
            "calibration_rows": (
                len(
                    calibration
                )
            ),
            "test_rows": len(
                test
            ),
            "variance_retained": float(
                pca[
                    "variance_retained"
                ]
            ),
            "scaler": str(
                pca[
                    "scaler"
                ]
            ),
            "threshold_source": (
                "calibration_only"
            ),
            "threshold_quantile": (
                protocol.threshold_quantile
            ),
            "threshold_interpolation": (
                "higher"
            ),
            "ewma_alpha": (
                protocol.ewma_alpha
            ),
            "persistence_hits": (
                protocol.persistence_hits
            ),
            "persistence_window": (
                protocol.persistence_window
            ),
            "reset_gap_minutes": (
                protocol.reset_gap_minutes
            ),
            "merge_minutes": (
                protocol.merge_minutes
            ),
            "early_warning_hours": (
                protocol.early_warning_hours
            ),
            "late_tolerance_hours": (
                protocol.late_tolerance_hours
            ),
            "bin_minutes": (
                protocol.bin_minutes
            ),
            "detector_test_retuned": (
                False
            ),
        },
        "metropt3_frozen_reference": {
            "dataset": baseline[
                "dataset"
            ],
            "mean_top3_concentration": (
                baseline[
                    "summary"
                ][
                    "mean_top3_concentration"
                ]
            ),
            "mean_effective_group_count": (
                baseline[
                    "summary"
                ][
                    "mean_effective_group_count"
                ]
            ),
            "incident_count": (
                baseline[
                    "summary"
                ][
                    "incident_count"
                ]
            ),
        },
        "variants": variant_reports,
        "interpretation_guardrails": [
            (
                "PCA contribution groups "
                "describe detector evidence, "
                "not physical root cause."
            ),
            (
                "MetroPT2 and MetroPT-3 are "
                "related Porto Metro compressor "
                "benchmarks, not independent "
                "cross-industry validation."
            ),
            (
                "The Flowmeter-inclusive result "
                "must be interpreted alongside "
                "both pre-registered flow "
                "ablations."
            ),
            (
                "No variant may be retuned after "
                "inspection of test performance "
                "within this benchmark."
            ),
        ],
    }

    output_path = (
        ROOT
        / benchmark_config[
            "output_file"
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    compact = {
        "metropt3_reference": (
            report[
                "metropt3_frozen_reference"
            ]
        ),
        "variants": {},
        "output": str(
            output_path
        ),
    }

    for variant_name, variant in (
        variant_reports.items()
    ):
        compact[
            "variants"
        ][
            variant_name
        ] = {
            "model": variant[
                "model"
            ],
            "metrics": {
                key: variant[
                    "metrics"
                ][key]
                for key in (
                    "timely_incident_recall",
                    "anytime_incident_recall",
                    "pre_onset_incident_recall",
                    "incident_overlap_recall",
                    "false_alerts_per_24h",
                    "time_in_alert_fraction",
                    "relevant_episode_precision",
                    "pr_auc",
                    "episodes_total",
                    "false_episodes",
                )
            },
            "incident_results": (
                variant[
                    "metrics"
                ][
                    "incident_results"
                ]
            ),
            "explanation_summary": (
                variant[
                    "explanation_summary"
                ]
            ),
            "incident_explanations": [
                {
                    "id": row[
                        "id"
                    ],
                    "condition": row[
                        "condition"
                    ],
                    "available": row[
                        "explanation_available"
                    ],
                    "timing": row.get(
                        "timing"
                    ),
                    "dominant_group": (
                        None
                        if not row[
                            "explanation_available"
                        ]
                        else row[
                            "concentration"
                        ][
                            "dominant_group"
                        ]
                    ),
                    "top1": (
                        None
                        if not row[
                            "explanation_available"
                        ]
                        else row[
                            "concentration"
                        ][
                            "top1_concentration"
                        ]
                    ),
                    "top3": (
                        None
                        if not row[
                            "explanation_available"
                        ]
                        else row[
                            "concentration"
                        ][
                            "top3_concentration"
                        ]
                    ),
                    "effective_groups": (
                        None
                        if not row[
                            "explanation_available"
                        ]
                        else row[
                            "concentration"
                        ][
                            "effective_group_count"
                        ]
                    ),
                }
                for row in variant[
                    "incident_explanations"
                ]
            ],
        }

    print(
        json.dumps(
            compact,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
