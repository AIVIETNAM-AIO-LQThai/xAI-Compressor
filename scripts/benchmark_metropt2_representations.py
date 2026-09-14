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
)
from ml.detection.representation_ablation import (
    matched_timestamp_table,
    run_feature_energy_representation,
    run_pca_representation,
    strip_internal_frames,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "metropt2.yaml"


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


def _protocol(
    detection: dict[str, Any],
) -> AlertProtocol:
    alerting = detection[
        "alerting"
    ]

    evaluation = detection[
        "evaluation"
    ]

    return AlertProtocol(
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


def main() -> None:
    config = _load_yaml(
        CONFIG_PATH
    )

    preprocessing = config[
        "preprocessing"
    ]

    benchmark = config[
        "benchmark"
    ]

    ablation = config[
        "representation_ablation"
    ]

    detection = _load_yaml(
        ROOT
        / benchmark[
            "detection_config"
        ]
    )

    preprocessing_report = (
        _load_json(
            ROOT
            / benchmark[
                "preprocessing_report"
            ]
        )
    )

    pca_benchmark = _load_json(
        ROOT
        / benchmark[
            "output_file"
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

    expected_counts = {
        name: int(
            preprocessing_report[
                "splits"
            ][
                name
            ][
                "valid_bins"
            ]
        )
        for name in (
            "train",
            "calibration",
            "test",
        )
    }

    observed_counts = {
        "train": len(train),
        "calibration": len(
            calibration
        ),
        "test": len(test),
    }

    if observed_counts != expected_counts:
        raise RuntimeError(
            "Processed partitions no longer "
            "match the frozen preprocessing "
            "report."
        )

    protocol = _protocol(
        detection
    )

    incidents = config[
        "dataset"
    ][
        "reported_incidents"
    ]

    all_variants = {}

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

        expected_feature_count = int(
            preprocessing_report[
                "feature_variants"
            ][
                variant_name
            ][
                "model_feature_count"
            ]
        )

        if len(features) != (
            expected_feature_count
        ):
            raise RuntimeError(
                "Feature variant drift: "
                f"{variant_name}"
            )

        representation_results = {}

        for name, specification in (
            ablation[
                "representations"
            ].items()
        ):
            print(
                f"Running {variant_name} / "
                f"{name}..."
            )

            family = specification[
                "family"
            ]

            if (
                family
                == "pca_reconstruction"
            ):
                result = (
                    run_pca_representation(
                        train=train,
                        calibration=(
                            calibration
                        ),
                        test=test,
                        features=features,
                        incidents=incidents,
                        protocol=protocol,
                        scaler_name=str(
                            specification[
                                "scaler"
                            ]
                        ),
                        variance_retained=float(
                            specification[
                                "variance_retained"
                            ]
                        ),
                    )
                )
            elif (
                family
                == "feature_energy"
            ):
                result = (
                    run_feature_energy_representation(
                        train=train,
                        calibration=(
                            calibration
                        ),
                        test=test,
                        features=features,
                        incidents=incidents,
                        protocol=protocol,
                        scaler_name=str(
                            specification[
                                "scaler"
                            ]
                        ),
                    )
                )
            else:
                raise ValueError(
                    "Unknown representation "
                    f"family: {family}"
                )

            representation_results[
                name
            ] = result

        reference_name = str(
            ablation[
                "matched_timestamp_reference"
            ]
        )

        matched = (
            matched_timestamp_table(
                representation_results=(
                    representation_results
                ),
                reference_name=(
                    reference_name
                ),
                incidents=incidents,
            )
        )

        all_variants[
            variant_name
        ] = {
            "model_feature_count": (
                len(features)
            ),
            "representations": {
                name: strip_internal_frames(
                    result
                )
                for name, result
                in (
                    representation_results
                    .items()
                )
            },
            "matched_timestamp_concentration": (
                matched
            ),
        }

    report = {
        "schema_version": (
            "aeroxai.metropt2_representation_ablation.v1"
        ),
        "dataset": "MetroPT2",
        "evidence_class": "REAL",
        "causal_claim": False,
        "research_question": (
            "Is explanation concentration driven "
            "primarily by the physical data or "
            "by the anomaly representation?"
        ),
        "pre_registration": {
            "feature_variants": list(
                preprocessing[
                    "feature_variants"
                ].keys()
            ),
            "representations": (
                ablation[
                    "representations"
                ]
            ),
            "matched_timestamp_reference": (
                ablation[
                    "matched_timestamp_reference"
                ]
            ),
            "test_retuning": False,
        },
        "protocol": {
            "train_rows": len(train),
            "calibration_rows": len(
                calibration
            ),
            "test_rows": len(test),
            "threshold_quantile": (
                protocol.threshold_quantile
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
        },
        "frozen_robust_pca_check": {
            variant_name: {
                "expected_pr_auc": (
                    pca_benchmark[
                        "variants"
                    ][
                        variant_name
                    ][
                        "metrics"
                    ][
                        "pr_auc"
                    ]
                ),
                "observed_pr_auc": (
                    all_variants[
                        variant_name
                    ][
                        "representations"
                    ][
                        "robust_pca"
                    ][
                        "metrics"
                    ][
                        "pr_auc"
                    ]
                ),
            }
            for variant_name
            in all_variants
        },
        "variants": all_variants,
        "interpretation_guardrails": [
            (
                "Own-alert explanation "
                "concentrations can be affected "
                "by different alert timestamps."
            ),
            (
                "Matched-timestamp concentration "
                "holds time fixed at the robust "
                "PCA first timely alert, isolating "
                "representation effects more "
                "cleanly."
            ),
            (
                "Feature-energy contribution "
                "means standardized deviation "
                "energy, not causal importance."
            ),
            (
                "PCA reconstruction contribution "
                "means detector residual evidence, "
                "not physical root cause."
            ),
            (
                "No representation is promoted "
                "from this reused test benchmark."
            ),
        ],
    }

    for variant_name, check in (
        report[
            "frozen_robust_pca_check"
        ].items()
    ):
        difference = abs(
            float(
                check[
                    "expected_pr_auc"
                ]
            )
            - float(
                check[
                    "observed_pr_auc"
                ]
            )
        )

        check[
            "absolute_difference"
        ] = difference

        check[
            "reproduced"
        ] = difference <= 1.0e-12

        if not check[
            "reproduced"
        ]:
            raise RuntimeError(
                "Robust PCA benchmark no longer "
                "reproduces for "
                f"{variant_name}."
            )

    output_path = (
        ROOT
        / ablation[
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
        "frozen_robust_pca_check": (
            report[
                "frozen_robust_pca_check"
            ]
        ),
        "variants": {},
        "output": str(
            output_path
        ),
    }

    for variant_name, variant in (
        all_variants.items()
    ):
        compact[
            "variants"
        ][
            variant_name
        ] = {
            "representations": {},
            "matched_timestamp_concentration": [],
        }

        for name, result in (
            variant[
                "representations"
            ].items()
        ):
            compact[
                "variants"
            ][
                variant_name
            ][
                "representations"
            ][name] = {
                "model": result[
                    "model"
                ],
                "pr_auc": (
                    result[
                        "metrics"
                    ][
                        "pr_auc"
                    ]
                ),
                "false_alerts_per_24h": (
                    result[
                        "metrics"
                    ][
                        "false_alerts_per_24h"
                    ]
                ),
                "pre_onset_incident_recall": (
                    result[
                        "metrics"
                    ][
                        "pre_onset_incident_recall"
                    ]
                ),
                "mean_effective_groups": (
                    result[
                        "explanation_summary"
                    ][
                        "mean_effective_group_count"
                    ]
                ),
                "mean_top3": (
                    result[
                        "explanation_summary"
                    ][
                        "mean_top3_concentration"
                    ]
                ),
                "incidents": [
                    {
                        "id": row["id"],
                        "condition": row[
                            "condition"
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
                    for row in (
                        result[
                            "incident_explanations"
                        ]
                    )
                ],
            }

        for row in variant[
            "matched_timestamp_concentration"
        ]:
            compact[
                "variants"
            ][
                variant_name
            ][
                "matched_timestamp_concentration"
            ].append(
                {
                    "id": row[
                        "id"
                    ],
                    "condition": row[
                        "condition"
                    ],
                    "reference_timestamp": (
                        row[
                            "reference_timestamp"
                        ]
                    ),
                    "representations": {
                        name: {
                            "dominant_group": (
                                concentration[
                                    "dominant_group"
                                ]
                            ),
                            "top1": (
                                concentration[
                                    "top1_concentration"
                                ]
                            ),
                            "top3": (
                                concentration[
                                    "top3_concentration"
                                ]
                            ),
                            "effective_groups": (
                                concentration[
                                    "effective_group_count"
                                ]
                            ),
                        }
                        for name, concentration
                        in row[
                            "representations"
                        ].items()
                    },
                }
            )

    print(
        json.dumps(
            compact,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
