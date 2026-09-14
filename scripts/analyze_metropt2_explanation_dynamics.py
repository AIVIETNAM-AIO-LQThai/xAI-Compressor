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
    run_feature_energy_representation,
    run_pca_representation,
)
from ml.explainability.temporal_dynamics import (
    concentration_trajectory,
    incident_window,
    summarize_incident_dynamics,
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

    dynamics = config[
        "temporal_explanation_dynamics"
    ]

    detection = _load_yaml(
        ROOT
        / benchmark[
            "detection_config"
        ]
    )

    frozen_ablation = _load_json(
        ROOT
        / ablation[
            "output_file"
        ]
    )

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

    protocol = _protocol(
        detection
    )

    incidents = config[
        "dataset"
    ][
        "reported_incidents"
    ]

    summary_variants = {}
    trajectory_frames = []

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

        summary_variants[
            variant_name
        ] = {
            "model_feature_count": len(
                features
            ),
            "representations": {},
        }

        for representation_name, specification in (
            ablation[
                "representations"
            ].items()
        ):
            print(
                f"Analyzing {variant_name} / "
                f"{representation_name}..."
            )

            family = specification[
                "family"
            ]

            if family == (
                "pca_reconstruction"
            ):
                result = (
                    run_pca_representation(
                        train=train,
                        calibration=calibration,
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
            elif family == (
                "feature_energy"
            ):
                result = (
                    run_feature_energy_representation(
                        train=train,
                        calibration=calibration,
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
                    "Unknown representation family: "
                    f"{family}"
                )

            frozen_pr_auc = float(
                frozen_ablation[
                    "variants"
                ][
                    variant_name
                ][
                    "representations"
                ][
                    representation_name
                ][
                    "metrics"
                ][
                    "pr_auc"
                ]
            )

            observed_pr_auc = float(
                result[
                    "metrics"
                ][
                    "pr_auc"
                ]
            )

            if abs(
                frozen_pr_auc
                - observed_pr_auc
            ) > 1.0e-12:
                raise RuntimeError(
                    "Representation benchmark "
                    "no longer reproduces: "
                    f"{variant_name} / "
                    f"{representation_name}"
                )

            trajectory = (
                concentration_trajectory(
                    result[
                        "_smoothed_contributions"
                    ],
                    threshold=float(
                        result[
                            "threshold"
                        ]
                    ),
                    protocol=protocol,
                )
            )

            incident_summaries = {}

            for incident in incidents:
                window = incident_window(
                    trajectory,
                    onset=incident[
                        "start"
                    ],
                    pre_onset_hours=float(
                        dynamics[
                            "pre_onset_hours"
                        ]
                    ),
                    post_onset_hours=float(
                        dynamics[
                            "post_onset_hours"
                        ]
                    ),
                )

                incident_id = str(
                    int(
                        incident[
                            "id"
                        ]
                    )
                )

                incident_summaries[
                    incident_id
                ] = {
                    "condition": (
                        incident[
                            "condition"
                        ]
                    ),
                    "onset": str(
                        pd.Timestamp(
                            incident[
                                "start"
                            ]
                        )
                    ),
                    **(
                        summarize_incident_dynamics(
                            window,
                            phases=dynamics[
                                "phases"
                            ],
                        )
                    ),
                }

                export = (
                    window.reset_index()
                )

                export[
                    "variant"
                ] = variant_name

                export[
                    "representation"
                ] = representation_name

                export[
                    "incident_id"
                ] = int(
                    incident[
                        "id"
                    ]
                )

                export[
                    "condition"
                ] = incident[
                    "condition"
                ]

                trajectory_frames.append(
                    export
                )

            summary_variants[
                variant_name
            ][
                "representations"
            ][
                representation_name
            ] = {
                "family": family,
                "scaler": (
                    specification[
                        "scaler"
                    ]
                ),
                "threshold": float(
                    result[
                        "threshold"
                    ]
                ),
                "pr_auc_reproduced": (
                    observed_pr_auc
                ),
                "incident_summaries": (
                    incident_summaries
                ),
            }

    trajectory_table = pd.concat(
        trajectory_frames,
        ignore_index=True,
    )

    trajectory_path = (
        ROOT
        / dynamics[
            "trajectory_file"
        ]
    )

    trajectory_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trajectory_table.to_parquet(
        trajectory_path,
        index=False,
    )

    report = {
        "schema_version": (
            "aeroxai.metropt2_temporal_explanation_dynamics.v1"
        ),
        "dataset": "MetroPT2",
        "evidence_class": "REAL",
        "causal_claim": False,
        "research_question": (
            "How does detector-evidence "
            "dimensionality evolve before "
            "and after reported fault onset?"
        ),
        "analysis_window": {
            "pre_onset_hours": float(
                dynamics[
                    "pre_onset_hours"
                ]
            ),
            "post_onset_hours": float(
                dynamics[
                    "post_onset_hours"
                ]
            ),
            "phases": dynamics[
                "phases"
            ],
        },
        "protocol": {
            "contribution_smoothing": (
                "Same causal EWMA and "
                "gap-reset protocol as detector."
            ),
            "alert_definition": (
                "Same calibration threshold "
                "and persistence logic as "
                "frozen representation ablation."
            ),
            "test_retuning": False,
        },
        "variants": (
            summary_variants
        ),
        "trajectory_file": str(
            trajectory_path
        ),
        "interpretation_guardrails": [
            (
                "Effective group count measures "
                "the distribution of detector "
                "evidence, not the number of "
                "physical root causes."
            ),
            (
                "Temporal changes in dominant "
                "groups are detector-evidence "
                "changes, not proven causal "
                "fault propagation."
            ),
            (
                "This is exploratory analysis "
                "on a reused test benchmark."
            ),
            (
                "No detector or representation "
                "is promoted from this analysis."
            ),
        ],
    }

    output_path = (
        ROOT
        / dynamics[
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
        "analysis_window": report[
            "analysis_window"
        ],
        "variants": {},
        "trajectory_file": str(
            trajectory_path
        ),
        "output": str(
            output_path
        ),
    }

    for variant_name, variant in (
        summary_variants.items()
    ):
        compact[
            "variants"
        ][
            variant_name
        ] = {}

        for representation_name, representation in (
            variant[
                "representations"
            ].items()
        ):
            compact[
                "variants"
            ][
                variant_name
            ][
                representation_name
            ] = (
                representation[
                    "incident_summaries"
                ]
            )

    print(
        json.dumps(
            compact,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
