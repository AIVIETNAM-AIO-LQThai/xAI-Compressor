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
from ml.explainability.evidence_states import (
    build_evidence_state_frame,
    first_state_entry,
    fit_scalar_reference,
    reference_quantiles,
    state_at_onset_boundary,
    state_runs,
    summarize_state_phase,
)
from ml.explainability.temporal_dynamics import (
    concentration_trajectory,
    incident_window,
    phase_mask,
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


def _phase_summaries(
    window: pd.DataFrame,
    phases: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    output = {}

    for index, phase in enumerate(
        phases
    ):
        mask = phase_mask(
            window[
                "hours_from_onset"
            ],
            start_hours=float(
                phase[
                    "start_hours"
                ]
            ),
            end_hours=float(
                phase[
                    "end_hours"
                ]
            ),
            is_last_phase=(
                index
                == len(phases) - 1
            ),
        )

        output[
            str(
                phase[
                    "name"
                ]
            )
        ] = summarize_state_phase(
            window.loc[
                mask
            ]
        )

    return output


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

    state_config = config[
        "evidence_state_analysis"
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

    variants = {}
    trajectory_exports = []

    concentrated_cutoff = float(
        state_config[
            "concentrated_percentile_max"
        ]
    )

    diffuse_cutoff = float(
        state_config[
            "diffuse_percentile_min"
        ]
    )

    for variant_name, variant in (
        preprocessing[
            "feature_variants"
        ].items()
    ):
        features = model_feature_columns(
            train.columns,
            excluded_groups=variant[
                "excluded_groups"
            ],
        )

        variant_report = {
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
                result = run_pca_representation(
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
                    "Unknown representation "
                    f"family: {family}"
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
                    "Frozen representation "
                    "benchmark no longer reproduces: "
                    f"{variant_name} / "
                    f"{representation_name}"
                )

            calibration_trajectory = (
                concentration_trajectory(
                    result[
                        "_calibration_smoothed_contributions"
                    ],
                    threshold=float(
                        result[
                            "threshold"
                        ]
                    ),
                    protocol=protocol,
                )
            )

            test_trajectory = (
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

            score_reference = (
                fit_scalar_reference(
                    result[
                        "_calibration_smoothed_score"
                    ].rename(
                        "smoothed_score"
                    )
                )
            )

            spread_reference = (
                fit_scalar_reference(
                    calibration_trajectory[
                        "effective_group_count"
                    ]
                )
            )

            state_frame = (
                build_evidence_state_frame(
                    trajectory=test_trajectory,
                    score=result[
                        "_test_smoothed_score"
                    ],
                    alerts=result[
                        "_test_alerts"
                    ],
                    threshold=float(
                        result[
                            "threshold"
                        ]
                    ),
                    score_reference=score_reference,
                    spread_reference=spread_reference,
                    concentrated_percentile_max=(
                        concentrated_cutoff
                    ),
                    diffuse_percentile_min=(
                        diffuse_cutoff
                    ),
                )
            )

            incident_reports = {}

            for incident in incidents:
                incident_id = str(
                    int(
                        incident[
                            "id"
                        ]
                    )
                )

                onset = pd.Timestamp(
                    incident[
                        "start"
                    ]
                )

                window = incident_window(
                    state_frame,
                    onset=onset,
                    pre_onset_hours=float(
                        state_config[
                            "pre_onset_hours"
                        ]
                    ),
                    post_onset_hours=float(
                        state_config[
                            "post_onset_hours"
                        ]
                    ),
                )

                phase_summaries = (
                    _phase_summaries(
                        window,
                        state_config[
                            "phases"
                        ],
                    )
                )

                first_alerting = (
                    window.loc[
                        window[
                            "magnitude_state"
                        ]
                        == "alerting"
                    ]
                )

                first_alerting_summary = None

                if not first_alerting.empty:
                    timestamp = (
                        first_alerting.index[
                            0
                        ]
                    )

                    row = (
                        first_alerting.iloc[
                            0
                        ]
                    )

                    offset_hours = (
                        timestamp - onset
                    ).total_seconds() / 3600.0

                    first_alerting_summary = {
                        "timestamp": str(
                            timestamp
                        ),
                        "hours_from_onset": float(
                            offset_hours
                        ),
                        "spread_state": str(
                            row[
                                "spread_state"
                            ]
                        ),
                        "joint_state": str(
                            row[
                                "joint_state"
                            ]
                        ),
                        "spread_percentile": float(
                            row[
                                "spread_calibration_percentile"
                            ]
                        ),
                        "effective_groups": float(
                            row[
                                "effective_group_count"
                            ]
                        ),
                        "dominant_group": (
                            None
                            if pd.isna(
                                row[
                                    "dominant_group"
                                ]
                            )
                            else str(
                                row[
                                    "dominant_group"
                                ]
                            )
                        ),
                    }

                alerting_diffuse = (
                    first_state_entry(
                        window,
                        joint_state=(
                            "alerting|diffuse"
                        ),
                        onset=onset,
                    )
                )

                alerting_concentrated = (
                    first_state_entry(
                        window,
                        joint_state=(
                            "alerting|concentrated"
                        ),
                        onset=onset,
                    )
                )

                diffuse_before_concentrated = None

                if (
                    alerting_diffuse is not None
                    and alerting_concentrated
                    is not None
                ):
                    diffuse_before_concentrated = (
                        pd.Timestamp(
                            alerting_diffuse[
                                "timestamp"
                            ]
                        )
                        < pd.Timestamp(
                            alerting_concentrated[
                                "timestamp"
                            ]
                        )
                    )

                incident_reports[
                    incident_id
                ] = {
                    "condition": incident[
                        "condition"
                    ],
                    "onset": str(
                        onset
                    ),
                    "window_bins": len(
                            window
                        ),
                    "phase_summaries": (
                        phase_summaries
                    ),
                    "onset_boundary": (
                        state_at_onset_boundary(
                            window,
                            onset=onset,
                        )
                    ),
                    "first_alerting_state": (
                        first_alerting_summary
                    ),
                    "first_alerting_diffuse": (
                        alerting_diffuse
                    ),
                    "first_alerting_concentrated": (
                        alerting_concentrated
                    ),
                    "alerting_diffuse_before_concentrated": (
                        diffuse_before_concentrated
                    ),
                    "state_runs": (
                        state_runs(
                            window
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

                trajectory_exports.append(
                    export
                )

            variant_report[
                "representations"
            ][
                representation_name
            ] = {
                "family": family,
                "scaler": specification[
                    "scaler"
                ],
                "threshold": float(
                    result[
                        "threshold"
                    ]
                ),
                "pr_auc_reproduced": (
                    observed_pr_auc
                ),
                "calibration_references": {
                    "score_quantiles": (
                        reference_quantiles(
                            score_reference,
                            [
                                0.50,
                                0.95,
                                0.995,
                            ],
                        )
                    ),
                    "effective_group_count_quantiles": (
                        reference_quantiles(
                            spread_reference,
                            [
                                0.10,
                                0.50,
                                0.90,
                            ],
                        )
                    ),
                },
                "incidents": (
                    incident_reports
                ),
            }

        variants[
            variant_name
        ] = variant_report

    trajectory_table = pd.concat(
        trajectory_exports,
        ignore_index=True,
    )

    trajectory_path = (
        ROOT
        / state_config[
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
            "aeroxai.metropt2_evidence_state_transitions.v1"
        ),
        "dataset": "MetroPT2",
        "evidence_class": "REAL",
        "causal_claim": False,
        "research_question": (
            "Does detector evidence transition "
            "between calibration-relative diffuse, "
            "typical, and concentrated states as "
            "anomaly magnitude crosses the frozen "
            "operational alert threshold?"
        ),
        "state_definition": {
            "spread_axis": {
                "reference": (
                    "Calibration-only empirical "
                    "distribution of effective "
                    "group count for each "
                    "variant/representation."
                ),
                "concentrated": (
                    f"percentile <= "
                    f"{concentrated_cutoff}"
                ),
                "typical": (
                    f"{concentrated_cutoff} < "
                    f"percentile < "
                    f"{diffuse_cutoff}"
                ),
                "diffuse": (
                    f"percentile >= "
                    f"{diffuse_cutoff}"
                ),
            },
            "magnitude_axis": {
                "subthreshold": (
                    "Smoothed score below frozen "
                    "calibration threshold and not "
                    "in persistent alert."
                ),
                "threshold_crossing": (
                    "Smoothed score at/above frozen "
                    "threshold but persistence has "
                    "not yet produced an alert."
                ),
                "alerting": (
                    "Frozen persistence-aware "
                    "detector alert is active."
                ),
            },
        },
        "analysis_window": {
            "pre_onset_hours": float(
                state_config[
                    "pre_onset_hours"
                ]
            ),
            "post_onset_hours": float(
                state_config[
                    "post_onset_hours"
                ]
            ),
            "phases": state_config[
                "phases"
            ],
        },
        "test_retuning": False,
        "variants": variants,
        "trajectory_file": str(
            trajectory_path
        ),
        "interpretation_guardrails": [
            (
                "Evidence states describe detector "
                "behavior, not physical fault states."
            ),
            (
                "Calibration-relative spread avoids "
                "using one absolute N_eff threshold "
                "across incompatible representations."
            ),
            (
                "A diffuse-to-concentrated transition "
                "is not a causal fault-propagation "
                "claim."
            ),
            (
                "This analysis reuses the MetroPT2 "
                "test incidents and remains exploratory."
            ),
            (
                "No detector or representation is "
                "promoted from this analysis."
            ),
        ],
    }

    output_path = (
        ROOT
        / state_config[
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
        "state_definition": report[
            "state_definition"
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
        variants.items()
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
            ] = {
                "calibration_references": (
                    representation[
                        "calibration_references"
                    ]
                ),
                "incidents": {
                    incident_id: {
                        "condition": incident[
                            "condition"
                        ],
                        "first_alerting_state": (
                            incident[
                                "first_alerting_state"
                            ]
                        ),
                        "first_alerting_diffuse": (
                            incident[
                                "first_alerting_diffuse"
                            ]
                        ),
                        "first_alerting_concentrated": (
                            incident[
                                "first_alerting_concentrated"
                            ]
                        ),
                        "alerting_diffuse_before_concentrated": (
                            incident[
                                "alerting_diffuse_before_concentrated"
                            ]
                        ),
                        "onset_boundary": (
                            incident[
                                "onset_boundary"
                            ]
                        ),
                        "phase_summaries": (
                            incident[
                                "phase_summaries"
                            ]
                        ),
                    }
                    for incident_id, incident
                    in representation[
                        "incidents"
                    ].items()
                },
            }

    print(
        json.dumps(
            compact,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
