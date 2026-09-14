from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from ml.detection.evaluate import (
    evaluate_detection,
)
from ml.detection.feature_energy import (
    explain_feature_energy_detector,
    fit_feature_energy_detector,
)
from ml.detection.pca_detector import (
    fit_pca_detector,
)
from ml.detection.pca_research_benchmark import (
    AlertProtocol,
    calibrate_alert_pipeline,
)
from ml.explainability.concentration import (
    explanation_concentration,
)
from ml.explainability.pca import (
    aggregate_contributions,
    causal_ewma_contributions,
    explain_pca_detector,
)
from ml.explainability.verification import (
    contribution_percentiles,
    fit_contribution_reference,
)


def _evaluate_representation(
    *,
    calibration_score: pd.Series,
    test_score: pd.Series,
    calibration_grouped: pd.DataFrame,
    test_grouped: pd.DataFrame,
    incidents: list[dict[str, Any]],
    protocol: AlertProtocol,
) -> dict[str, Any]:
    pipeline = calibrate_alert_pipeline(
        calibration_scores=(
            calibration_score
        ),
        test_scores=test_score,
        protocol=protocol,
    )

    metrics = evaluate_detection(
        scores=pipeline.test_smoothed,
        alerts=pipeline.alerts,
        episodes=list(
            pipeline.episodes
        ),
        incidents=incidents,
        early_warning_hours=(
            protocol.early_warning_hours
        ),
        late_tolerance_hours=(
            protocol.late_tolerance_hours
        ),
        bin_minutes=protocol.bin_minutes,
    )

    calibration_smoothed = (
        causal_ewma_contributions(
            calibration_grouped,
            alpha=protocol.ewma_alpha,
            reset_gap_minutes=(
                protocol.reset_gap_minutes
            ),
        )
    )

    test_smoothed = (
        causal_ewma_contributions(
            test_grouped,
            alpha=protocol.ewma_alpha,
            reset_gap_minutes=(
                protocol.reset_gap_minutes
            ),
        )
    )

    reference = fit_contribution_reference(
        calibration_smoothed
    )

    percentiles = contribution_percentiles(
        test_smoothed,
        reference,
    )

    explanations = (
        explain_incident_alerts(
            incident_results=(
                metrics[
                    "incident_results"
                ]
            ),
            incidents=incidents,
            smoothed_contributions=(
                test_smoothed
            ),
            percentiles=percentiles,
        )
    )

    available = [
        row
        for row in explanations
        if row[
            "explanation_available"
        ]
    ]

    return {
        "threshold": pipeline.threshold,
        "metrics": metrics,
        "incident_explanations": (
            explanations
        ),
        "explanation_summary": {
            "timely_explanations": len(
                available
            ),
            "mean_effective_group_count": (
                None
                if not available
                else sum(
                    row[
                        "concentration"
                    ][
                        "effective_group_count"
                    ]
                    for row in available
                )
                / len(available)
            ),
            "mean_top3_concentration": (
                None
                if not available
                else sum(
                    row[
                        "concentration"
                    ][
                        "top3_concentration"
                    ]
                    for row in available
                )
                / len(available)
            ),
        },
        "_smoothed_contributions": (
            test_smoothed
        ),
    }


def explain_incident_alerts(
    *,
    incident_results: list[
        dict[str, Any]
    ],
    incidents: list[dict[str, Any]],
    smoothed_contributions: pd.DataFrame,
    percentiles: pd.DataFrame,
) -> list[dict[str, Any]]:
    incident_lookup = {
        int(item["id"]): item
        for item in incidents
    }

    output = []

    for result in incident_results:
        incident_id = int(
            result["id"]
        )

        incident = incident_lookup[
            incident_id
        ]

        first_alert = result[
            "first_alert"
        ]

        if first_alert is None:
            output.append(
                {
                    "id": incident_id,
                    "condition": (
                        incident[
                            "condition"
                        ]
                    ),
                    "explanation_available": (
                        False
                    ),
                    "reason": (
                        "No timely alert within "
                        "the pre-registered "
                        "evaluation window."
                    ),
                }
            )
            continue

        timestamp = pd.Timestamp(
            first_alert
        )

        concentration = (
            concentration_at_timestamp(
                smoothed_contributions,
                timestamp,
            )
        )

        row = (
            smoothed_contributions.loc[
                timestamp
            ].astype(float)
        )

        total = float(
            row.sum()
        )

        percentile_row = (
            percentiles.loc[
                timestamp
            ]
        )

        ranked = sorted(
            row.index,
            key=lambda group: (
                float(
                    row[group]
                )
            ),
            reverse=True,
        )

        start = pd.Timestamp(
            incident["start"]
        )

        timing = (
            "pre_onset"
            if timestamp < start
            else (
                "post_onset"
                if timestamp > start
                else "at_onset"
            )
        )

        output.append(
            {
                "id": incident_id,
                "condition": (
                    incident[
                        "condition"
                    ]
                ),
                "explanation_available": (
                    True
                ),
                "explanation_timestamp": (
                    str(timestamp)
                ),
                "timing": timing,
                "lead_hours": result[
                    "lead_hours"
                ],
                "delay_hours": result[
                    "delay_hours"
                ],
                "concentration": (
                    concentration
                ),
                "ranked_groups": [
                    {
                        "group": str(
                            group
                        ),
                        "smoothed_contribution": (
                            float(
                                row[
                                    group
                                ]
                            )
                        ),
                        "share": (
                            float(
                                row[
                                    group
                                ]
                            )
                            / total
                        ),
                        "calibration_percentile": (
                            float(
                                percentile_row[
                                    group
                                ]
                            )
                        ),
                    }
                    for group in ranked
                ],
            }
        )

    return output


def concentration_at_timestamp(
    smoothed_contributions: pd.DataFrame,
    timestamp: pd.Timestamp | str,
) -> dict[str, Any]:
    point = pd.Timestamp(
        timestamp
    )

    if point not in (
        smoothed_contributions.index
    ):
        raise ValueError(
            "Matched timestamp is missing "
            f"from contribution frame: {point}"
        )

    row = (
        smoothed_contributions.loc[
            point
        ].astype(float)
    )

    total = float(
        row.sum()
    )

    if total <= 0.0:
        raise ValueError(
            "Contribution mass must be "
            "positive at matched timestamp."
        )

    shares = {
        str(group): (
            float(value)
            / total
        )
        for group, value
        in row.items()
    }

    return asdict(
        explanation_concentration(
            shares
        )
    )


def run_pca_representation(
    *,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    incidents: list[dict[str, Any]],
    protocol: AlertProtocol,
    scaler_name: str,
    variance_retained: float,
) -> dict[str, Any]:
    detector = fit_pca_detector(
        train,
        features,
        variance_retained=(
            variance_retained
        ),
        scaler_name=scaler_name,
    )

    calibration_xai = (
        explain_pca_detector(
            calibration,
            detector,
        )
    )

    test_xai = (
        explain_pca_detector(
            test,
            detector,
        )
    )

    result = _evaluate_representation(
        calibration_score=(
            calibration_xai.score
        ),
        test_score=test_xai.score,
        calibration_grouped=(
            aggregate_contributions(
                calibration_xai.contributions
            )
        ),
        test_grouped=(
            aggregate_contributions(
                test_xai.contributions
            )
        ),
        incidents=incidents,
        protocol=protocol,
    )

    result["model"] = {
        "family": "pca_reconstruction",
        "scaler": scaler_name,
        "input_feature_count": len(
            features
        ),
        "components": int(
            detector.model.n_components_
        ),
        "variance_retained": (
            variance_retained
        ),
        "explained_variance_ratio": (
            float(
                detector.model
                .explained_variance_ratio_
                .sum()
            )
        ),
        "threshold": result[
            "threshold"
        ],
    }

    return result


def run_feature_energy_representation(
    *,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    incidents: list[dict[str, Any]],
    protocol: AlertProtocol,
    scaler_name: str,
) -> dict[str, Any]:
    detector = (
        fit_feature_energy_detector(
            train,
            features,
            scaler_name=scaler_name,
        )
    )

    calibration_xai = (
        explain_feature_energy_detector(
            calibration,
            detector,
        )
    )

    test_xai = (
        explain_feature_energy_detector(
            test,
            detector,
        )
    )

    result = _evaluate_representation(
        calibration_score=(
            calibration_xai.score
        ),
        test_score=test_xai.score,
        calibration_grouped=(
            aggregate_contributions(
                calibration_xai.contributions
            )
        ),
        test_grouped=(
            aggregate_contributions(
                test_xai.contributions
            )
        ),
        incidents=incidents,
        protocol=protocol,
    )

    result["model"] = {
        "family": "feature_energy",
        "scaler": scaler_name,
        "input_feature_count": len(
            features
        ),
        "threshold": result[
            "threshold"
        ],
    }

    return result


def matched_timestamp_table(
    *,
    representation_results: dict[
        str,
        dict[str, Any],
    ],
    reference_name: str,
    incidents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if reference_name not in (
        representation_results
    ):
        raise KeyError(
            "Unknown matched timestamp "
            f"reference: {reference_name}"
        )

    reference = (
        representation_results[
            reference_name
        ]
    )

    reference_results = {
        int(item["id"]): item
        for item in reference[
            "metrics"
        ][
            "incident_results"
        ]
    }

    rows = []

    for incident in incidents:
        incident_id = int(
            incident["id"]
        )

        first_alert = (
            reference_results[
                incident_id
            ][
                "first_alert"
            ]
        )

        if first_alert is None:
            rows.append(
                {
                    "id": incident_id,
                    "condition": (
                        incident[
                            "condition"
                        ]
                    ),
                    "reference_timestamp": (
                        None
                    ),
                    "representations": {},
                }
            )
            continue

        timestamp = pd.Timestamp(
            first_alert
        )

        representations = {}

        for name, result in (
            representation_results.items()
        ):
            smoothed = result[
                "_smoothed_contributions"
            ]

            representations[
                name
            ] = (
                concentration_at_timestamp(
                    smoothed,
                    timestamp,
                )
            )

        rows.append(
            {
                "id": incident_id,
                "condition": (
                    incident[
                        "condition"
                    ]
                ),
                "reference_timestamp": (
                    str(timestamp)
                ),
                "reference_representation": (
                    reference_name
                ),
                "representations": (
                    representations
                ),
            }
        )

    return rows


def strip_internal_frames(
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value
        in result.items()
        if not key.startswith("_")
    }
