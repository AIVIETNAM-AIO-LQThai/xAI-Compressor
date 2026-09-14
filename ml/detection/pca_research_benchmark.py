from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from ml.detection.alerts import (
    AlertEpisode,
    causal_ewma,
    extract_alert_episodes,
    persistent_alerts,
)
from ml.detection.evaluate import (
    evaluate_detection,
)
from ml.detection.pca_detector import (
    PCADetector,
    fit_pca_detector,
    score_pca_detector,
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


@dataclass(frozen=True)
class AlertProtocol:
    ewma_alpha: float
    threshold_quantile: float
    persistence_hits: int
    persistence_window: int
    reset_gap_minutes: int
    merge_minutes: int
    early_warning_hours: int
    late_tolerance_hours: int
    bin_minutes: int


@dataclass(frozen=True)
class AlertPipelineResult:
    calibration_smoothed: pd.Series
    test_smoothed: pd.Series
    threshold: float
    alerts: pd.Series
    episodes: tuple[AlertEpisode, ...]


def calibrate_alert_pipeline(
    *,
    calibration_scores: pd.Series,
    test_scores: pd.Series,
    protocol: AlertProtocol,
) -> AlertPipelineResult:
    calibration_smoothed = causal_ewma(
        calibration_scores,
        alpha=protocol.ewma_alpha,
        reset_gap_minutes=(
            protocol.reset_gap_minutes
        ),
    )

    threshold = float(
        calibration_smoothed.quantile(
            protocol.threshold_quantile,
            interpolation="higher",
        )
    )

    test_smoothed = causal_ewma(
        test_scores,
        alpha=protocol.ewma_alpha,
        reset_gap_minutes=(
            protocol.reset_gap_minutes
        ),
    )

    _, alerts = persistent_alerts(
        test_smoothed,
        threshold=threshold,
        required_hits=(
            protocol.persistence_hits
        ),
        window_bins=(
            protocol.persistence_window
        ),
        reset_gap_minutes=(
            protocol.reset_gap_minutes
        ),
    )

    episodes = extract_alert_episodes(
        alerts,
        merge_minutes=(
            protocol.merge_minutes
        ),
        reset_gap_minutes=(
            protocol.reset_gap_minutes
        ),
    )

    return AlertPipelineResult(
        calibration_smoothed=(
            calibration_smoothed
        ),
        test_smoothed=test_smoothed,
        threshold=threshold,
        alerts=alerts,
        episodes=tuple(episodes),
    )


def _explain_timely_incidents(
    *,
    detector: PCADetector,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    incident_results: list[dict[str, Any]],
    incidents: list[dict[str, Any]],
    protocol: AlertProtocol,
) -> list[dict[str, Any]]:
    calibration_xai = explain_pca_detector(
        calibration,
        detector,
    )

    test_xai = explain_pca_detector(
        test,
        detector,
    )

    calibration_grouped = (
        aggregate_contributions(
            calibration_xai.contributions
        )
    )

    test_grouped = (
        aggregate_contributions(
            test_xai.contributions
        )
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

    incident_lookup = {
        int(item["id"]): item
        for item in incidents
    }

    explanations: list[
        dict[str, Any]
    ] = []

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
            explanations.append(
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

        if timestamp not in (
            test_smoothed.index
        ):
            raise RuntimeError(
                "Timely alert timestamp is "
                "missing from XAI frame: "
                f"{timestamp}"
            )

        row = test_smoothed.loc[
            timestamp
        ].astype(float)

        total = float(
            row.sum()
        )

        if total <= 0.0:
            raise RuntimeError(
                "Timely alert has no positive "
                "PCA contribution mass."
            )

        shares = {
            str(group): (
                float(value)
                / total
            )
            for group, value
            in row.items()
        }

        concentration = (
            explanation_concentration(
                shares
            )
        )

        percentile_row = (
            percentiles.loc[
                timestamp
            ]
        )

        ranked_groups = sorted(
            row.index,
            key=lambda group: (
                shares[str(group)]
            ),
            reverse=True,
        )

        start = pd.Timestamp(
            incident["start"]
        )

        if timestamp < start:
            timing = "pre_onset"
        elif timestamp > start:
            timing = "post_onset"
        else:
            timing = "at_onset"

        explanations.append(
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
                "lead_hours": (
                    result[
                        "lead_hours"
                    ]
                ),
                "delay_hours": (
                    result[
                        "delay_hours"
                    ]
                ),
                "concentration": (
                    asdict(
                        concentration
                    )
                ),
                "ranked_groups": [
                    {
                        "group": (
                            str(group)
                        ),
                        "smoothed_contribution": (
                            float(
                                row[
                                    group
                                ]
                            )
                        ),
                        "share": (
                            shares[
                                str(
                                    group
                                )
                            ]
                        ),
                        "calibration_percentile": (
                            float(
                                percentile_row[
                                    group
                                ]
                            )
                        ),
                    }
                    for group in (
                        ranked_groups
                    )
                ],
            }
        )

    return explanations


def benchmark_pca_variant(
    *,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    incidents: list[dict[str, Any]],
    protocol: AlertProtocol,
    variance_retained: float,
    scaler_name: str,
) -> dict[str, Any]:
    detector = fit_pca_detector(
        train,
        features,
        variance_retained=(
            variance_retained
        ),
        scaler_name=scaler_name,
    )

    calibration_scores = (
        score_pca_detector(
            calibration,
            detector,
        )
    )

    test_scores = (
        score_pca_detector(
            test,
            detector,
        )
    )

    pipeline = calibrate_alert_pipeline(
        calibration_scores=(
            calibration_scores
        ),
        test_scores=test_scores,
        protocol=protocol,
    )

    metrics = evaluate_detection(
        scores=(
            pipeline.test_smoothed
        ),
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
        bin_minutes=(
            protocol.bin_minutes
        ),
    )

    explanations = (
        _explain_timely_incidents(
            detector=detector,
            calibration=calibration,
            test=test,
            incident_results=(
                metrics[
                    "incident_results"
                ]
            ),
            incidents=incidents,
            protocol=protocol,
        )
    )

    explanation_rows = [
        row
        for row in explanations
        if row[
            "explanation_available"
        ]
    ]

    mean_effective_groups = None
    mean_top3 = None

    if explanation_rows:
        mean_effective_groups = (
            sum(
                row[
                    "concentration"
                ][
                    "effective_group_count"
                ]
                for row
                in explanation_rows
            )
            / len(
                explanation_rows
            )
        )

        mean_top3 = (
            sum(
                row[
                    "concentration"
                ][
                    "top3_concentration"
                ]
                for row
                in explanation_rows
            )
            / len(
                explanation_rows
            )
        )

    return {
        "model": {
            "name": (
                "pca_reconstruction"
            ),
            "scaler": scaler_name,
            "input_feature_count": (
                len(
                    features
                )
            ),
            "variance_retained": (
                variance_retained
            ),
            "components": int(
                detector.model.n_components_
            ),
            "explained_variance_ratio": (
                float(
                    detector.model
                    .explained_variance_ratio_
                    .sum()
                )
            ),
            "threshold": (
                pipeline.threshold
            ),
        },
        "metrics": metrics,
        "incident_explanations": (
            explanations
        ),
        "explanation_summary": {
            "timely_explanations": (
                len(
                    explanation_rows
                )
            ),
            "mean_effective_group_count": (
                mean_effective_groups
            ),
            "mean_top3_concentration": (
                mean_top3
            ),
        },
    }
