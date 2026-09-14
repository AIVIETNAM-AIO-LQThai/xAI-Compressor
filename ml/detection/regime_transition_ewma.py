from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.detection.alerts import (
    extract_alert_episodes,
    persistent_alerts,
)
from ml.detection.evaluate import evaluate_detection
from ml.detection.pca_research_benchmark import (
    AlertPipelineResult,
    AlertProtocol,
)
from ml.detection.regime_pca import (
    RegimePCADetector,
    explain_regime_pca_detector,
    fit_regime_pca_detector,
)
from ml.explainability.concentration import explanation_concentration
from ml.explainability.pca import aggregate_contributions
from ml.explainability.verification import (
    contribution_percentiles,
    fit_contribution_reference,
)


def transition_reset_ewma(
    scores: pd.Series,
    regimes: pd.Series,
    *,
    alpha: float,
    reset_gap_minutes: int,
) -> pd.Series:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1].")

    scores, regimes = scores.align(regimes, join="inner")
    scores = scores.sort_index()
    regimes = regimes.reindex(scores.index)

    if not isinstance(scores.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex.")

    reset_gap = pd.Timedelta(minutes=reset_gap_minutes)
    values = scores.to_numpy(dtype=float)
    labels = regimes.astype(str).to_numpy()

    output = np.empty(len(scores), dtype=float)
    previous_time: pd.Timestamp | None = None
    previous_value: float | None = None
    previous_regime: str | None = None

    for i, (timestamp, value, regime) in enumerate(
        zip(scores.index, values, labels, strict=True)
    ):
        reset = (
            previous_time is None
            or timestamp - previous_time > reset_gap
            or previous_regime is None
            or regime != previous_regime
        )

        if reset or previous_value is None:
            current = float(value)
        else:
            current = float(
                alpha * value
                + (1.0 - alpha) * previous_value
            )

        output[i] = current
        previous_time = timestamp
        previous_value = current
        previous_regime = regime

    return pd.Series(
        output,
        index=scores.index,
        name="smoothed_score",
    )


def transition_reset_ewma_frame(
    contributions: pd.DataFrame,
    regimes: pd.Series,
    *,
    alpha: float,
    reset_gap_minutes: int,
) -> pd.DataFrame:
    contributions = contributions.sort_index()
    regimes = regimes.reindex(contributions.index)

    if not isinstance(contributions.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex.")

    reset_gap = pd.Timedelta(minutes=reset_gap_minutes)
    x = contributions.to_numpy(dtype=float)
    labels = regimes.astype(str).to_numpy()
    output = np.empty_like(x, dtype=float)

    previous_time: pd.Timestamp | None = None
    previous_value: np.ndarray | None = None
    previous_regime: str | None = None

    for i, (timestamp, row, regime) in enumerate(
        zip(contributions.index, x, labels, strict=True)
    ):
        reset = (
            previous_time is None
            or timestamp - previous_time > reset_gap
            or previous_regime is None
            or regime != previous_regime
        )

        if reset or previous_value is None:
            current = row.copy()
        else:
            current = (
                alpha * row
                + (1.0 - alpha) * previous_value
            )

        output[i] = current
        previous_time = timestamp
        previous_value = current
        previous_regime = regime

    return pd.DataFrame(
        output,
        index=contributions.index,
        columns=contributions.columns,
    )


def calibrate_transition_reset_pipeline(
    *,
    calibration_scores: pd.Series,
    calibration_regimes: pd.Series,
    test_scores: pd.Series,
    test_regimes: pd.Series,
    protocol: AlertProtocol,
) -> AlertPipelineResult:
    calibration_smoothed = transition_reset_ewma(
        calibration_scores,
        calibration_regimes,
        alpha=protocol.ewma_alpha,
        reset_gap_minutes=protocol.reset_gap_minutes,
    )
    threshold = float(
        calibration_smoothed.quantile(
            protocol.threshold_quantile,
            interpolation="higher",
        )
    )

    test_smoothed = transition_reset_ewma(
        test_scores,
        test_regimes,
        alpha=protocol.ewma_alpha,
        reset_gap_minutes=protocol.reset_gap_minutes,
    )

    _, alerts = persistent_alerts(
        test_smoothed,
        threshold=threshold,
        required_hits=protocol.persistence_hits,
        window_bins=protocol.persistence_window,
        reset_gap_minutes=protocol.reset_gap_minutes,
    )

    episodes = extract_alert_episodes(
        alerts,
        merge_minutes=protocol.merge_minutes,
        reset_gap_minutes=protocol.reset_gap_minutes,
    )

    return AlertPipelineResult(
        calibration_smoothed=calibration_smoothed,
        test_smoothed=test_smoothed,
        threshold=threshold,
        alerts=alerts,
        episodes=tuple(episodes),
    )


def _candidate_explanations(
    *,
    detector: RegimePCADetector,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    incident_results: list[dict[str, Any]],
    incidents: list[dict[str, Any]],
    protocol: AlertProtocol,
) -> list[dict[str, Any]]:
    calibration_xai = explain_regime_pca_detector(
        calibration,
        detector,
    )
    test_xai = explain_regime_pca_detector(
        test,
        detector,
    )

    calibration_grouped = aggregate_contributions(
        calibration_xai.contributions
    )
    test_grouped = aggregate_contributions(
        test_xai.contributions
    )

    calibration_smoothed = transition_reset_ewma_frame(
        calibration_grouped,
        calibration_xai.regime,
        alpha=protocol.ewma_alpha,
        reset_gap_minutes=protocol.reset_gap_minutes,
    )
    test_smoothed = transition_reset_ewma_frame(
        test_grouped,
        test_xai.regime,
        alpha=protocol.ewma_alpha,
        reset_gap_minutes=protocol.reset_gap_minutes,
    )

    score_check = transition_reset_ewma(
        test_xai.score,
        test_xai.regime,
        alpha=protocol.ewma_alpha,
        reset_gap_minutes=protocol.reset_gap_minutes,
    )
    if not np.allclose(
        test_smoothed.sum(axis=1).to_numpy(dtype=float),
        score_check.to_numpy(dtype=float),
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Transition-reset XAI contributions do not add "
            "to the transition-reset anomaly score."
        )

    reference = fit_contribution_reference(
        calibration_smoothed
    )
    percentiles = contribution_percentiles(
        test_smoothed,
        reference,
    )

    incident_lookup = {
        int(item["id"]): item for item in incidents
    }

    explanations: list[dict[str, Any]] = []

    for result in incident_results:
        incident_id = int(result["id"])
        incident = incident_lookup[incident_id]
        first_alert = result["first_alert"]

        if first_alert is None:
            explanations.append(
                {
                    "id": incident_id,
                    "condition": incident["condition"],
                    "explanation_available": False,
                    "reason": "No timely alert within the frozen evaluation window.",
                }
            )
            continue

        timestamp = pd.Timestamp(first_alert)
        row = test_smoothed.loc[timestamp].astype(float)
        total = float(row.sum())

        if total <= 0.0:
            raise RuntimeError(
                "Timely transition-reset alert has no "
                "positive contribution mass."
            )

        shares = {
            str(group): float(value) / total
            for group, value in row.items()
        }
        concentration = explanation_concentration(shares)
        percentile_row = percentiles.loc[timestamp]
        ranked_groups = sorted(
            row.index,
            key=lambda group: shares[str(group)],
            reverse=True,
        )

        start = pd.Timestamp(incident["start"])
        if timestamp < start:
            timing = "pre_onset"
        elif timestamp > start:
            timing = "post_onset"
        else:
            timing = "at_onset"

        explanations.append(
            {
                "id": incident_id,
                "condition": incident["condition"],
                "explanation_available": True,
                "explanation_timestamp": str(timestamp),
                "operating_regime": str(
                    test_xai.regime.loc[timestamp]
                ),
                "timing": timing,
                "lead_hours": result["lead_hours"],
                "delay_hours": result["delay_hours"],
                "concentration": {
                    "dominant_group": concentration.dominant_group,
                    "dominant_share": concentration.dominant_share,
                    "top1_concentration": concentration.top1_concentration,
                    "top3_concentration": concentration.top3_concentration,
                    "top5_concentration": concentration.top5_concentration,
                    "shannon_entropy_nats": concentration.shannon_entropy_nats,
                    "normalized_entropy": concentration.normalized_entropy,
                    "effective_group_count": concentration.effective_group_count,
                    "herfindahl_index": concentration.herfindahl_index,
                    "group_count": concentration.group_count,
                    "nonzero_group_count": concentration.nonzero_group_count,
                },
                "ranked_groups": [
                    {
                        "group": str(group),
                        "smoothed_contribution": float(row[group]),
                        "share": shares[str(group)],
                        "calibration_percentile": float(
                            percentile_row[group]
                        ),
                    }
                    for group in ranked_groups
                ],
            }
        )

    return explanations


def benchmark_transition_reset_variant(
    *,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    incidents: list[dict[str, Any]],
    protocol: AlertProtocol,
    variance_retained: float,
    current_feature: str,
    pressure_feature: str,
    high_quantile: float,
) -> tuple[
    dict[str, Any],
    pd.Series,
    pd.Series,
]:
    detector = fit_regime_pca_detector(
        train,
        features,
        current_feature=current_feature,
        pressure_feature=pressure_feature,
        high_quantile=high_quantile,
        variance_retained=variance_retained,
    )

    calibration_xai = explain_regime_pca_detector(
        calibration,
        detector,
    )
    test_xai = explain_regime_pca_detector(
        test,
        detector,
    )

    calibration_scores = calibration_xai.score
    test_scores = test_xai.score

    pipeline = calibrate_transition_reset_pipeline(
        calibration_scores=calibration_scores,
        calibration_regimes=calibration_xai.regime,
        test_scores=test_scores,
        test_regimes=test_xai.regime,
        protocol=protocol,
    )

    metrics = evaluate_detection(
        scores=pipeline.test_smoothed,
        alerts=pipeline.alerts,
        episodes=list(pipeline.episodes),
        incidents=incidents,
        early_warning_hours=protocol.early_warning_hours,
        late_tolerance_hours=protocol.late_tolerance_hours,
        bin_minutes=protocol.bin_minutes,
    )

    explanations = _candidate_explanations(
        detector=detector,
        calibration=calibration,
        test=test,
        incident_results=metrics["incident_results"],
        incidents=incidents,
        protocol=protocol,
    )

    available = [
        item
        for item in explanations
        if item["explanation_available"]
    ]

    result = {
        "model": {
            "name": "regime_conditioned_pca_transition_reset_ewma",
            "scaler": detector.scaler_name,
            "input_feature_count": len(detector.features),
            "fixed_component_count": detector.component_count,
            "global_standard_explained_variance_ratio": (
                detector.global_explained_variance_ratio
            ),
            "router": {
                "current_feature": detector.current_feature,
                "pressure_feature": detector.pressure_feature,
                "high_quantile": detector.high_quantile,
                "current_threshold": detector.current_threshold,
                "pressure_threshold": detector.pressure_threshold,
            },
            "train_regime_counts": detector.train_regime_counts,
            "threshold": pipeline.threshold,
            "ewma_reset_rule": "time_gap_or_regime_change",
            "persistence_reset_on_regime_change": False,
        },
        "metrics": metrics,
        "incident_explanations": explanations,
        "explanation_summary": {
            "timely_explanations": len(available),
            "mean_effective_group_count": (
                float(
                    np.mean(
                        [
                            item["concentration"][
                                "effective_group_count"
                            ]
                            for item in available
                        ]
                    )
                )
                if available
                else None
            ),
            "mean_top3_concentration": (
                float(
                    np.mean(
                        [
                            item["concentration"][
                                "top3_concentration"
                            ]
                            for item in available
                        ]
                    )
                )
                if available
                else None
            ),
            "mean_normalized_entropy": (
                float(
                    np.mean(
                        [
                            item["concentration"][
                                "normalized_entropy"
                            ]
                            for item in available
                        ]
                    )
                )
                if available
                else None
            ),
        },
    }

    return (
        result,
        pipeline.calibration_smoothed,
        calibration_xai.regime,
    )
