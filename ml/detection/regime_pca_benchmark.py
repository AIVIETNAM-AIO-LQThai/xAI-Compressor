from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from ml.detection.alerts import (
    causal_ewma,
)
from ml.detection.evaluate import (
    evaluate_detection,
)
from ml.detection.pca_research_benchmark import (
    AlertProtocol,
    calibrate_alert_pipeline,
)
from ml.detection.regime_pca import (
    RegimePCADetector,
    explain_regime_pca_detector,
    fit_regime_pca_detector,
    score_regime_pca_detector,
)
from ml.explainability.concentration import (
    explanation_concentration,
)
from ml.explainability.pca import (
    aggregate_contributions,
    causal_ewma_contributions,
)
from ml.explainability.verification import (
    contribution_percentiles,
    fit_contribution_reference,
)


def _incident_explanations(
    *,
    detector: RegimePCADetector,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    incident_results: list[dict[str, Any]],
    incidents: list[dict[str, Any]],
    protocol: AlertProtocol,
) -> list[dict[str, Any]]:
    calibration_xai = (
        explain_regime_pca_detector(
            calibration,
            detector,
        )
    )

    test_xai = (
        explain_regime_pca_detector(
            test,
            detector,
        )
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
                    "condition": incident[
                        "condition"
                    ],
                    "explanation_available": False,
                    "reason": (
                        "No timely alert within "
                        "the frozen evaluation window."
                    ),
                }
            )
            continue

        timestamp = pd.Timestamp(
            first_alert
        )

        row = (
            test_smoothed.loc[
                timestamp
            ].astype(float)
        )

        total = float(
            row.sum()
        )

        if total <= 0.0:
            raise RuntimeError(
                "Timely regime-PCA alert has no "
                "positive contribution mass."
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

        percentile_row = percentiles.loc[
            timestamp
        ]

        ranked_groups = sorted(
            row.index,
            key=lambda group: shares[
                str(group)
            ],
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
                "condition": incident[
                    "condition"
                ],
                "explanation_available": True,
                "explanation_timestamp": str(
                    timestamp
                ),
                "operating_regime": str(
                    test_xai.regime.loc[
                        timestamp
                    ]
                ),
                "timing": timing,
                "lead_hours": result[
                    "lead_hours"
                ],
                "delay_hours": result[
                    "delay_hours"
                ],
                "concentration": asdict(
                    concentration
                ),
                "ranked_groups": [
                    {
                        "group": str(group),
                        "smoothed_contribution": (
                            float(
                                row[group]
                            )
                        ),
                        "share": shares[
                            str(group)
                        ],
                        "calibration_percentile": (
                            float(
                                percentile_row[
                                    group
                                ]
                            )
                        ),
                    }
                    for group in ranked_groups
                ],
            }
        )

    return explanations


def benchmark_regime_pca_variant(
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
) -> tuple[dict[str, Any], pd.Series]:
    detector = fit_regime_pca_detector(
        train,
        features,
        current_feature=current_feature,
        pressure_feature=pressure_feature,
        high_quantile=high_quantile,
        variance_retained=variance_retained,
    )

    calibration_scores = (
        score_regime_pca_detector(
            calibration,
            detector,
        )
    )

    test_scores = (
        score_regime_pca_detector(
            test,
            detector,
        )
    )

    pipeline = calibrate_alert_pipeline(
        calibration_scores=calibration_scores,
        test_scores=test_scores,
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

    explanations = _incident_explanations(
        detector=detector,
        calibration=calibration,
        test=test,
        incident_results=metrics[
            "incident_results"
        ],
        incidents=incidents,
        protocol=protocol,
    )

    available = [
        item
        for item in explanations
        if item[
            "explanation_available"
        ]
    ]

    mean_effective_groups = None
    mean_top3 = None
    mean_normalized_entropy = None

    if available:
        mean_effective_groups = float(
            np.mean(
                [
                    item[
                        "concentration"
                    ][
                        "effective_group_count"
                    ]
                    for item in available
                ]
            )
        )

        mean_top3 = float(
            np.mean(
                [
                    item[
                        "concentration"
                    ][
                        "top3_concentration"
                    ]
                    for item in available
                ]
            )
        )

        mean_normalized_entropy = float(
            np.mean(
                [
                    item[
                        "concentration"
                    ][
                        "normalized_entropy"
                    ]
                    for item in available
                ]
            )
        )

    calibration_regimes = (
        detector.train_regime_counts.copy()
    )

    model_details = {
        "name": (
            "regime_conditioned_pca_reconstruction"
        ),
        "scaler": detector.scaler_name,
        "input_feature_count": len(
            detector.features
        ),
        "variance_retained_rule": (
            variance_retained
        ),
        "fixed_component_count": (
            detector.component_count
        ),
        "global_standard_explained_variance_ratio": (
            detector.global_explained_variance_ratio
        ),
        "router": {
            "current_feature": (
                detector.current_feature
            ),
            "pressure_feature": (
                detector.pressure_feature
            ),
            "high_quantile": (
                detector.high_quantile
            ),
            "current_threshold": (
                detector.current_threshold
            ),
            "pressure_threshold": (
                detector.pressure_threshold
            ),
        },
        "train_regime_counts": (
            calibration_regimes
        ),
        "regime_explained_variance_ratio": {
            regime: float(
                model
                .explained_variance_ratio_
                .sum()
            )
            for regime, model
            in detector.models.items()
        },
        "threshold": pipeline.threshold,
    }

    result = {
        "model": model_details,
        "metrics": metrics,
        "incident_explanations": (
            explanations
        ),
        "explanation_summary": {
            "timely_explanations": len(
                available
            ),
            "mean_effective_group_count": (
                mean_effective_groups
            ),
            "mean_top3_concentration": (
                mean_top3
            ),
            "mean_normalized_entropy": (
                mean_normalized_entropy
            ),
        },
    }

    return (
        result,
        calibration_scores,
    )


def calibration_threshold_stability(
    calibration_scores: pd.Series,
    *,
    protocol: AlertProtocol,
    bootstrap_replicates: int,
    block_lengths: list[int],
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    if bootstrap_replicates <= 0:
        raise ValueError(
            "bootstrap_replicates must be positive."
        )

    if not 0.0 < confidence < 1.0:
        raise ValueError(
            "confidence must be in (0, 1)."
        )

    smoothed = causal_ewma(
        calibration_scores,
        alpha=protocol.ewma_alpha,
        reset_gap_minutes=(
            protocol.reset_gap_minutes
        ),
    )

    values = smoothed.to_numpy(
        dtype=float
    )

    if (
        len(values) == 0
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "Calibration scores must be finite and non-empty."
        )

    observed = float(
        pd.Series(values).quantile(
            protocol.threshold_quantile,
            interpolation="higher",
        )
    )

    alpha = (
        1.0 - confidence
    )

    rng = np.random.default_rng(
        seed
    )

    modes: dict[
        str,
        Any,
    ] = {}

    n = len(values)

    for block_length in block_lengths:
        if block_length <= 0:
            raise ValueError(
                "block lengths must be positive."
            )

        thresholds = np.empty(
            bootstrap_replicates,
            dtype=float,
        )

        blocks_needed = int(
            np.ceil(
                n
                / block_length
            )
        )

        offsets = np.arange(
            block_length,
            dtype=int,
        )

        for replicate in range(
            bootstrap_replicates
        ):
            starts = rng.integers(
                0,
                n,
                size=blocks_needed,
            )

            indices = (
                starts[:, None]
                + offsets[None, :]
            ) % n

            sample = values[
                indices.ravel()[:n]
            ]

            thresholds[
                replicate
            ] = float(
                np.quantile(
                    sample,
                    protocol.threshold_quantile,
                    method="higher",
                )
            )

        lower = float(
            np.quantile(
                thresholds,
                alpha / 2.0,
            )
        )

        median = float(
            np.median(
                thresholds
            )
        )

        upper = float(
            np.quantile(
                thresholds,
                1.0 - alpha / 2.0,
            )
        )

        if median == 0.0:
            relative_width = None
        else:
            relative_width = float(
                (upper - lower)
                / abs(median)
            )

        modes[
            f"circular_block_{block_length}"
        ] = {
            "block_length": int(
                block_length
            ),
            "replicates": int(
                bootstrap_replicates
            ),
            "lower": lower,
            "median": median,
            "upper": upper,
            "std": float(
                np.std(
                    thresholds,
                    ddof=1,
                )
            ),
            "relative_95_width": (
                relative_width
            ),
            "minimum": float(
                thresholds.min()
            ),
            "maximum": float(
                thresholds.max()
            ),
        }

    return {
        "observed_threshold": observed,
        "calibration_rows": len(values),
        "confidence": confidence,
        "seed": int(seed),
        "modes": modes,
    }


def strict_dominance(
    *,
    candidate: dict[str, Any],
    references: list[dict[str, Any]],
    candidate_stability: dict[str, Any],
    reference_stabilities: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    candidate_metrics = candidate[
        "metrics"
    ]

    candidate_width = (
        candidate_stability[
            "modes"
        ][
            "circular_block_12"
        ][
            "relative_95_width"
        ]
    )

    reference_widths = [
        item[
            "modes"
        ][
            "circular_block_12"
        ][
            "relative_95_width"
        ]
        for item in reference_stabilities
    ]

    if candidate_width is None:
        width_ok = False
    else:
        finite_reference_widths = [
            float(value)
            for value in reference_widths
            if value is not None
        ]

        width_ok = (
            len(finite_reference_widths)
            == len(reference_widths)
            and candidate_width
            <= min(
                finite_reference_widths
            )
        )

    timely_reference = max(
        float(
            item[
                "metrics"
            ][
                "timely_incident_recall"
            ]
        )
        for item in references
    )

    false_reference = min(
        float(
            item[
                "metrics"
            ][
                "false_alerts_per_24h"
            ]
        )
        for item in references
    )

    pr_reference = max(
        float(
            item[
                "metrics"
            ][
                "pr_auc"
            ]
        )
        for item in references
    )

    checks = {
        "timely_incident_recall": (
            float(
                candidate_metrics[
                    "timely_incident_recall"
                ]
            )
            >= timely_reference
        ),
        "false_alerts_per_24h": (
            float(
                candidate_metrics[
                    "false_alerts_per_24h"
                ]
            )
            <= false_reference
        ),
        "pr_auc": (
            float(
                candidate_metrics[
                    "pr_auc"
                ]
            )
            >= pr_reference
        ),
        "calibration_relative_width_block12": (
            width_ok
        ),
    }

    return {
        "strict_dominance": all(
            checks.values()
        ),
        "checks": checks,
        "reference_targets": {
            "timely_incident_recall_minimum": (
                timely_reference
            ),
            "false_alerts_per_24h_maximum": (
                false_reference
            ),
            "pr_auc_minimum": (
                pr_reference
            ),
            "calibration_relative_width_block12_maximum": (
                (
                    min(
                        float(value)
                        for value
                        in reference_widths
                        if value is not None
                    )
                )
                if all(
                    value is not None
                    for value in reference_widths
                )
                else None
            ),
        },
    }
