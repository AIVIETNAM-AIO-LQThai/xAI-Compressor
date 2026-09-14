from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from ml.data.iitk_air_compressor import (
    FEATURE_GROUPS,
)
from ml.detection.feature_energy import (
    explain_feature_energy_detector,
    fit_feature_energy_detector,
)
from ml.detection.pca_detector import (
    fit_pca_detector,
)
from ml.explainability.concentration import (
    explanation_concentration,
)
from ml.explainability.pca import (
    explain_pca_detector,
)


def calibration_threshold(
    scores: pd.Series,
    *,
    quantile: float,
) -> float:
    if not 0.0 < quantile < 1.0:
        raise ValueError(
            "quantile must be in (0, 1)."
        )

    values = scores.to_numpy(
        dtype=float,
    )

    if values.size == 0:
        raise ValueError(
            "Calibration scores cannot be empty."
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Calibration scores must be finite."
        )

    return float(
        np.quantile(
            values,
            quantile,
            method="higher",
        )
    )


def aggregate_iitk_contributions(
    contributions: pd.DataFrame,
) -> pd.DataFrame:
    groups = list(
        dict.fromkeys(
            FEATURE_GROUPS.values()
        )
    )

    output = pd.DataFrame(
        0.0,
        index=contributions.index,
        columns=groups,
        dtype=float,
    )

    unexpected = sorted(
        set(
            contributions.columns
        )
        - set(
            FEATURE_GROUPS
        )
    )

    if unexpected:
        raise ValueError(
            "Unexpected IITK feature columns: "
            f"{unexpected}"
        )

    for feature in (
        contributions.columns
    ):
        output[
            FEATURE_GROUPS[
                feature
            ]
        ] += (
            contributions[
                feature
            ].astype(float)
        )

    return output


def concentration_rows(
    grouped_contributions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict[str, Any]
    ] = []

    for index, row in (
        grouped_contributions
        .iterrows()
    ):
        total = float(
            row.sum()
        )

        if total <= 0.0:
            rows.append(
                {
                    "index": index,
                    "explanation_available": False,
                    "dominant_group": None,
                    "top1_concentration": np.nan,
                    "top3_concentration": np.nan,
                    "normalized_entropy": np.nan,
                    "effective_group_count": np.nan,
                    "herfindahl_index": np.nan,
                }
            )
            continue

        metrics = (
            explanation_concentration(
                {
                    str(group): float(
                        value
                    )
                    for group, value
                    in row.items()
                }
            )
        )

        payload = asdict(
            metrics
        )

        rows.append(
            {
                "index": index,
                "explanation_available": True,
                "dominant_group": (
                    payload[
                        "dominant_group"
                    ]
                ),
                "top1_concentration": (
                    payload[
                        "top1_concentration"
                    ]
                ),
                "top3_concentration": (
                    payload[
                        "top3_concentration"
                    ]
                ),
                "normalized_entropy": (
                    payload[
                        "normalized_entropy"
                    ]
                ),
                "effective_group_count": (
                    payload[
                        "effective_group_count"
                    ]
                ),
                "herfindahl_index": (
                    payload[
                        "herfindahl_index"
                    ]
                ),
            }
        )

    frame = pd.DataFrame(
        rows
    ).set_index(
        "index"
    )

    frame.index.name = (
        grouped_contributions
        .index.name
    )

    return frame


def _distribution_summary(
    values: pd.Series,
) -> dict[str, float | None]:
    finite = (
        values.astype(float)
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna()
    )

    if finite.empty:
        return {
            "mean": None,
            "median": None,
            "q25": None,
            "q75": None,
        }

    return {
        "mean": float(
            finite.mean()
        ),
        "median": float(
            finite.median()
        ),
        "q25": float(
            finite.quantile(
                0.25
            )
        ),
        "q75": float(
            finite.quantile(
                0.75
            )
        ),
    }


def summarize_explanations(
    explanations: pd.DataFrame,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    if not explanations.index.equals(
        metadata.index
    ):
        raise ValueError(
            "Explanation and metadata indexes "
            "must match."
        )

    joined = pd.concat(
        [
            metadata[
                [
                    "condition",
                    "is_fault",
                ]
            ],
            explanations,
        ],
        axis=1,
    )

    summaries = {}

    for condition, frame in (
        joined.groupby(
            "condition",
            sort=True,
        )
    ):
        available = frame.loc[
            frame[
                "explanation_available"
            ]
        ]

        dominant_counts = (
            available[
                "dominant_group"
            ]
            .value_counts()
            .sort_index()
        )

        summaries[
            str(
                condition
            )
        ] = {
            "recordings": len(
                    frame
                ),
            "available_explanations": len(
                    available
                ),
            "dominant_group_counts": {
                str(group): int(
                    count
                )
                for group, count
                in dominant_counts.items()
            },
            "dominant_group_fractions": {
                str(group): float(
                    count
                    / max(
                        len(
                            available
                        ),
                        1,
                    )
                )
                for group, count
                in dominant_counts.items()
            },
            "top1_concentration": (
                _distribution_summary(
                    available[
                        "top1_concentration"
                    ]
                )
            ),
            "top3_concentration": (
                _distribution_summary(
                    available[
                        "top3_concentration"
                    ]
                )
            ),
            "normalized_entropy": (
                _distribution_summary(
                    available[
                        "normalized_entropy"
                    ]
                )
            ),
            "effective_group_count": (
                _distribution_summary(
                    available[
                        "effective_group_count"
                    ]
                )
            ),
            "herfindahl_index": (
                _distribution_summary(
                    available[
                        "herfindahl_index"
                    ]
                )
            ),
        }

    return summaries


def detection_metrics(
    *,
    scores: pd.Series,
    metadata: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    if not scores.index.equals(
        metadata.index
    ):
        raise ValueError(
            "Score and metadata indexes must match."
        )

    truth = metadata[
        "is_fault"
    ].astype(
        bool
    )

    predictions = (
        scores
        >= threshold
    )

    y_true = truth.astype(
        int
    ).to_numpy()

    y_score = scores.to_numpy(
        dtype=float,
    )

    y_pred = predictions.astype(
        int
    ).to_numpy()

    fault_prevalence = float(
        truth.mean()
    )

    healthy = ~truth
    faults = truth

    healthy_fpr = float(
        predictions.loc[
            healthy
        ].mean()
    )

    overall_fault_recall = float(
        predictions.loc[
            faults
        ].mean()
    )

    per_fault_recall = {}
    per_fault_roc_auc = {}
    per_fault_average_precision = {}

    healthy_scores = scores.loc[
        healthy
    ]

    for condition in sorted(
        metadata.loc[
            faults,
            "condition",
        ].unique()
    ):
        condition_mask = (
            metadata[
                "condition"
            ]
            == condition
        )

        per_fault_recall[
            str(
                condition
            )
        ] = float(
            predictions.loc[
                condition_mask
            ].mean()
        )

        class_scores = pd.concat(
            [
                healthy_scores,
                scores.loc[
                    condition_mask
                ],
            ]
        )

        class_truth = pd.concat(
            [
                pd.Series(
                    False,
                    index=healthy_scores.index,
                ),
                pd.Series(
                    True,
                    index=scores.loc[
                        condition_mask
                    ].index,
                ),
            ]
        ).loc[
            class_scores.index
        ]

        per_fault_roc_auc[
            str(
                condition
            )
        ] = float(
            roc_auc_score(
                class_truth.astype(
                    int
                ),
                class_scores,
            )
        )

        per_fault_average_precision[
            str(
                condition
            )
        ] = float(
            average_precision_score(
                class_truth.astype(
                    int
                ),
                class_scores,
            )
        )

    macro_fault_recall = float(
        np.mean(
            list(
                per_fault_recall
                .values()
            )
        )
    )

    return {
        "test_recordings": len(
                metadata
            ),
        "fault_recordings": int(
            faults.sum()
        ),
        "healthy_recordings": int(
            healthy.sum()
        ),
        "fault_prevalence": (
            fault_prevalence
        ),
        "average_precision_prevalence_baseline": (
            fault_prevalence
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_score,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                y_score,
            )
        ),
        "heldout_healthy_false_positive_rate": (
            healthy_fpr
        ),
        "heldout_healthy_specificity": float(
            1.0
            - healthy_fpr
        ),
        "overall_fault_recall": (
            overall_fault_recall
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "macro_fault_recall": (
            macro_fault_recall
        ),
        "per_fault_recall": (
            per_fault_recall
        ),
        "per_fault_roc_auc": (
            per_fault_roc_auc
        ),
        "per_fault_average_precision": (
            per_fault_average_precision
        ),
    }


def run_representation(
    *,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    metadata: pd.DataFrame,
    specification: dict[str, Any],
    threshold_quantile: float,
) -> dict[str, Any]:
    family = str(
        specification[
            "family"
        ]
    )

    scaler_name = str(
        specification[
            "scaler"
        ]
    )

    if family == (
        "pca_reconstruction"
    ):
        detector = (
            fit_pca_detector(
                train,
                features,
                variance_retained=float(
                    specification[
                        "variance_retained"
                    ]
                ),
                scaler_name=scaler_name,
            )
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

        model = {
            "family": family,
            "scaler": (
                scaler_name
            ),
            "feature_count": len(
                    features
                ),
            "components": int(
                detector.model.n_components_
            ),
            "explained_variance_ratio": float(
                detector.model
                .explained_variance_ratio_
                .sum()
            ),
            "variance_retained": float(
                specification[
                    "variance_retained"
                ]
            ),
        }

    elif family == (
        "feature_energy"
    ):
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

        model = {
            "family": family,
            "scaler": (
                scaler_name
            ),
            "feature_count": len(
                    features
                ),
        }

    else:
        raise ValueError(
            "Unknown representation family: "
            f"{family}"
        )

    threshold = (
        calibration_threshold(
            calibration_xai.score,
            quantile=(
                threshold_quantile
            ),
        )
    )

    grouped = (
        aggregate_iitk_contributions(
            test_xai.contributions
        )
    )

    explanations = (
        concentration_rows(
            grouped
        )
    )

    predictions = (
        test_xai.score
        >= threshold
    )

    detected_fault_mask = (
        metadata[
            "is_fault"
        ].astype(bool)
        & predictions
    )

    detected_fault_explanations = (
        explanations.loc[
            detected_fault_mask
        ]
    )

    detected_fault_metadata = (
        metadata.loc[
            detected_fault_mask
        ]
    )

    model[
        "threshold"
    ] = threshold

    model[
        "calibration_score_min"
    ] = float(
        calibration_xai.score.min()
    )

    model[
        "calibration_score_median"
    ] = float(
        calibration_xai.score.median()
    )

    model[
        "calibration_score_max"
    ] = float(
        calibration_xai.score.max()
    )

    return {
        "model": model,
        "metrics": (
            detection_metrics(
                scores=test_xai.score,
                metadata=metadata,
                threshold=threshold,
            )
        ),
        "explanations_all_test": (
            summarize_explanations(
                explanations,
                metadata,
            )
        ),
        "explanations_detected_faults_only": (
            {}
            if detected_fault_metadata.empty
            else summarize_explanations(
                detected_fault_explanations,
                detected_fault_metadata,
            )
        ),
    }
