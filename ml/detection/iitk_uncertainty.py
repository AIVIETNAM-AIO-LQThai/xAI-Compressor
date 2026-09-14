from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml.detection.feature_energy import (
    fit_feature_energy_detector,
    score_feature_energy_detector,
)
from ml.detection.iitk_external_benchmark import (
    calibration_threshold,
)
from ml.detection.pca_detector import (
    fit_pca_detector,
    score_pca_detector,
)


def fit_representation_scores(
    *,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    specification: dict[str, Any],
) -> tuple[pd.Series, pd.Series]:
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

    if family == "pca_reconstruction":
        detector = fit_pca_detector(
            train,
            features,
            variance_retained=float(
                specification[
                    "variance_retained"
                ]
            ),
            scaler_name=scaler_name,
        )

        calibration_scores = (
            score_pca_detector(
                calibration,
                detector,
            )
        )

        test_scores = score_pca_detector(
            test,
            detector,
        )

    elif family == "feature_energy":
        detector = (
            fit_feature_energy_detector(
                train,
                features,
                scaler_name=scaler_name,
            )
        )

        calibration_scores = (
            score_feature_energy_detector(
                calibration,
                detector,
            )
        )

        test_scores = (
            score_feature_energy_detector(
                test,
                detector,
            )
        )

    else:
        raise ValueError(
            "Unknown representation family: "
            f"{family}"
        )

    return (
        calibration_scores,
        test_scores,
    )


def wilson_interval(
    successes: int,
    trials: int,
    *,
    confidence_level: float,
) -> dict[str, float]:
    if trials <= 0:
        raise ValueError(
            "trials must be positive."
        )

    if not 0 <= successes <= trials:
        raise ValueError(
            "successes must be in [0, trials]."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must be in (0, 1)."
        )

    proportion = (
        successes
        / trials
    )

    z = NormalDist().inv_cdf(
        0.5
        + confidence_level
        / 2.0
    )

    z2 = z * z

    denominator = (
        1.0
        + z2
        / trials
    )

    center = (
        proportion
        + z2
        / (
            2.0
            * trials
        )
    ) / denominator

    half_width = (
        z
        / denominator
        * np.sqrt(
            proportion
            * (
                1.0
                - proportion
            )
            / trials
            + z2
            / (
                4.0
                * trials
                * trials
            )
        )
    )

    return {
        "estimate": float(
            proportion
        ),
        "lower": float(
            max(
                0.0,
                center
                - half_width,
            )
        ),
        "upper": float(
            min(
                1.0,
                center
                + half_width,
            )
        ),
        "confidence_level": float(
            confidence_level
        ),
        "successes": int(
            successes
        ),
        "trials": int(
            trials
        ),
    }


def percentile_summary(
    values: np.ndarray,
    *,
    confidence_level: float,
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if (
        array.ndim != 1
        or array.size == 0
    ):
        raise ValueError(
            "values must be a non-empty vector."
        )

    if not np.isfinite(
        array
    ).all():
        raise ValueError(
            "values must be finite."
        )

    alpha = (
        1.0
        - confidence_level
    )

    lower_quantile = (
        alpha
        / 2.0
    )

    upper_quantile = (
        1.0
        - alpha
        / 2.0
    )

    return {
        "mean": float(
            array.mean()
        ),
        "std": float(
            array.std(
                ddof=1
            )
            if array.size > 1
            else 0.0
        ),
        "median": float(
            np.median(
                array
            )
        ),
        "lower": float(
            np.quantile(
                array,
                lower_quantile,
            )
        ),
        "upper": float(
            np.quantile(
                array,
                upper_quantile,
            )
        ),
        "min": float(
            array.min()
        ),
        "max": float(
            array.max()
        ),
        "confidence_level": float(
            confidence_level
        ),
    }


def quantile_order_information(
    sample_count: int,
    *,
    quantile: float,
) -> dict[str, int | float]:
    if sample_count <= 0:
        raise ValueError(
            "sample_count must be positive."
        )

    position = (
        quantile
        * (
            sample_count
            - 1
        )
    )

    index_zero_based = int(
        np.ceil(
            position
        )
    )

    return {
        "sample_count": int(
            sample_count
        ),
        "quantile": float(
            quantile
        ),
        "higher_index_zero_based": (
            index_zero_based
        ),
        "higher_rank_one_based": int(
            index_zero_based
            + 1
        ),
        "upper_order_statistics_including_threshold": int(
            sample_count
            - index_zero_based
        ),
    }


def leave_one_out_thresholds(
    scores: pd.Series,
    *,
    quantile: float,
) -> np.ndarray:
    values = scores.to_numpy(
        dtype=np.float64,
    )

    if values.size < 3:
        raise ValueError(
            "Need at least three calibration scores."
        )

    output = np.empty(
        values.size,
        dtype=np.float64,
    )

    for index in range(
        values.size
    ):
        reduced = np.delete(
            values,
            index,
        )

        output[
            index
        ] = calibration_threshold(
            pd.Series(
                reduced
            ),
            quantile=quantile,
        )

    return output


def _circular_block_sample_indices(
    *,
    size: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not 1 <= block_length <= size:
        raise ValueError(
            "block_length must be in [1, size]."
        )

    block_count = int(
        np.ceil(
            size
            / block_length
        )
    )

    starts = rng.integers(
        0,
        size,
        size=block_count,
    )

    offsets = np.arange(
        block_length,
        dtype=int,
    )

    blocks = (
        starts[
            :,
            None,
        ]
        + offsets[
            None,
            :,
        ]
    ) % size

    return (
        blocks.reshape(
            -1
        )[:size]
    )


def bootstrap_calibration_thresholds(
    scores: pd.Series,
    *,
    quantile: float,
    iterations: int,
    seed: int,
    block_length: int | None,
) -> np.ndarray:
    values = scores.to_numpy(
        dtype=np.float64,
    )

    if iterations <= 0:
        raise ValueError(
            "iterations must be positive."
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Calibration scores must be finite."
        )

    rng = np.random.default_rng(
        seed
    )

    thresholds = np.empty(
        iterations,
        dtype=np.float64,
    )

    if block_length is None:
        indices = rng.integers(
            0,
            values.size,
            size=(
                iterations,
                values.size,
            ),
        )

        samples = values[
            indices
        ]

        thresholds[:] = np.quantile(
            samples,
            quantile,
            axis=1,
            method="higher",
        )

        return thresholds

    for iteration in range(
        iterations
    ):
        indices = (
            _circular_block_sample_indices(
                size=values.size,
                block_length=block_length,
                rng=rng,
            )
        )

        sample = values[
            indices
        ]

        thresholds[
            iteration
        ] = float(
            np.quantile(
                sample,
                quantile,
                method="higher",
            )
        )

    return thresholds


def threshold_operating_distributions(
    *,
    thresholds: np.ndarray,
    test_scores: pd.Series,
    metadata: pd.DataFrame,
) -> dict[str, np.ndarray]:
    if not test_scores.index.equals(
        metadata.index
    ):
        raise ValueError(
            "test_scores and metadata indexes must match."
        )

    threshold_array = np.asarray(
        thresholds,
        dtype=np.float64,
    )

    scores = test_scores.to_numpy(
        dtype=np.float64,
    )

    predictions = (
        scores[
            None,
            :,
        ]
        >= threshold_array[
            :,
            None,
        ]
    )

    truth = metadata[
        "is_fault"
    ].astype(
        bool
    ).to_numpy()

    healthy_mask = ~truth
    fault_mask = truth

    fpr = (
        predictions[
            :,
            healthy_mask,
        ]
        .mean(
            axis=1
        )
    )

    overall_recall = (
        predictions[
            :,
            fault_mask,
        ]
        .mean(
            axis=1
        )
    )

    specificity = (
        1.0
        - fpr
    )

    balanced_accuracy = (
        specificity
        + overall_recall
    ) / 2.0

    condition_values = metadata[
        "condition"
    ].astype(
        str
    ).to_numpy()

    per_fault: dict[
        str,
        np.ndarray,
    ] = {}

    for condition in sorted(
        np.unique(
            condition_values[
                fault_mask
            ]
        )
    ):
        mask = (
            condition_values
            == condition
        )

        per_fault[
            str(
                condition
            )
        ] = (
            predictions[
                :,
                mask,
            ]
            .mean(
                axis=1
            )
        )

    macro_recall = np.mean(
        np.column_stack(
            list(
                per_fault.values()
            )
        ),
        axis=1,
    )

    output = {
        "heldout_healthy_false_positive_rate": (
            fpr
        ),
        "overall_fault_recall": (
            overall_recall
        ),
        "macro_fault_recall": (
            macro_recall
        ),
        "balanced_accuracy": (
            balanced_accuracy
        ),
    }

    for condition, values in (
        per_fault.items()
    ):
        output[
            f"per_fault_recall::{condition}"
        ] = values

    return output


def summarize_threshold_operating_distributions(
    *,
    thresholds: np.ndarray,
    test_scores: pd.Series,
    metadata: pd.DataFrame,
    confidence_level: float,
) -> dict[str, Any]:
    distributions = (
        threshold_operating_distributions(
            thresholds=thresholds,
            test_scores=test_scores,
            metadata=metadata,
        )
    )

    return {
        "threshold": (
            percentile_summary(
                thresholds,
                confidence_level=confidence_level,
            )
        ),
        "operating_metrics": {
            metric: percentile_summary(
                values,
                confidence_level=confidence_level,
            )
            for metric, values
            in distributions.items()
        },
    }


def top_calibration_scores(
    scores: pd.Series,
    metadata: pd.DataFrame,
    *,
    count: int,
) -> list[dict[str, Any]]:
    if not scores.index.equals(
        metadata.index
    ):
        raise ValueError(
            "scores and metadata indexes must match."
        )

    ranked = scores.sort_values(
        ascending=False
    )

    output = []

    for index, score in (
        ranked.iloc[
            :count
        ].items()
    ):
        output.append(
            {
                "reading": int(
                    metadata.loc[
                        index,
                        "reading",
                    ]
                ),
                "score": float(
                    score
                ),
            }
        )

    return output


def stratified_auc_bootstrap(
    *,
    scores: pd.Series,
    metadata: pd.DataFrame,
    iterations: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    if not scores.index.equals(
        metadata.index
    ):
        raise ValueError(
            "scores and metadata indexes must match."
        )

    rng = np.random.default_rng(
        seed
    )

    truth = metadata[
        "is_fault"
    ].astype(
        bool
    )

    healthy_scores = scores.loc[
        ~truth
    ].to_numpy(
        dtype=np.float64,
    )

    fault_conditions = sorted(
        metadata.loc[
            truth,
            "condition",
        ]
        .astype(
            str
        )
        .unique()
    )

    fault_scores = {
        condition: scores.loc[
            (
                truth
                & (
                    metadata[
                        "condition"
                    ].astype(
                        str
                    )
                    == condition
                )
            )
        ].to_numpy(
            dtype=np.float64,
        )
        for condition in (
            fault_conditions
        )
    }

    overall = np.empty(
        iterations,
        dtype=np.float64,
    )

    per_fault = {
        condition: np.empty(
            iterations,
            dtype=np.float64,
        )
        for condition in (
            fault_conditions
        )
    }

    for iteration in range(
        iterations
    ):
        healthy_sample = (
            healthy_scores[
                rng.integers(
                    0,
                    healthy_scores.size,
                    size=healthy_scores.size,
                )
            ]
        )

        sampled_faults = {}

        for condition in (
            fault_conditions
        ):
            source = fault_scores[
                condition
            ]

            sampled_faults[
                condition
            ] = source[
                rng.integers(
                    0,
                    source.size,
                    size=source.size,
                )
            ]

        positive_sample = np.concatenate(
            [
                sampled_faults[
                    condition
                ]
                for condition
                in fault_conditions
            ]
        )

        combined_scores = np.concatenate(
            [
                healthy_sample,
                positive_sample,
            ]
        )

        combined_truth = np.concatenate(
            [
                np.zeros(
                    healthy_sample.size,
                    dtype=int,
                ),
                np.ones(
                    positive_sample.size,
                    dtype=int,
                ),
            ]
        )

        overall[
            iteration
        ] = roc_auc_score(
            combined_truth,
            combined_scores,
        )

        for condition in (
            fault_conditions
        ):
            class_scores = np.concatenate(
                [
                    healthy_sample,
                    sampled_faults[
                        condition
                    ],
                ]
            )

            class_truth = np.concatenate(
                [
                    np.zeros(
                        healthy_sample.size,
                        dtype=int,
                    ),
                    np.ones(
                        sampled_faults[
                            condition
                        ].size,
                        dtype=int,
                    ),
                ]
            )

            per_fault[
                condition
            ][
                iteration
            ] = roc_auc_score(
                class_truth,
                class_scores,
            )

    return {
        "overall_roc_auc": (
            percentile_summary(
                overall,
                confidence_level=confidence_level,
            )
        ),
        "per_fault_roc_auc": {
            condition: percentile_summary(
                values,
                confidence_level=confidence_level,
            )
            for condition, values
            in per_fault.items()
        },
        "iterations": int(
            iterations
        ),
        "seed": int(
            seed
        ),
    }
