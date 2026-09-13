from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.detection.alerts import (
    causal_ewma,
    persistent_alerts,
)
from ml.detection.common import (
    transform_frame,
)
from ml.detection.pca_detector import (
    PCADetector,
)
from ml.explainability.pca import (
    feature_group,
)


@dataclass(frozen=True)
class ContributionReference:
    groups: tuple[str, ...]
    sorted_values: dict[str, np.ndarray]
    sample_count: int


@dataclass(frozen=True)
class CounterfactualAlertResult:
    repaired_groups: tuple[str, ...]
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    window_rows: int

    original_raw_score: float
    repaired_raw_score: float

    original_smoothed_score: float
    repaired_smoothed_score: float

    original_alert: bool
    repaired_alert: bool
    alert_cleared: bool


def fit_contribution_reference(
    calibration_contributions: pd.DataFrame,
) -> ContributionReference:
    if calibration_contributions.empty:
        raise ValueError(
            "calibration_contributions cannot be empty."
        )

    sorted_values: dict[
        str,
        np.ndarray,
    ] = {}

    for column in calibration_contributions.columns:
        values = (
            calibration_contributions[
                column
            ]
            .to_numpy(dtype=float)
        )

        if not np.isfinite(values).all():
            raise ValueError(
                "Calibration contributions contain "
                f"non-finite values in {column!r}."
            )

        sorted_values[column] = np.sort(
            values.copy()
        )

    return ContributionReference(
        groups=tuple(
            calibration_contributions.columns
        ),
        sorted_values=sorted_values,
        sample_count=len(
            calibration_contributions
        ),
    )


def contribution_percentiles(
    contributions: pd.DataFrame,
    reference: ContributionReference,
) -> pd.DataFrame:
    missing = (
        set(contributions.columns)
        - set(reference.groups)
    )

    if missing:
        raise ValueError(
            "Contribution groups are missing from "
            f"the calibration reference: {sorted(missing)}"
        )

    percentiles = pd.DataFrame(
        index=contributions.index
    )

    for column in contributions.columns:
        values = (
            contributions[column]
            .to_numpy(dtype=float)
        )

        if not np.isfinite(values).all():
            raise ValueError(
                "Observed contributions contain "
                f"non-finite values in {column!r}."
            )

        reference_values = (
            reference.sorted_values[column]
        )

        left_ranks = np.searchsorted(
            reference_values,
            values,
            side="left",
        )

        right_ranks = np.searchsorted(
            reference_values,
            values,
            side="right",
        )

        mid_ranks = (
            left_ranks.astype(float)
            + right_ranks.astype(float)
        ) / 2.0

        percentiles[column] = (
            mid_ranks
            / reference.sample_count
        )

    return percentiles


def temporal_evidence_window(
    frame: pd.DataFrame,
    *,
    end_time: pd.Timestamp | str,
    window_bins: int,
    bin_minutes: int,
) -> pd.DataFrame:
    if window_bins <= 0:
        raise ValueError(
            "window_bins must be positive."
        )

    if bin_minutes <= 0:
        raise ValueError(
            "bin_minutes must be positive."
        )

    if not isinstance(
        frame.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Expected a DatetimeIndex."
        )

    ordered = frame.sort_index()

    end = pd.Timestamp(end_time)

    if end not in ordered.index:
        raise ValueError(
            "end_time must exist in the frame index."
        )

    start = (
        end
        - pd.Timedelta(
            minutes=(
                bin_minutes
                * (window_bins - 1)
            )
        )
    )

    window = ordered.loc[
        (ordered.index >= start)
        & (ordered.index <= end)
    ].tail(window_bins)

    if window.empty:
        raise ValueError(
            "Temporal evidence window is empty."
        )

    return window


def _feature_group_mask(
    detector: PCADetector,
    groups: Collection[str],
) -> np.ndarray:
    requested = set(groups)

    available = {
        feature_group(feature)
        for feature in detector.features
    }

    unknown = (
        requested
        - available
    )

    if unknown:
        raise ValueError(
            "Unknown feature groups: "
            f"{sorted(unknown)}"
        )

    return np.asarray(
        [
            feature_group(feature)
            in requested
            for feature in detector.features
        ],
        dtype=bool,
    )


def score_with_feature_group_repair(
    frame: pd.DataFrame,
    detector: PCADetector,
    *,
    groups: Collection[str],
) -> pd.Series:
    if frame.empty:
        raise ValueError(
            "frame cannot be empty."
        )

    x = transform_frame(
        frame,
        detector.features,
        detector.scaler,
    )

    encoded = (
        detector.model.transform(x)
    )

    reconstructed = (
        detector.model.inverse_transform(
            encoded
        )
    )

    repair_mask = _feature_group_mask(
        detector,
        groups,
    )

    repaired = x.copy()

    if repair_mask.any():
        repaired[
            :,
            repair_mask,
        ] = reconstructed[
            :,
            repair_mask,
        ]

    repaired_encoded = (
        detector.model.transform(
            repaired
        )
    )

    repaired_reconstruction = (
        detector.model.inverse_transform(
            repaired_encoded
        )
    )

    residual = (
        repaired
        - repaired_reconstruction
    )

    scores = np.mean(
        residual ** 2,
        axis=1,
    )

    return pd.Series(
        scores,
        index=frame.index,
        name="counterfactual_raw_score",
    )


def verify_counterfactual_alert(
    frame: pd.DataFrame,
    detector: PCADetector,
    original_raw_scores: pd.Series,
    *,
    alert_timestamp: pd.Timestamp | str,
    groups: Collection[str],
    window_bins: int,
    bin_minutes: int,
    ewma_alpha: float,
    threshold: float,
    persistence_hits: int,
    persistence_window: int,
    reset_gap_minutes: int,
) -> CounterfactualAlertResult:
    if not isinstance(
        frame.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Expected frame to use a DatetimeIndex."
        )

    if not isinstance(
        original_raw_scores.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Expected scores to use a DatetimeIndex."
        )

    ordered_frame = (
        frame.sort_index()
    )

    ordered_scores = (
        original_raw_scores
        .sort_index()
    )

    if not ordered_frame.index.equals(
        ordered_scores.index
    ):
        raise ValueError(
            "frame and original_raw_scores "
            "must use identical indexes."
        )

    alert_time = pd.Timestamp(
        alert_timestamp
    )

    window = temporal_evidence_window(
        ordered_frame,
        end_time=alert_time,
        window_bins=window_bins,
        bin_minutes=bin_minutes,
    )

    repaired_window_scores = (
        score_with_feature_group_repair(
            window,
            detector,
            groups=groups,
        )
    )

    counterfactual_raw = (
        ordered_scores.copy()
    )

    counterfactual_raw.loc[
        window.index
    ] = repaired_window_scores

    original_smoothed = causal_ewma(
        ordered_scores,
        alpha=ewma_alpha,
        reset_gap_minutes=(
            reset_gap_minutes
        ),
    )

    repaired_smoothed = causal_ewma(
        counterfactual_raw,
        alpha=ewma_alpha,
        reset_gap_minutes=(
            reset_gap_minutes
        ),
    )

    _, original_alerts = persistent_alerts(
        original_smoothed,
        threshold=threshold,
        required_hits=persistence_hits,
        window_bins=persistence_window,
        reset_gap_minutes=(
            reset_gap_minutes
        ),
    )

    _, repaired_alerts = persistent_alerts(
        repaired_smoothed,
        threshold=threshold,
        required_hits=persistence_hits,
        window_bins=persistence_window,
        reset_gap_minutes=(
            reset_gap_minutes
        ),
    )

    original_alert = bool(
        original_alerts.loc[
            alert_time
        ]
    )

    repaired_alert = bool(
        repaired_alerts.loc[
            alert_time
        ]
    )

    return CounterfactualAlertResult(
        repaired_groups=tuple(
            sorted(set(groups))
        ),
        window_start=window.index[0],
        window_end=window.index[-1],
        window_rows=len(window),
        original_raw_score=float(
            ordered_scores.loc[
                alert_time
            ]
        ),
        repaired_raw_score=float(
            counterfactual_raw.loc[
                alert_time
            ]
        ),
        original_smoothed_score=float(
            original_smoothed.loc[
                alert_time
            ]
        ),
        repaired_smoothed_score=float(
            repaired_smoothed.loc[
                alert_time
            ]
        ),
        original_alert=original_alert,
        repaired_alert=repaired_alert,
        alert_cleared=(
            original_alert
            and not repaired_alert
        ),
    )