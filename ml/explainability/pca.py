from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.detection.common import transform_frame
from ml.detection.pca_detector import PCADetector


@dataclass(frozen=True)
class PCAExplanation:
    score: pd.Series
    contributions: pd.DataFrame
    normalized_contributions: pd.DataFrame

def explain_pca_detector(
    frame: pd.DataFrame,
    detector: PCADetector,
) -> PCAExplanation:
    x = transform_frame(
        frame,
        detector.features,
        detector.scaler,
    )

    encoded = detector.model.transform(x)

    reconstructed = (
        detector.model.inverse_transform(
            encoded
        )
    )

    residual_squared = (
        x - reconstructed
    ) ** 2

    feature_count = len(
        detector.features
    )

    contributions_array = (
        residual_squared
        / feature_count
    )

    contributions = pd.DataFrame(
        contributions_array,
        index=frame.index,
        columns=detector.features,
    )

    score = contributions.sum(
        axis=1
    ).rename("raw_score")

    residual_total = (
        residual_squared.sum(axis=1)
    )

    normalized_array = np.divide(
        residual_squared,
        residual_total[:, None],
        out=np.zeros_like(
            residual_squared,
            dtype=float,
        ),
        where=(
            residual_total[:, None] > 0
        ),
    )

    normalized = pd.DataFrame(
        normalized_array,
        index=frame.index,
        columns=detector.features,
    )

    return PCAExplanation(
        score=score,
        contributions=contributions,
        normalized_contributions=normalized,
    )

PRESSURE_RELATIONSHIPS = {
    "tp3_minus_reservoirs",
    "tp2_minus_tp3",
}


def feature_group(
    feature_name: str,
) -> str:
    base_name = feature_name.split(
        "__",
        maxsplit=1,
    )[0]

    if base_name in PRESSURE_RELATIONSHIPS:
        return "pressure_relationships"

    return base_name


def aggregate_contributions(
    contributions: pd.DataFrame,
) -> pd.DataFrame:
    grouped: dict[str, pd.Series] = {}

    for column in contributions.columns:
        group = feature_group(column)

        if group not in grouped:
            grouped[group] = (
                contributions[column].copy()
            )
        else:
            grouped[group] = (
                grouped[group]
                + contributions[column]
            )

    return pd.DataFrame(
        grouped,
        index=contributions.index,
    )

def causal_ewma_contributions(
    contributions: pd.DataFrame,
    *,
    alpha: float,
    reset_gap_minutes: int,
) -> pd.DataFrame:
    if not 0.0 < alpha <= 1.0:
        raise ValueError(
            "alpha must be in (0, 1]."
        )

    frame = contributions.sort_index()

    if not isinstance(
        frame.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Expected a DatetimeIndex."
        )

    reset_gap = pd.Timedelta(
        minutes=reset_gap_minutes
    )

    values = frame.to_numpy(
        dtype=float
    )

    smoothed = np.empty_like(
        values,
        dtype=float,
    )

    previous: np.ndarray | None = None
    previous_time: pd.Timestamp | None = None

    for position, (
        timestamp,
        row,
    ) in enumerate(
        zip(
            frame.index,
            values,
            strict=True,
        )
    ):
        reset = (
            previous_time is None
            or timestamp - previous_time
            > reset_gap
        )

        if reset or previous is None:
            current = row.copy()
        else:
            current = (
                alpha * row
                + (1.0 - alpha)
                * previous
            )

        smoothed[position] = current

        previous = current
        previous_time = timestamp

    return pd.DataFrame(
        smoothed,
        index=frame.index,
        columns=frame.columns,
    )