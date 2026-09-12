from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pandas as pd


def require_datetime_index(
    series: pd.Series,
) -> pd.DatetimeIndex:
    if not isinstance(
        series.index, pd.DatetimeIndex
    ):
        raise TypeError("Expected a DatetimeIndex.")
    return series.index

@dataclass(frozen=True)
class AlertEpisode:
    start: pd.Timestamp
    end: pd.Timestamp
    alert_bins: int


def causal_ewma(
    scores: pd.Series,
    *,
    alpha: float,
    reset_gap_minutes: int,
) -> pd.Series:
    if not 0.0 < alpha <= 1.0:
        raise ValueError(
            "alpha must be in (0, 1]."
        )

    scores = scores.sort_index()

    reset_gap = pd.Timedelta(
        minutes=reset_gap_minutes
    )

    smoothed: list[float] = []

    previous_value: float | None = None
    previous_time: pd.Timestamp | None = None

    timestamps = require_datetime_index(scores)

    values = scores.to_numpy(dtype=float)

    for timestamp, value in zip(
        timestamps, values, strict=True,
    ):
        reset = (
            previous_time is None
            or timestamp - previous_time
            > reset_gap
        )

        if reset or previous_value is None:
            current = value
        else:
            current = (
                alpha * value
                + (1.0 - alpha)
                * previous_value
            )

        smoothed.append(current)

        previous_value = current
        previous_time = timestamp

    return pd.Series(
        smoothed,
        index=scores.index,
        name="smoothed_score",
    )


def persistent_alerts(
    scores: pd.Series,
    *,
    threshold: float,
    required_hits: int,
    window_bins: int,
    reset_gap_minutes: int,
) -> tuple[pd.Series, pd.Series]:
    if required_hits > window_bins:
        raise ValueError(
            "required_hits cannot exceed window_bins."
        )

    scores = scores.sort_index()

    threshold_hits = (
        scores >= threshold
    )

    reset_gap = pd.Timedelta(
        minutes=reset_gap_minutes
    )

    history: deque[bool] = deque(
        maxlen=window_bins
    )

    alerts: list[bool] = []

    previous_time: pd.Timestamp | None = None

    timestamps = require_datetime_index(threshold_hits)
    hits = threshold_hits.to_numpy(dtype=bool)

    for timestamp, hit in zip(timestamps, hits, strict=True):
        if (
            previous_time is None
            or timestamp - previous_time
            > reset_gap
        ):
            history.clear()

        history.append(bool(hit))

        alerts.append(sum(history) >= required_hits)

        previous_time = timestamp

    return (
        threshold_hits.rename("threshold_hit"),
        pd.Series(
            alerts,
            index=scores.index,
            name="alert",
            dtype=bool,
        ),
    )


def extract_alert_episodes(
    alerts: pd.Series,
    *,
    merge_minutes: int,
    reset_gap_minutes: int,
) -> list[AlertEpisode]:
    alerts = alerts.sort_index()

    merge_gap = pd.Timedelta(
        minutes=merge_minutes
    )

    reset_gap = pd.Timedelta(
        minutes=reset_gap_minutes
    )

    episodes: list[AlertEpisode] = []

    current_start: pd.Timestamp | None = None
    current_end: pd.Timestamp | None = None
    current_bins = 0

    previous_observation: pd.Timestamp | None = None
    previous_alert: pd.Timestamp | None = None

    def close_current() -> None:
        nonlocal current_start
        nonlocal current_end
        nonlocal current_bins

        if (
            current_start is not None
            and current_end is not None
        ):
            episodes.append(
                AlertEpisode(
                    start=current_start,
                    end=current_end,
                    alert_bins=current_bins,
                )
            )

        current_start = None
        current_end = None
        current_bins = 0

    timestamps = require_datetime_index(alerts)
    active_values = alerts.to_numpy(dtype=bool)

    for timestamp, active in zip(timestamps, active_values, strict=True):
        if (
            previous_observation is not None
            and timestamp - previous_observation
            > reset_gap
        ):
            close_current()
            previous_alert = None

        if bool(active):
            if current_start is None:
                current_start = timestamp
                current_end = timestamp
                current_bins = 1

            elif (
                previous_alert is not None
                and timestamp - previous_alert
                <= merge_gap
            ):
                current_end = timestamp
                current_bins += 1

            else:
                close_current()

                current_start = timestamp
                current_end = timestamp
                current_bins = 1

            previous_alert = timestamp

        previous_observation = timestamp

    close_current()

    return episodes