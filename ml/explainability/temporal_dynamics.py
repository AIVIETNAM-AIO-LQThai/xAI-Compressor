from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from ml.detection.alerts import persistent_alerts
from ml.detection.pca_research_benchmark import AlertProtocol
from ml.explainability.concentration import explanation_concentration


def concentration_trajectory(
    smoothed_contributions: pd.DataFrame,
    *,
    threshold: float,
    protocol: AlertProtocol,
) -> pd.DataFrame:
    if smoothed_contributions.empty:
        raise ValueError(
            "smoothed_contributions cannot be empty."
        )

    if not isinstance(
        smoothed_contributions.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "Expected a DatetimeIndex."
        )

    ordered = (
        smoothed_contributions
        .sort_index()
        .astype(float)
    )

    if (
        ordered.to_numpy()
        < 0.0
    ).any():
        raise ValueError(
            "Contribution values cannot be negative."
        )

    score = (
        ordered
        .sum(axis=1)
        .rename("smoothed_score")
    )

    _, alerts = persistent_alerts(
        score,
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

    rows: list[
        dict[str, Any]
    ] = []

    for timestamp, row in (
        ordered.iterrows()
    ):
        values = row.to_numpy(
            dtype=float
        )

        total = float(
            values.sum()
        )

        if total <= 0.0:
            rows.append(
                {
                    "timestamp": (
                        timestamp
                    ),
                    "smoothed_score": 0.0,
                    "score_to_threshold": 0.0,
                    "above_threshold": False,
                    "alert": bool(
                        alerts.loc[
                            timestamp
                        ]
                    ),
                    "dominant_group": None,
                    "top1_concentration": np.nan,
                    "top3_concentration": np.nan,
                    "normalized_entropy": np.nan,
                    "effective_group_count": np.nan,
                    "herfindahl_index": np.nan,
                }
            )
            continue

        shares = {
            str(group): (
                float(value)
                / total
            )
            for group, value
            in row.items()
        }

        metrics = (
            explanation_concentration(
                shares
            )
        )

        rows.append(
            {
                "timestamp": (
                    timestamp
                ),
                "smoothed_score": (
                    total
                ),
                "score_to_threshold": (
                    total
                    / threshold
                ),
                "above_threshold": (
                    total >= threshold
                ),
                "alert": bool(
                    alerts.loc[
                        timestamp
                    ]
                ),
                "dominant_group": (
                    metrics.dominant_group
                ),
                "top1_concentration": (
                    metrics
                    .top1_concentration
                ),
                "top3_concentration": (
                    metrics
                    .top3_concentration
                ),
                "normalized_entropy": (
                    metrics
                    .normalized_entropy
                ),
                "effective_group_count": (
                    metrics
                    .effective_group_count
                ),
                "herfindahl_index": (
                    metrics
                    .herfindahl_index
                ),
            }
        )

    frame = (
        pd.DataFrame(
            rows
        )
        .set_index(
            "timestamp"
        )
        .sort_index()
    )

    return frame


def incident_window(
    trajectory: pd.DataFrame,
    *,
    onset: pd.Timestamp | str,
    pre_onset_hours: float,
    post_onset_hours: float,
) -> pd.DataFrame:
    onset_time = pd.Timestamp(
        onset
    )

    start = onset_time - pd.Timedelta(
        hours=pre_onset_hours
    )

    end = onset_time + pd.Timedelta(
        hours=post_onset_hours
    )

    window = trajectory.loc[
        (
            trajectory.index
            >= start
        )
        & (
            trajectory.index
            <= end
        )
    ].copy()

    window[
        "hours_from_onset"
    ] = (
        (
            window.index
            - onset_time
        )
        .total_seconds()
        / 3600.0
    )

    return window


def phase_mask(
    hours_from_onset: pd.Series,
    *,
    start_hours: float,
    end_hours: float,
    is_last_phase: bool,
) -> pd.Series:
    if start_hours >= end_hours:
        raise ValueError(
            "phase start must precede phase end."
        )

    if is_last_phase:
        return (
            (
                hours_from_onset
                >= start_hours
            )
            & (
                hours_from_onset
                <= end_hours
            )
        )

    return (
        (
            hours_from_onset
            >= start_hours
        )
        & (
            hours_from_onset
            < end_hours
        )
    )


def summarize_phase(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "valid_bins": 0,
            "alert_bins": 0,
            "alert_fraction": None,
            "median_top1": None,
            "median_top3": None,
            "median_effective_groups": None,
            "max_effective_groups": None,
            "max_effective_groups_timestamp": None,
            "dominant_group_mode": None,
            "dominant_group_switches": 0,
        }

    concentration_frame = (
        frame.dropna(
            subset=[
                "effective_group_count"
            ]
        )
    )

    dominant = [
        str(value)
        for value in (
            concentration_frame[
                "dominant_group"
            ]
            .dropna()
            .tolist()
        )
    ]

    switches = 0

    for previous, current in pairwise(dominant):
        switches += int(current != previous)

    mode = None

    if dominant:
        mode = Counter(
            dominant
        ).most_common(1)[0][0]

    max_effective = None
    max_timestamp = None

    if not (
        concentration_frame.empty
    ):
        position = (
            concentration_frame[
                "effective_group_count"
            ].idxmax()
        )

        max_effective = float(
            concentration_frame.loc[
                position,
                "effective_group_count",
            ]
        )

        max_timestamp = str(
            position
        )

    return {
        "valid_bins": len(frame),
        "alert_bins": int(
            frame[
                "alert"
            ].sum()
        ),
        "alert_fraction": float(
            frame[
                "alert"
            ].mean()
        ),
        "median_top1": (
            None
            if concentration_frame.empty
            else float(
                concentration_frame[
                    "top1_concentration"
                ].median()
            )
        ),
        "median_top3": (
            None
            if concentration_frame.empty
            else float(
                concentration_frame[
                    "top3_concentration"
                ].median()
            )
        ),
        "median_effective_groups": (
            None
            if concentration_frame.empty
            else float(
                concentration_frame[
                    "effective_group_count"
                ].median()
            )
        ),
        "max_effective_groups": (
            max_effective
        ),
        "max_effective_groups_timestamp": (
            max_timestamp
        ),
        "dominant_group_mode": mode,
        "dominant_group_switches": (
            switches
        ),
    }


def summarize_incident_dynamics(
    window: pd.DataFrame,
    *,
    phases: Sequence[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    phase_summaries = {}

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

        phase_summaries[
            str(
                phase[
                    "name"
                ]
            )
        ] = summarize_phase(
            window.loc[
                mask
            ]
        )

    alert_active = window.loc[
        window[
            "alert"
        ]
    ]

    alert_active = (
        alert_active.dropna(
            subset=[
                "effective_group_count"
            ]
        )
    )

    peak_alert_effective = None
    peak_alert_timestamp = None

    if not alert_active.empty:
        timestamp = (
            alert_active[
                "effective_group_count"
            ].idxmax()
        )

        peak_alert_effective = float(
            alert_active.loc[
                timestamp,
                "effective_group_count",
            ]
        )

        peak_alert_timestamp = str(
            timestamp
        )

    return {
        "window_bins": len(window),
        "window_start": (
            None
            if window.empty
            else str(
                window.index.min()
            )
        ),
        "window_end": (
            None
            if window.empty
            else str(
                window.index.max()
            )
        ),
        "phase_summaries": (
            phase_summaries
        ),
        "peak_effective_groups_while_alerting": (
            peak_alert_effective
        ),
        "peak_effective_groups_while_alerting_timestamp": (
            peak_alert_timestamp
        ),
    }


def compact_trajectory_rows(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    output = []

    for timestamp, row in (
        frame.iterrows()
    ):
        output.append(
            {
                "timestamp": str(
                    timestamp
                ),
                "hours_from_onset": float(
                    row[
                        "hours_from_onset"
                    ]
                ),
                "smoothed_score": float(
                    row[
                        "smoothed_score"
                    ]
                ),
                "score_to_threshold": float(
                    row[
                        "score_to_threshold"
                    ]
                ),
                "alert": bool(
                    row[
                        "alert"
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
                "top1": (
                    None
                    if pd.isna(
                        row[
                            "top1_concentration"
                        ]
                    )
                    else float(
                        row[
                            "top1_concentration"
                        ]
                    )
                ),
                "top3": (
                    None
                    if pd.isna(
                        row[
                            "top3_concentration"
                        ]
                    )
                    else float(
                        row[
                            "top3_concentration"
                        ]
                    )
                ),
                "effective_groups": (
                    None
                    if pd.isna(
                        row[
                            "effective_group_count"
                        ]
                    )
                    else float(
                        row[
                            "effective_group_count"
                        ]
                    )
                ),
            }
        )

    return output
