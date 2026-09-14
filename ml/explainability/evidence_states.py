from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScalarReference:
    sorted_values: np.ndarray
    sample_count: int


def fit_scalar_reference(
    values: pd.Series,
) -> ScalarReference:
    array = values.to_numpy(
        dtype=float,
    )

    array = array[
        np.isfinite(array)
    ]

    if array.size == 0:
        raise ValueError(
            "Reference values contain no finite observations."
        )

    return ScalarReference(
        sorted_values=np.sort(
            array.copy()
        ),
        sample_count=int(
            array.size
        ),
    )


def scalar_percentiles(
    values: pd.Series,
    reference: ScalarReference,
) -> pd.Series:
    array = values.to_numpy(
        dtype=float,
    )

    output = np.full(
        shape=array.shape,
        fill_value=np.nan,
        dtype=float,
    )

    finite = np.isfinite(
        array
    )

    finite_values = array[
        finite
    ]

    left = np.searchsorted(
        reference.sorted_values,
        finite_values,
        side="left",
    )

    right = np.searchsorted(
        reference.sorted_values,
        finite_values,
        side="right",
    )

    midranks = (
        left.astype(float)
        + right.astype(float)
    ) / 2.0

    output[
        finite
    ] = (
        midranks
        / reference.sample_count
    )

    return pd.Series(
        output,
        index=values.index,
        name=(
            f"{values.name}_percentile"
            if values.name
            else "percentile"
        ),
    )


def reference_quantiles(
    reference: ScalarReference,
    quantiles: Sequence[float],
) -> dict[str, float]:
    if not quantiles:
        raise ValueError(
            "quantiles cannot be empty."
        )

    values = pd.Series(
        reference.sorted_values
    )

    output: dict[
        str,
        float,
    ] = {}

    for quantile in quantiles:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(
                "quantiles must be in [0, 1]."
            )

        output[
            f"q{quantile:g}"
        ] = float(
            values.quantile(
                quantile,
                interpolation="linear",
            )
        )

    return output


def classify_spread_state(
    percentiles: pd.Series,
    *,
    concentrated_percentile_max: float,
    diffuse_percentile_min: float,
) -> pd.Series:
    if not (
        0.0
        <= concentrated_percentile_max
        < diffuse_percentile_min
        <= 1.0
    ):
        raise ValueError(
            "Spread percentile cutoffs must satisfy "
            "0 <= concentrated < diffuse <= 1."
        )

    states = pd.Series(
        "typical",
        index=percentiles.index,
        dtype="object",
        name="spread_state",
    )

    states.loc[
        percentiles
        <= concentrated_percentile_max
    ] = "concentrated"

    states.loc[
        percentiles
        >= diffuse_percentile_min
    ] = "diffuse"

    states.loc[
        percentiles.isna()
    ] = "unavailable"

    return states


def classify_magnitude_state(
    *,
    score: pd.Series,
    threshold: float,
    alerts: pd.Series,
) -> pd.Series:
    if threshold <= 0.0:
        raise ValueError(
            "threshold must be positive."
        )

    if not score.index.equals(
        alerts.index
    ):
        raise ValueError(
            "score and alerts must share the same index."
        )

    states = pd.Series(
        "subthreshold",
        index=score.index,
        dtype="object",
        name="magnitude_state",
    )

    states.loc[
        score >= threshold
    ] = "threshold_crossing"

    # Persistence-aware alert state takes precedence even if
    # the current score has just fallen below threshold.
    states.loc[
        alerts.astype(bool)
    ] = "alerting"

    return states


def build_evidence_state_frame(
    *,
    trajectory: pd.DataFrame,
    score: pd.Series,
    alerts: pd.Series,
    threshold: float,
    score_reference: ScalarReference,
    spread_reference: ScalarReference,
    concentrated_percentile_max: float,
    diffuse_percentile_min: float,
) -> pd.DataFrame:
    required = {
        "effective_group_count",
        "top1_concentration",
        "top3_concentration",
        "dominant_group",
    }

    missing = sorted(
        required
        - set(
            trajectory.columns
        )
    )

    if missing:
        raise ValueError(
            "Trajectory is missing required columns: "
            f"{missing}"
        )

    if not (
        trajectory.index.equals(
            score.index
        )
        and trajectory.index.equals(
            alerts.index
        )
    ):
        raise ValueError(
            "trajectory, score, and alerts must share the same index."
        )

    frame = trajectory.copy()

    score_named = score.rename(
        "smoothed_score"
    )

    score_percentile = (
        scalar_percentiles(
            score_named,
            score_reference,
        )
        .rename(
            "score_calibration_percentile"
        )
    )

    spread_named = (
        frame[
            "effective_group_count"
        ].rename(
            "effective_group_count"
        )
    )

    spread_percentile = (
        scalar_percentiles(
            spread_named,
            spread_reference,
        )
        .rename(
            "spread_calibration_percentile"
        )
    )

    magnitude_state = (
        classify_magnitude_state(
            score=score_named,
            threshold=threshold,
            alerts=alerts,
        )
    )

    spread_state = (
        classify_spread_state(
            spread_percentile,
            concentrated_percentile_max=(
                concentrated_percentile_max
            ),
            diffuse_percentile_min=(
                diffuse_percentile_min
            ),
        )
    )

    frame[
        "smoothed_score"
    ] = score_named

    frame[
        "score_to_threshold"
    ] = (
        score_named
        / threshold
    )

    frame[
        "score_calibration_percentile"
    ] = score_percentile

    frame[
        "spread_calibration_percentile"
    ] = spread_percentile

    frame[
        "magnitude_state"
    ] = magnitude_state

    frame[
        "spread_state"
    ] = spread_state

    frame[
        "joint_state"
    ] = (
        magnitude_state.astype(str)
        + "|"
        + spread_state.astype(str)
    )

    return frame


def state_runs(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    if "joint_state" not in frame.columns:
        raise ValueError(
            "joint_state column is required."
        )

    ordered = frame.sort_index()

    runs: list[
        dict[str, Any]
    ] = []

    run_start = ordered.index[0]
    previous_time = ordered.index[0]
    state = str(
        ordered.iloc[0][
            "joint_state"
        ]
    )
    bins = 1

    for timestamp, current_state in zip(
        ordered.index[1:],
        ordered[
            "joint_state"
        ].iloc[1:],
        strict=True,
    ):
        current_state = str(
            current_state
        )

        if current_state == state:
            bins += 1
            previous_time = timestamp
            continue

        runs.append(
            {
                "state": state,
                "start": str(
                    run_start
                ),
                "end": str(
                    previous_time
                ),
                "bins": bins,
            }
        )

        run_start = timestamp
        previous_time = timestamp
        state = current_state
        bins = 1

    runs.append(
        {
            "state": state,
            "start": str(
                run_start
            ),
            "end": str(
                previous_time
            ),
            "bins": bins,
        }
    )

    return runs


def summarize_state_phase(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "valid_bins": 0,
            "dominant_joint_state": None,
            "joint_state_occupancy": {},
            "state_transition_count": 0,
            "median_score_percentile": None,
            "median_spread_percentile": None,
        }

    states = [
        str(value)
        for value in frame[
            "joint_state"
        ].tolist()
    ]

    counts = Counter(
        states
    )

    occupancy = {
        state: (
            count
            / len(states)
        )
        for state, count
        in sorted(
            counts.items()
        )
    }

    transitions = sum(
        int(
            current
            != previous
        )
        for previous, current
        in pairwise(
            states
        )
    )

    dominant = (
        counts.most_common(
            1
        )[0][0]
    )

    return {
        "valid_bins": len(frame),
        "dominant_joint_state": (
            dominant
        ),
        "joint_state_occupancy": (
            occupancy
        ),
        "state_transition_count": (
            transitions
        ),
        "median_score_percentile": (
            float(
                frame[
                    "score_calibration_percentile"
                ].median()
            )
        ),
        "median_spread_percentile": (
            float(
                frame[
                    "spread_calibration_percentile"
                ].median()
            )
        ),
    }


def first_state_entry(
    frame: pd.DataFrame,
    *,
    joint_state: str,
    onset: pd.Timestamp | str,
) -> dict[str, Any] | None:
    matches = frame.loc[
        frame[
            "joint_state"
        ]
        == joint_state
    ]

    if matches.empty:
        return None

    timestamp = matches.index[
        0
    ]

    onset_time = pd.Timestamp(
        onset
    )

    offset_hours = (
        timestamp
        - onset_time
    ).total_seconds() / 3600.0

    return {
        "timestamp": str(
            timestamp
        ),
        "hours_from_onset": float(
            offset_hours
        ),
        "pre_onset": bool(
            offset_hours < 0.0
        ),
    }


def state_at_onset_boundary(
    frame: pd.DataFrame,
    *,
    onset: pd.Timestamp | str,
) -> dict[str, Any]:
    if frame.empty:
        return {
            "last_pre_onset": None,
            "first_post_onset": None,
        }

    onset_time = pd.Timestamp(
        onset
    )

    before = frame.loc[
        frame.index
        < onset_time
    ]

    after = frame.loc[
        frame.index
        >= onset_time
    ]

    def summarize_row(
        timestamp: pd.Timestamp,
        row: pd.Series,
    ) -> dict[str, Any]:
        return {
            "timestamp": str(
                timestamp
            ),
            "joint_state": str(
                row[
                    "joint_state"
                ]
            ),
            "magnitude_state": str(
                row[
                    "magnitude_state"
                ]
            ),
            "spread_state": str(
                row[
                    "spread_state"
                ]
            ),
            "score_percentile": float(
                row[
                    "score_calibration_percentile"
                ]
            ),
            "spread_percentile": float(
                row[
                    "spread_calibration_percentile"
                ]
            ),
            "effective_groups": float(
                row[
                    "effective_group_count"
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
        }

    last_pre = None

    if not before.empty:
        timestamp = before.index[
            -1
        ]

        last_pre = summarize_row(
            timestamp,
            before.iloc[
                -1
            ],
        )

    first_post = None

    if not after.empty:
        timestamp = after.index[
            0
        ]

        first_post = summarize_row(
            timestamp,
            after.iloc[
                0
            ],
        )

    return {
        "last_pre_onset": (
            last_pre
        ),
        "first_post_onset": (
            first_post
        ),
    }
