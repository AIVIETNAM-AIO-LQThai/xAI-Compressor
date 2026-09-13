from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CausalSequenceBatch:
    inputs: np.ndarray
    targets: np.ndarray
    target_index: pd.DatetimeIndex
    sequence_length: int
    feature_count: int

    def __post_init__(self) -> None:
        if self.inputs.ndim != 3:
            raise ValueError(
                "inputs must have shape "
                "(samples, sequence, features)."
            )

        if self.targets.ndim != 2:
            raise ValueError(
                "targets must have shape "
                "(samples, features)."
            )

        if (
            self.inputs.shape[0]
            != self.targets.shape[0]
            or self.inputs.shape[0]
            != len(self.target_index)
        ):
            raise ValueError(
                "Sequence samples, targets, and "
                "timestamps must have equal length."
            )

        if (
            self.inputs.shape[1]
            != self.sequence_length
        ):
            raise ValueError(
                "Input sequence dimension does "
                "not match sequence_length."
            )

        if (
            self.inputs.shape[2]
            != self.feature_count
            or self.targets.shape[1]
            != self.feature_count
        ):
            raise ValueError(
                "Feature dimension does not "
                "match feature_count."
            )


def build_causal_sequences(
    values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    *,
    sequence_length: int,
    expected_step: pd.Timedelta,
) -> CausalSequenceBatch:
    array = np.asarray(
        values,
        dtype=np.float32,
    )

    index = pd.DatetimeIndex(
        timestamps
    )

    if array.ndim != 2:
        raise ValueError(
            "values must be a 2D array."
        )

    if len(array) != len(index):
        raise ValueError(
            "values and timestamps must "
            "have equal length."
        )

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be positive."
        )

    if expected_step <= pd.Timedelta(0):
        raise ValueError(
            "expected_step must be positive."
        )

    if len(index) <= sequence_length:
        raise ValueError(
            "Not enough rows to build one "
            "causal prediction sequence."
        )

    if index.has_duplicates:
        raise ValueError(
            "timestamps must not contain "
            "duplicates."
        )

    if not index.is_monotonic_increasing:
        raise ValueError(
            "timestamps must be sorted in "
            "strictly increasing order."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            "values contain non-finite entries."
        )

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_times: list[pd.Timestamp] = []

    for target_position in range(
        sequence_length,
        len(index),
    ):
        window_start = (
            target_position
            - sequence_length
        )

        time_window = index[
            window_start:
            target_position + 1
        ]

        deltas = (
            time_window[1:]
            - time_window[:-1]
        )

        if not (
            deltas == expected_step
        ).all():
            continue

        inputs.append(
            array[
                window_start:
                target_position
            ]
        )

        targets.append(
            array[target_position]
        )

        target_times.append(
            index[target_position]
        )

    if not inputs:
        raise ValueError(
            "No contiguous causal sequences "
            "were found."
        )

    input_array = np.stack(
        inputs
    ).astype(
        np.float32,
        copy=False,
    )

    target_array = np.stack(
        targets
    ).astype(
        np.float32,
        copy=False,
    )

    return CausalSequenceBatch(
        inputs=input_array,
        targets=target_array,
        target_index=pd.DatetimeIndex(
            target_times
        ),
        sequence_length=sequence_length,
        feature_count=array.shape[1],
    )
