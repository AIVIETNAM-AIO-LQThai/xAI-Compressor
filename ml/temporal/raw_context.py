from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RawContextBatch:
    inputs: np.ndarray
    targets: np.ndarray
    target_index: pd.DatetimeIndex
    context_samples: int
    input_feature_count: int
    target_feature_count: int

    def __post_init__(self) -> None:
        if self.inputs.ndim != 3:
            raise ValueError(
                "inputs must have shape "
                "(samples, context, features)."
            )

        if self.targets.ndim != 2:
            raise ValueError(
                "targets must have shape "
                "(samples, target_features)."
            )

        if (
            self.inputs.shape[0]
            != self.targets.shape[0]
            or self.inputs.shape[0]
            != len(self.target_index)
        ):
            raise ValueError(
                "Inputs, targets, and timestamps "
                "must have equal sample counts."
            )

        if (
            self.inputs.shape[1]
            != self.context_samples
        ):
            raise ValueError(
                "Input context dimension does not "
                "match context_samples."
            )

        if (
            self.inputs.shape[2]
            != self.input_feature_count
        ):
            raise ValueError(
                "Input feature dimension mismatch."
            )

        if (
            self.targets.shape[1]
            != self.target_feature_count
        ):
            raise ValueError(
                "Target feature dimension mismatch."
            )


def build_raw_context_batch(
    raw_values: np.ndarray,
    raw_timestamps: pd.DatetimeIndex,
    target_values: np.ndarray,
    target_timestamps: pd.DatetimeIndex,
    *,
    context_samples: int,
    raw_step: pd.Timedelta,
    forecast_gap: pd.Timedelta,
) -> RawContextBatch:
    raw_array = np.asarray(
        raw_values,
        dtype=np.float32,
    )

    target_array = np.asarray(
        target_values,
        dtype=np.float32,
    )

    raw_index = pd.DatetimeIndex(
        raw_timestamps
    )

    target_index = pd.DatetimeIndex(
        target_timestamps
    )

    if raw_array.ndim != 2:
        raise ValueError(
            "raw_values must be a 2D array."
        )

    if target_array.ndim != 2:
        raise ValueError(
            "target_values must be a 2D array."
        )

    if len(raw_array) != len(raw_index):
        raise ValueError(
            "raw values and timestamps must "
            "have equal length."
        )

    if len(target_array) != len(
        target_index
    ):
        raise ValueError(
            "target values and timestamps must "
            "have equal length."
        )

    if context_samples <= 0:
        raise ValueError(
            "context_samples must be positive."
        )

    if raw_step <= pd.Timedelta(0):
        raise ValueError(
            "raw_step must be positive."
        )

    if forecast_gap <= pd.Timedelta(0):
        raise ValueError(
            "forecast_gap must be positive."
        )

    if raw_index.has_duplicates:
        raise ValueError(
            "raw timestamps must not contain "
            "duplicates."
        )

    if not raw_index.is_monotonic_increasing:
        raise ValueError(
            "raw timestamps must be sorted."
        )

    if target_index.has_duplicates:
        raise ValueError(
            "target timestamps must not contain "
            "duplicates."
        )

    if not target_index.is_monotonic_increasing:
        raise ValueError(
            "target timestamps must be sorted."
        )

    if not np.isfinite(raw_array).all():
        raise ValueError(
            "raw_values contain non-finite "
            "entries."
        )

    if not np.isfinite(
        target_array
    ).all():
        raise ValueError(
            "target_values contain non-finite "
            "entries."
        )

    position_by_time = {
        timestamp: position
        for position, timestamp
        in enumerate(raw_index)
    }

    if len(raw_index) > 1:
        breaks = np.concatenate(
            [
                np.array(
                    [True],
                    dtype=bool,
                ),
                (
                    raw_index[1:]
                    - raw_index[:-1]
                    != raw_step
                ),
            ]
        )
    else:
        breaks = np.array(
            [True],
            dtype=bool,
        )

    run_ids = np.cumsum(
        breaks
    )

    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    kept_times: list[pd.Timestamp] = []

    context_span = (
        raw_step
        * (context_samples - 1)
    )

    for target_position, target_time in enumerate(
        target_index
    ):
        context_end = (
            target_time
            - forecast_gap
        )

        context_start = (
            context_end
            - context_span
        )

        start_position = (
            position_by_time.get(
                context_start
            )
        )

        end_position = (
            position_by_time.get(
                context_end
            )
        )

        if (
            start_position is None
            or end_position is None
        ):
            continue

        if (
            end_position
            - start_position
            + 1
            != context_samples
        ):
            continue

        if (
            run_ids[start_position]
            != run_ids[end_position]
        ):
            continue

        inputs.append(
            raw_array[
                start_position:
                end_position + 1
            ]
        )

        targets.append(
            target_array[
                target_position
            ]
        )

        kept_times.append(
            target_time
        )

    if not inputs:
        raise ValueError(
            "No contiguous raw-context "
            "examples were found."
        )

    return RawContextBatch(
        inputs=np.stack(
            inputs
        ).astype(
            np.float32,
            copy=False,
        ),
        targets=np.stack(
            targets
        ).astype(
            np.float32,
            copy=False,
        ),
        target_index=pd.DatetimeIndex(
            kept_times
        ),
        context_samples=context_samples,
        input_feature_count=(
            raw_array.shape[1]
        ),
        target_feature_count=(
            target_array.shape[1]
        ),
    )
