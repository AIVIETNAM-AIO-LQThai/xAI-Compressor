from __future__ import annotations

import numpy as np
import pandas as pd

from ml.data.schema import SENSOR_COLUMNS
from ml.temporal.raw_context import build_raw_context_batch
from scripts.train_raw_context_tcn import resample_raw_context_frame


def test_raw_context_ends_before_target_bin():
    raw_times = pd.date_range(
        "2020-01-01",
        periods=12,
        freq="10s",
    )

    raw_values = np.arange(
        24,
        dtype=np.float32,
    ).reshape(
        12,
        2,
    )

    target_times = pd.DatetimeIndex(
        [
            "2020-01-01 00:00:50",
            "2020-01-01 00:01:20",
        ]
    )

    target_values = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )

    batch = build_raw_context_batch(
        raw_values,
        raw_times,
        target_values,
        target_times,
        context_samples=3,
        raw_step=pd.Timedelta(
            seconds=10
        ),
        forecast_gap=pd.Timedelta(
            seconds=10
        ),
    )

    assert batch.inputs.shape == (
        2,
        3,
        2,
    )

    np.testing.assert_allclose(
        batch.inputs[0],
        raw_values[2:5],
    )

    np.testing.assert_allclose(
        batch.targets[0],
        target_values[0],
    )


def test_raw_context_does_not_cross_gap():
    raw_times = pd.DatetimeIndex(
        [
            "2020-01-01 00:00:00",
            "2020-01-01 00:00:10",
            "2020-01-01 00:00:20",
            "2020-01-01 00:00:40",
            "2020-01-01 00:00:50",
            "2020-01-01 00:01:00",
        ]
    )

    raw_values = np.arange(
        12,
        dtype=np.float32,
    ).reshape(
        6,
        2,
    )

    target_times = pd.DatetimeIndex(
        [
            "2020-01-01 00:00:30",
            "2020-01-01 00:01:00",
        ]
    )

    target_values = np.ones(
        (2, 3),
        dtype=np.float32,
    )

    batch = build_raw_context_batch(
        raw_values,
        raw_times,
        target_values,
        target_times,
        context_samples=3,
        raw_step=pd.Timedelta(
            seconds=10
        ),
        forecast_gap=pd.Timedelta(
            seconds=10
        ),
    )

    assert list(
        batch.target_index
    ) == [
        pd.Timestamp(
            "2020-01-01 00:00:30"
        )
    ]

def test_raw_resampling_is_causal_and_preserves_gap():
    timestamps = pd.DatetimeIndex(
        [
            "2020-01-01 00:00:01",
            "2020-01-01 00:00:09",
            "2020-01-01 00:00:21",
        ]
    )

    frame = pd.DataFrame(
        {
            column: [1.0, 2.0, 3.0]
            for column in SENSOR_COLUMNS
        },
        index=timestamps,
    )

    sampled = resample_raw_context_frame(
        frame,
        step_seconds=10,
    )

    assert list(sampled.index) == [
        pd.Timestamp("2020-01-01 00:00:10"),
        pd.Timestamp("2020-01-01 00:00:30"),
    ]

    for column in SENSOR_COLUMNS:
        assert sampled.iloc[0][column] == 2.0
        assert sampled.iloc[1][column] == 3.0