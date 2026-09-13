from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.temporal.sequences import build_causal_sequences


def test_sequences_predict_next_row():
    values = np.arange(
        20,
        dtype=np.float32,
    ).reshape(10, 2)

    timestamps = pd.date_range(
        "2020-01-01",
        periods=10,
        freq="5min",
    )

    batch = build_causal_sequences(
        values,
        timestamps,
        sequence_length=3,
        expected_step=pd.Timedelta(
            minutes=5
        ),
    )

    assert batch.inputs.shape == (
        7,
        3,
        2,
    )

    assert batch.targets.shape == (
        7,
        2,
    )

    np.testing.assert_allclose(
        batch.inputs[0],
        values[0:3],
    )

    np.testing.assert_allclose(
        batch.targets[0],
        values[3],
    )

    assert (
        batch.target_index[0]
        == timestamps[3]
    )


def test_sequences_do_not_cross_time_gap():
    values = np.arange(
        16,
        dtype=np.float32,
    ).reshape(8, 2)

    timestamps = pd.DatetimeIndex(
        [
            "2020-01-01 00:00",
            "2020-01-01 00:05",
            "2020-01-01 00:10",
            "2020-01-01 00:15",
            "2020-01-01 00:30",
            "2020-01-01 00:35",
            "2020-01-01 00:40",
            "2020-01-01 00:45",
        ]
    )

    batch = build_causal_sequences(
        values,
        timestamps,
        sequence_length=3,
        expected_step=pd.Timedelta(
            minutes=5
        ),
    )

    assert list(
        batch.target_index
    ) == [
        pd.Timestamp(
            "2020-01-01 00:15"
        ),
        pd.Timestamp(
            "2020-01-01 00:45"
        ),
    ]


def test_sequences_reject_unsorted_index():
    values = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    timestamps = pd.DatetimeIndex(
        [
            "2020-01-01 00:00",
            "2020-01-01 00:10",
            "2020-01-01 00:05",
            "2020-01-01 00:15",
        ]
    )

    with pytest.raises(
        ValueError,
        match="sorted",
    ):
        build_causal_sequences(
            values,
            timestamps,
            sequence_length=2,
            expected_step=pd.Timedelta(
                minutes=5
            ),
        )
