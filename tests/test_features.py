from typing import cast

import numpy as np
import pandas as pd

from ml.data.features import (
    aggregate_causal_bins,
    filter_valid_bins,
    model_feature_columns,
    select_time_range,
)
from ml.data.schema import (
    ANALOG_COLUMNS,
    DIGITAL_COLUMNS,
)


def make_frame(
    start: str,
    periods: int,
) -> pd.DataFrame:
    timestamps = pd.date_range(
        start=start,
        periods=periods,
        freq="10s",
    )

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
        }
    )

    for index, column in enumerate(
        ANALOG_COLUMNS
    ):
        frame[column] = (
            np.arange(periods)
            + index
        ).astype(float)

    for index, column in enumerate(
        DIGITAL_COLUMNS
    ):
        frame[column] = (
            np.arange(periods)
            + index
        ) % 2

    return frame


def test_model_feature_count():
    frame = make_frame(
        "2020-01-01 00:00:10",
        30,
    )

    features = aggregate_causal_bins(
        frame,
        bin_minutes=5,
        expected_samples=30,
    )

    model_columns = model_feature_columns(
        features.columns
    )

    assert len(model_columns) == 63

    assert "sample_count" not in model_columns
    assert "coverage_ratio" not in model_columns


def test_complete_five_minute_bin_has_full_coverage():
    frame = make_frame(
        "2020-01-01 00:00:10",
        30,
    )

    features = aggregate_causal_bins(
        frame,
        bin_minutes=5,
        expected_samples=30,
    )

    row = features.iloc[0]

    assert row["sample_count"] == 30
    assert row["coverage_ratio"] == 1.0


def test_low_coverage_bin_is_removed():
    frame = make_frame(
        "2020-01-01 00:00:10",
        10,
    )

    features = aggregate_causal_bins(
        frame,
        bin_minutes=5,
        expected_samples=30,
    )

    clean = filter_valid_bins(
        features,
        minimum_coverage=0.8,
    )

    assert clean.empty


def test_future_samples_do_not_change_previous_bin():
    frame = make_frame(
        "2020-01-01 00:00:10",
        60,
    )

    original = aggregate_causal_bins(
        frame,
        bin_minutes=5,
        expected_samples=30,
    )

    modified = frame.copy()

    future_mask = (
        modified["timestamp"]
        > pd.Timestamp(
            "2020-01-01 00:05:00"
        )
    )

    modified.loc[
        future_mask,
        "motor_current",
    ] = 99999.0

    changed = aggregate_causal_bins(
        modified,
        bin_minutes=5,
        expected_samples=30,
    )

    timestamp = pd.Timestamp("2020-01-01 00:05:00")

    original_row = cast(
        pd.Series,
        original.loc[timestamp],
    )

    changed_row = cast(
        pd.Series,
        changed.loc[timestamp],
    )

    pd.testing.assert_series_equal(
        original_row, changed_row,
    )


def test_time_range_is_inclusive_and_bounded():
    index = pd.date_range(
        "2020-01-01",
        periods=10,
        freq="5min",
    )

    frame = pd.DataFrame(
        {"value": range(10)},
        index=index,
    )

    selected = select_time_range(
        frame,
        start="2020-01-01 00:10:00",
        end="2020-01-01 00:25:00",
    )

    assert selected.index.min() == pd.Timestamp(
        "2020-01-01 00:10:00"
    )

    assert selected.index.max() == pd.Timestamp(
        "2020-01-01 00:25:00"
    )