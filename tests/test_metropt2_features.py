from __future__ import annotations

import pandas as pd
import pytest

from ml.data.metropt2_features import (
    aggregate_metropt2_split,
    feature_group,
    model_feature_columns,
)


def _frame(
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    rows = len(
        timestamps
    )

    values: dict[
        str,
        object,
    ] = {
        "timestamp": timestamps,
        "tp2": [7.0] * rows,
        "tp3": [7.1] * rows,
        "h1": [6.9] * rows,
        "dv_pressure": [1.0] * rows,
        "reservoirs": [7.0] * rows,
        "oil_temperature": [65.0] * rows,
        "flowmeter": [10.0] * rows,
        "motor_current": [8.0] * rows,
        "comp": [1.0] * rows,
        "dv_electric": [1.0] * rows,
        "towers": [0.0] * rows,
        "mpg": [1.0] * rows,
        "lps": [0.0] * rows,
        "pressure_switch": [1.0] * rows,
        "oil_level": [1.0] * rows,
        "caudal_impulses": [1.0] * rows,
    }

    return pd.DataFrame(
        values
    )


def test_feature_variant_counts():
    columns = []

    for sensor in (
        "tp2",
        "tp3",
        "h1",
        "dv_pressure",
        "reservoirs",
        "oil_temperature",
        "flowmeter",
        "motor_current",
    ):
        for stat in (
            "mean",
            "std",
            "min",
            "max",
            "last",
        ):
            columns.append(
                f"{sensor}__{stat}"
            )

    for sensor in (
        "comp",
        "dv_electric",
        "towers",
        "mpg",
        "lps",
        "pressure_switch",
        "oil_level",
        "caudal_impulses",
    ):
        for stat in (
            "active_ratio",
            "transitions",
            "last",
        ):
            columns.append(
                f"{sensor}__{stat}"
            )

    columns.extend(
        [
            "tp3_minus_reservoirs__mean",
            "tp3_minus_reservoirs__std",
            "tp2_minus_tp3__mean",
            "tp2_minus_tp3__std",
            "sample_count",
            "coverage_ratio",
        ]
    )

    assert len(
        model_feature_columns(
            columns
        )
    ) == 68

    assert len(
        model_feature_columns(
            columns,
            excluded_groups=[
                "flowmeter"
            ],
        )
    ) == 63

    assert len(
        model_feature_columns(
            columns,
            excluded_groups=[
                "flowmeter",
                "caudal_impulses",
            ],
        )
    ) == 60


def test_pressure_relationships_share_group():
    assert (
        feature_group(
            "tp2_minus_tp3__mean"
        )
        == "pressure_relationships"
    )


def test_large_gap_taints_affected_bin():
    timestamps = pd.date_range(
        "2022-06-01 00:00:01",
        periods=299,
        freq="1s",
    )

    frame = _frame(
        timestamps
    )

    # Create a >10 s gap while leaving enough
    # samples for the nominal 5-minute bin.
    frame.loc[
        frame.index >= 20,
        "timestamp",
    ] += pd.Timedelta(
        seconds=11
    )

    result = (
        aggregate_metropt2_split(
            frame,
            start=(
                "2022-06-01 "
                "00:00:00"
            ),
            end=(
                "2022-06-01 "
                "00:05:30"
            ),
            bin_minutes=5,
            expected_samples_per_bin=300,
            minimum_coverage=0.80,
            max_gap_seconds=10.0,
        )
    )

    assert (
        result.gap_tainted_bins
        >= 1
    )

    # The 00:05 endpoint contains the gap and
    # therefore must not survive.
    assert pd.Timestamp(
        "2022-06-01 00:05:00"
    ) not in result.features.index


def test_split_start_is_aggregated_independently():
    timestamps = pd.date_range(
        "2022-06-01 00:00:00",
        periods=601,
        freq="1s",
    )

    frame = _frame(
        timestamps
    )

    result = (
        aggregate_metropt2_split(
            frame,
            start=(
                "2022-06-01 "
                "00:02:00"
            ),
            end=(
                "2022-06-01 "
                "00:10:00"
            ),
            bin_minutes=5,
            expected_samples_per_bin=300,
            minimum_coverage=0.80,
            max_gap_seconds=10.0,
        )
    )

    # The 00:05 bin has only 181 samples from
    # this split, so it cannot import pre-split
    # samples to pass coverage.
    assert pd.Timestamp(
        "2022-06-01 00:05:00"
    ) not in result.features.index

    assert pd.Timestamp(
        "2022-06-01 00:10:00"
    ) in result.features.index


@pytest.mark.parametrize(
    "coverage",
    [
        0.0,
        1.1,
    ],
)
def test_invalid_coverage_rejected(
    coverage: float,
):
    frame = _frame(
        pd.date_range(
            "2022-06-01",
            periods=10,
            freq="1s",
        )
    )

    with pytest.raises(
        ValueError,
        match="minimum_coverage",
    ):
        aggregate_metropt2_split(
            frame,
            start=(
                "2022-06-01 "
                "00:00:00"
            ),
            end=(
                "2022-06-01 "
                "00:01:00"
            ),
            bin_minutes=5,
            expected_samples_per_bin=300,
            minimum_coverage=coverage,
            max_gap_seconds=10.0,
        )
