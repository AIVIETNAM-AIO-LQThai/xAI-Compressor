from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.data.cobra import (
    aggregate_five_minute,
    assert_train_only_day,
    assign_contiguous_segments,
    day_from_archive_name,
    parse_source_timestamp,
    semantic_representation_sha256,
)


def test_archive_day_parsing() -> None:
    assert (
        day_from_archive_name(
            "CoBra20240603.zip"
        )
        == "2024-06-03"
    )


def test_train_only_partition_guard() -> None:
    train = {"2024-06-03"}
    calibration = {"2025-01-17"}
    evaluation = {"2025-02-17"}

    assert_train_only_day(
        "2024-06-03",
        train_days=train,
        calibration_days=calibration,
        evaluation_days=evaluation,
    )

    with pytest.raises(
        RuntimeError,
        match="CALIBRATION",
    ):
        assert_train_only_day(
            "2025-01-17",
            train_days=train,
            calibration_days=calibration,
            evaluation_days=evaluation,
        )

    with pytest.raises(
        RuntimeError,
        match="EVALUATION",
    ):
        assert_train_only_day(
            "2025-02-17",
            train_days=train,
            calibration_days=calibration,
            evaluation_days=evaluation,
        )


def test_timestamp_parser_preserves_minute_resolution() -> None:
    value = parse_source_timestamp(
        "27.06.2024 14:33"
    )

    assert value == pd.Timestamp(
        "2024-06-27 14:33:00"
    )
    assert value.microsecond == 0


def test_duplicate_source_timestamps_aggregate_without_fake_seconds() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                "27.06.2024 14:30",
                "27.06.2024 14:30",
                "27.06.2024 14:31",
                "27.06.2024 14:35",
            ],
            "sensor_a": [
                "1",
                "3",
                "5",
                "7",
            ],
            "sensor_b": [
                "2",
                "4",
                "6",
                "8",
            ],
        }
    )

    result = aggregate_five_minute(
        frame,
        timestamp_column="timestamp",
        feature_names=[
            "sensor_a",
            "sensor_b",
        ],
    )

    assert result.source_rows == 4
    assert result.produced_bins == 2
    assert result.valid_bins == 2
    assert result.invalid_bins == 0

    first = result.frame.iloc[0]

    assert first["timestamp"] == pd.Timestamp(
        "2024-06-27 14:30:00"
    )

    assert first["sensor_a"] == 3.0
    assert first["sensor_b"] == 4.0


def test_bin_invalid_when_one_feature_has_no_finite_sample() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2024-06-03 10:00:00",
                "2024-06-03 10:01:00",
                "2024-06-03 10:05:00",
            ],
            "sensor_a": [
                "1",
                "3",
                "5",
            ],
            "sensor_b": [
                "",
                "bad",
                "7",
            ],
        }
    )

    result = aggregate_five_minute(
        frame,
        timestamp_column="timestamp",
        feature_names=[
            "sensor_a",
            "sensor_b",
        ],
    )

    assert result.produced_bins == 2
    assert result.valid_bins == 1
    assert result.invalid_bins == 1

    assert result.frame.iloc[
        0
    ]["timestamp"] == pd.Timestamp(
        "2024-06-03 10:05:00"
    )


def test_non_five_minute_gap_breaks_segment() -> None:
    timestamps = pd.Series(
        pd.to_datetime(
            [
                "2024-06-03 10:00:00",
                "2024-06-03 10:05:00",
                "2024-06-03 10:15:00",
                "2024-06-03 10:20:00",
            ]
        )
    )

    segments = assign_contiguous_segments(
        timestamps,
        step_minutes=5,
    )

    assert segments.tolist() == [
        0,
        0,
        1,
        1,
    ]


def test_infinite_values_do_not_make_valid_bin() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2024-06-03 10:00:00",
                "2024-06-03 10:05:00",
            ],
            "sensor_a": [
                "inf",
                "2",
            ],
            "sensor_b": [
                "1",
                "3",
            ],
        }
    )

    result = aggregate_five_minute(
        frame,
        timestamp_column="timestamp",
        feature_names=[
            "sensor_a",
            "sensor_b",
        ],
    )

    assert (
        result.infinite_numeric_cells
        == 1
    )

    assert result.valid_bins == 1
    assert result.invalid_bins == 1


def test_semantic_hash_is_stable() -> None:
    frame = pd.DataFrame(
        {
            "day": [
                "2024-06-03",
                "2024-06-03",
            ],
            "timestamp": pd.to_datetime(
                [
                    "2024-06-03 10:00:00",
                    "2024-06-03 10:05:00",
                ]
            ),
            "segment_id": [
                0,
                0,
            ],
            "sensor_a": [
                1.0,
                2.0,
            ],
            "sensor_b": [
                3.0,
                4.0,
            ],
        }
    )

    first = (
        semantic_representation_sha256(
            frame,
            feature_names=[
                "sensor_a",
                "sensor_b",
            ],
        )
    )

    second = (
        semantic_representation_sha256(
            frame.copy(),
            feature_names=[
                "sensor_a",
                "sensor_b",
            ],
        )
    )

    assert first == second
    assert len(first) == 64


def test_valid_output_contains_only_finite_values() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [
                "2024-06-03 10:00:00",
            ],
            "sensor_a": [
                "1.5",
            ],
            "sensor_b": [
                "2.5",
            ],
        }
    )

    result = aggregate_five_minute(
        frame,
        timestamp_column="timestamp",
        feature_names=[
            "sensor_a",
            "sensor_b",
        ],
    )

    values = result.frame[
        [
            "sensor_a",
            "sensor_b",
        ]
    ].to_numpy()

    assert np.isfinite(values).all()
