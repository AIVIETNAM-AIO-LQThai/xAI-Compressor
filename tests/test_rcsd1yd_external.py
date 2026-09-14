from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ml.data.rcsd1yd import (
    count_gaps,
    frame_bounds,
    load_and_split_rcsd1yd,
)


def _config() -> dict:
    path = Path("configs/rcsd1yd_external_evaluation.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_preregistered_schema_has_25_sensors() -> None:
    config = _config()

    assert len(config["sensor_columns"]) == 25
    assert config["dataset"]["expected_total_columns"] == 26
    assert config["model_features"]["raw_feature_count"] == 25


def test_preregistered_calendar_split_is_fixed() -> None:
    config = _config()
    split = config["chronological_split"]

    assert split["train"]["start"] == "2022-01-01T00:00:00"
    assert split["train"]["end_exclusive"] == "2022-07-01T00:00:00"
    assert split["calibration"]["start"] == "2022-07-01T00:00:00"
    assert split["calibration"]["end_exclusive"] == "2022-10-01T00:00:00"
    assert split["evaluation"]["start"] == "2022-10-01T00:00:00"
    assert split["evaluation"]["end_inclusive"] == "2022-12-31T23:45:00"


def test_frozen_router_cannot_refit_on_rcsd() -> None:
    config = _config()
    router = config["frozen_router"]

    assert router["C"] == 10.0
    assert router["class_weight"] is None
    assert router["probability_threshold"] == 0.30
    assert router["coefficients_refit_allowed"] is False
    assert router["intercept_refit_allowed"] is False
    assert router["router_feature_scaler_refit_allowed"] is False
    assert router["probability_threshold_change_allowed"] is False


def test_gap_counter_uses_strictly_greater_than_30_minutes() -> None:
    frame = pd.DataFrame(
        {"x": [1.0, 2.0, 3.0]},
        index=pd.DatetimeIndex(
            [
                "2022-01-01 00:00:00",
                "2022-01-01 00:30:00",
                "2022-01-01 01:15:00",
            ]
        ),
    )

    assert count_gaps(frame, greater_than_minutes=30) == 1


def test_frame_bounds() -> None:
    frame = pd.DataFrame(
        {"x": [1.0, 2.0]},
        index=pd.DatetimeIndex(
            ["2022-01-01", "2022-01-02"]
        ),
    )

    bounds = frame_bounds(frame)

    assert bounds["start"] == "2022-01-01 00:00:00"
    assert bounds["end"] == "2022-01-02 00:00:00"


def test_nonfinite_sensor_rows_are_dropped_without_imputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    sensors = list(config["sensor_columns"])

    timestamps = pd.DatetimeIndex(
        [
            "2022-01-01 00:00:00",
            "2022-06-30 23:45:00",
            "2022-07-01 00:00:00",
            "2022-09-30 23:45:00",
            "2022-10-01 00:00:00",
            "2022-12-31 23:45:00",
        ]
    )

    raw = pd.DataFrame({"Timestamp": timestamps})
    for sensor in sensors:
        raw[sensor] = np.arange(len(raw), dtype=float)

    raw.loc[1, sensors[0]] = np.nan
    raw.loc[3, sensors[1]] = np.inf

    monkeypatch.setattr(
        pd,
        "read_excel",
        lambda *args, **kwargs: raw.copy(),
    )

    fake = tmp_path / "fake.xlsx"
    fake.write_bytes(b"placeholder")

    result = load_and_split_rcsd1yd(
        fake,
        sensors=sensors,
        expected_total_columns=26,
        expected_start="2022-01-01T00:00:00",
        expected_end="2022-12-31T23:45:00",
        train_cfg=config["chronological_split"]["train"],
        calibration_cfg=config["chronological_split"]["calibration"],
        evaluation_cfg=config["chronological_split"]["evaluation"],
    )

    assert result.train_dropped_rows == 1
    assert result.calibration_dropped_rows == 1
    assert len(result.train) == 1
    assert len(result.calibration) == 1
    assert len(result.evaluation) == 2


def test_sensor_order_mismatch_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    sensors = list(config["sensor_columns"])

    raw = pd.DataFrame(
        {
            "Timestamp": pd.date_range(
                "2022-01-01",
                periods=2,
                freq="15min",
            )
        }
    )
    for sensor in sensors:
        raw[sensor] = 1.0

    columns = ["Timestamp", sensors[1], sensors[0], *sensors[2:]]
    raw = raw.loc[:, columns]

    monkeypatch.setattr(
        pd,
        "read_excel",
        lambda *args, **kwargs: raw.copy(),
    )

    fake = tmp_path / "fake.xlsx"
    fake.write_bytes(b"placeholder")

    with pytest.raises(RuntimeError, match="schema mismatch"):
        load_and_split_rcsd1yd(
            fake,
            sensors=sensors,
            expected_total_columns=26,
            expected_start="2022-01-01T00:00:00",
            expected_end="2022-01-01T00:15:00",
            train_cfg={
                "start": "2022-01-01T00:00:00",
                "end_exclusive": "2022-01-01T00:05:00",
            },
            calibration_cfg={
                "start": "2022-01-01T00:05:00",
                "end_exclusive": "2022-01-01T00:10:00",
            },
            evaluation_cfg={
                "start": "2022-01-01T00:10:00",
                "end_inclusive": "2022-01-01T00:15:00",
            },
        )
