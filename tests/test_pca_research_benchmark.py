from __future__ import annotations

import pandas as pd
import pytest

from ml.detection.pca_research_benchmark import (
    AlertProtocol,
    calibrate_alert_pipeline,
)


def _protocol() -> AlertProtocol:
    return AlertProtocol(
        ewma_alpha=1.0,
        threshold_quantile=0.75,
        persistence_hits=1,
        persistence_window=1,
        reset_gap_minutes=10,
        merge_minutes=30,
        early_warning_hours=24,
        late_tolerance_hours=2,
        bin_minutes=5,
    )


def test_threshold_uses_calibration_only():
    calibration = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.date_range(
            "2022-05-21",
            periods=4,
            freq="5min",
        ),
    )

    test = pd.Series(
        [
            1000.0,
            2000.0,
        ],
        index=pd.date_range(
            "2022-06-01",
            periods=2,
            freq="5min",
        ),
    )

    result = calibrate_alert_pipeline(
        calibration_scores=calibration,
        test_scores=test,
        protocol=_protocol(),
    )

    # pandas quantile with interpolation="higher"
    # gives 4.0 at q=0.75 for four ordered values.
    assert (
        result.threshold
        == pytest.approx(4.0)
    )


def test_alert_pipeline_resets_across_gap():
    calibration = pd.Series(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ],
        index=pd.date_range(
            "2022-05-21",
            periods=4,
            freq="5min",
        ),
    )

    test = pd.Series(
        [
            10.0,
            10.0,
            10.0,
        ],
        index=pd.to_datetime(
            [
                "2022-06-01 00:00:00",
                "2022-06-01 00:05:00",
                "2022-06-01 01:00:00",
            ]
        ),
    )

    protocol = AlertProtocol(
        ewma_alpha=1.0,
        threshold_quantile=0.75,
        persistence_hits=2,
        persistence_window=2,
        reset_gap_minutes=10,
        merge_minutes=30,
        early_warning_hours=24,
        late_tolerance_hours=2,
        bin_minutes=5,
    )

    result = calibrate_alert_pipeline(
        calibration_scores=calibration,
        test_scores=test,
        protocol=protocol,
    )

    assert bool(
        result.alerts.iloc[0]
    ) is False

    assert bool(
        result.alerts.iloc[1]
    ) is True

    # A 55-minute gap clears persistence history.
    assert bool(
        result.alerts.iloc[2]
    ) is False


def test_threshold_interpolation_is_higher():
    calibration = pd.Series(
        [
            1.0,
            10.0,
            20.0,
        ],
        index=pd.date_range(
            "2022-05-21",
            periods=3,
            freq="5min",
        ),
    )

    test = pd.Series(
        [0.0],
        index=pd.to_datetime(
            [
                "2022-06-01 00:00:00",
            ]
        ),
    )

    protocol = AlertProtocol(
        ewma_alpha=1.0,
        threshold_quantile=0.5,
        persistence_hits=1,
        persistence_window=1,
        reset_gap_minutes=10,
        merge_minutes=30,
        early_warning_hours=24,
        late_tolerance_hours=2,
        bin_minutes=5,
    )

    result = calibrate_alert_pipeline(
        calibration_scores=calibration,
        test_scores=test,
        protocol=protocol,
    )

    assert (
        result.threshold
        == pytest.approx(10.0)
    )
