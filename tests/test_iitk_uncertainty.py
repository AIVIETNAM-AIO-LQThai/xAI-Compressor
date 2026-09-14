from __future__ import annotations

import numpy as np
import pandas as pd

from ml.detection.iitk_uncertainty import (
    bootstrap_calibration_thresholds,
    leave_one_out_thresholds,
    quantile_order_information,
    threshold_operating_distributions,
    wilson_interval,
)


def test_quantile_order_for_sixty_calibration_points():
    info = quantile_order_information(
        60,
        quantile=0.95,
    )

    assert (
        info[
            "higher_index_zero_based"
        ]
        == 57
    )

    assert (
        info[
            "higher_rank_one_based"
        ]
        == 58
    )

    assert (
        info[
            "upper_order_statistics_including_threshold"
        ]
        == 3
    )


def test_wilson_interval_contains_observed_rate():
    interval = wilson_interval(
        1,
        45,
        confidence_level=0.95,
    )

    assert (
        interval[
            "lower"
        ]
        <= 1.0
        / 45.0
        <= interval[
            "upper"
        ]
    )

    assert (
        interval[
            "lower"
        ]
        >= 0.0
    )

    assert (
        interval[
            "upper"
        ]
        <= 1.0
    )


def test_leave_one_out_thresholds_are_finite():
    scores = pd.Series(
        np.arange(
            1.0,
            11.0,
        )
    )

    thresholds = (
        leave_one_out_thresholds(
            scores,
            quantile=0.8,
        )
    )

    assert thresholds.shape == (
        10,
    )

    assert np.isfinite(
        thresholds
    ).all()


def test_bootstrap_thresholds_are_reproducible():
    scores = pd.Series(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ]
    )

    first = bootstrap_calibration_thresholds(
        scores,
        quantile=0.8,
        iterations=20,
        seed=123,
        block_length=None,
    )

    second = bootstrap_calibration_thresholds(
        scores,
        quantile=0.8,
        iterations=20,
        seed=123,
        block_length=None,
    )

    assert first.tolist() == (
        second.tolist()
    )


def test_moving_block_thresholds_are_reproducible():
    scores = pd.Series(
        np.arange(
            1.0,
            11.0,
        )
    )

    first = bootstrap_calibration_thresholds(
        scores,
        quantile=0.8,
        iterations=20,
        seed=456,
        block_length=5,
    )

    second = bootstrap_calibration_thresholds(
        scores,
        quantile=0.8,
        iterations=20,
        seed=456,
        block_length=5,
    )

    assert first.tolist() == (
        second.tolist()
    )


def test_threshold_operating_distributions_match_expected():
    scores = pd.Series(
        [
            0.1,
            0.2,
            0.6,
            0.8,
        ]
    )

    metadata = pd.DataFrame(
        {
            "condition": [
                "healthy",
                "healthy",
                "bearing",
                "bearing",
            ],
            "is_fault": [
                False,
                False,
                True,
                True,
            ],
        }
    )

    distributions = (
        threshold_operating_distributions(
            thresholds=np.asarray(
                [
                    0.5,
                    0.7,
                ]
            ),
            test_scores=scores,
            metadata=metadata,
        )
    )

    assert (
        distributions[
            "heldout_healthy_false_positive_rate"
        ].tolist()
        == [
            0.0,
            0.0,
        ]
    )

    assert (
        distributions[
            "overall_fault_recall"
        ].tolist()
        == [
            1.0,
            0.5,
        ]
    )

    assert (
        distributions[
            "per_fault_recall::bearing"
        ].tolist()
        == [
            1.0,
            0.5,
        ]
    )
