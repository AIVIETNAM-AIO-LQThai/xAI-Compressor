from __future__ import annotations

import pandas as pd
import pytest

from ml.detection.iitk_external_benchmark import (
    aggregate_iitk_contributions,
    calibration_threshold,
    concentration_rows,
    detection_metrics,
)


def test_calibration_threshold_uses_higher_interpolation():
    scores = pd.Series(
        [
            0.0,
            1.0,
            2.0,
            3.0,
        ]
    )

    threshold = calibration_threshold(
        scores,
        quantile=0.75,
    )

    assert threshold == pytest.approx(
        3.0
    )


def test_iitk_contribution_aggregation_preserves_mass():
    contributions = pd.DataFrame(
        {
            "mean": [
                1.0
            ],
            "std": [
                2.0
            ],
            "skewness": [
                3.0
            ],
            "zero_crossing_rate": [
                4.0
            ],
            "spectral_centroid_norm": [
                5.0
            ],
            "spectral_band_0_energy_fraction": [
                6.0
            ],
        }
    )

    grouped = (
        aggregate_iitk_contributions(
            contributions
        )
    )

    assert grouped.sum(
        axis=1
    ).iloc[0] == pytest.approx(
        contributions.sum(
            axis=1
        ).iloc[0]
    )

    assert grouped.loc[
        0,
        "amplitude",
    ] == pytest.approx(
        3.0
    )

    assert grouped.loc[
        0,
        "distribution_shape",
    ] == pytest.approx(
        3.0
    )

    assert grouped.loc[
        0,
        "temporal_structure",
    ] == pytest.approx(
        4.0
    )

    assert grouped.loc[
        0,
        "spectral_shape",
    ] == pytest.approx(
        5.0
    )

    assert grouped.loc[
        0,
        "spectral_band_energy",
    ] == pytest.approx(
        6.0
    )


def test_concentration_rows_use_five_feature_families():
    grouped = pd.DataFrame(
        {
            "amplitude": [
                8.0
            ],
            "distribution_shape": [
                1.0
            ],
            "temporal_structure": [
                1.0
            ],
            "spectral_shape": [
                0.0
            ],
            "spectral_band_energy": [
                0.0
            ],
        }
    )

    result = concentration_rows(
        grouped
    )

    assert (
        result.loc[
            0,
            "dominant_group",
        ]
        == "amplitude"
    )

    assert result.loc[
        0,
        "top1_concentration",
    ] == pytest.approx(
        0.8
    )

    assert result.loc[
        0,
        "top3_concentration",
    ] == pytest.approx(
        1.0
    )


def test_detection_metrics_reports_per_fault_behavior():
    index = pd.RangeIndex(
        6
    )

    metadata = pd.DataFrame(
        {
            "condition": [
                "healthy",
                "healthy",
                "bearing",
                "bearing",
                "flywheel",
                "flywheel",
            ],
            "reading": [
                181,
                182,
                1,
                2,
                1,
                2,
            ],
            "is_fault": [
                False,
                False,
                True,
                True,
                True,
                True,
            ],
        },
        index=index,
    )

    scores = pd.Series(
        [
            0.1,
            0.2,
            0.8,
            0.9,
            0.3,
            0.4,
        ],
        index=index,
    )

    metrics = detection_metrics(
        scores=scores,
        metadata=metadata,
        threshold=0.5,
    )

    assert metrics[
        "heldout_healthy_false_positive_rate"
    ] == pytest.approx(
        0.0
    )

    assert metrics[
        "per_fault_recall"
    ][
        "bearing"
    ] == pytest.approx(
        1.0
    )

    assert metrics[
        "per_fault_recall"
    ][
        "flywheel"
    ] == pytest.approx(
        0.0
    )

    assert metrics[
        "macro_fault_recall"
    ] == pytest.approx(
        0.5
    )

    assert metrics[
        "per_fault_roc_auc"
    ][
        "bearing"
    ] == pytest.approx(
        1.0
    )

    assert metrics[
        "per_fault_roc_auc"
    ][
        "flywheel"
    ] == pytest.approx(
        1.0
    )
