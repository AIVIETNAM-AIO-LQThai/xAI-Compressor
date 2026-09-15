from __future__ import annotations

import numpy as np

from scripts.calibrate_cobra_router_v2 import (
    causal_ewma,
    higher_quantile,
    recent_hit_fraction,
    right_inclusive_percentile,
)


def test_ewma_resets_at_segment_boundary() -> None:
    values = np.array(
        [10.0, 20.0, 100.0, 200.0]
    )
    segments = np.array(
        [0, 0, 1, 1]
    )

    result = causal_ewma(
        values,
        segments,
        alpha=0.2,
    )

    assert np.allclose(
        result,
        [10.0, 12.0, 100.0, 120.0],
    )


def test_higher_quantile() -> None:
    values = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    assert higher_quantile(
        values,
        0.75,
    ) == 4.0


def test_right_inclusive_percentile() -> None:
    reference = np.array(
        [1.0, 2.0, 2.0, 4.0]
    )

    result = right_inclusive_percentile(
        np.array([2.0, 3.0]),
        reference,
    )

    assert np.allclose(
        result,
        [0.75, 0.75],
    )


def test_recent_hit_fraction_requires_full_window() -> None:
    values = np.array(
        [0.0, 2.0, 2.0, 0.0, 2.0]
    )
    segments = np.array(
        [0, 0, 0, 0, 0]
    )

    result = recent_hit_fraction(
        values,
        segments,
        threshold=1.0,
        history_bins=3,
    )

    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert result[2] == 2.0 / 3.0
    assert result[3] == 2.0 / 3.0
    assert result[4] == 2.0 / 3.0


def test_recent_hit_fraction_resets_by_segment() -> None:
    values = np.ones(6)
    segments = np.array(
        [0, 0, 0, 1, 1, 1]
    )

    result = recent_hit_fraction(
        values,
        segments,
        threshold=0.5,
        history_bins=3,
    )

    assert np.isnan(result[0])
    assert np.isnan(result[1])
    assert result[2] == 1.0
    assert np.isnan(result[3])
    assert np.isnan(result[4])
    assert result[5] == 1.0
