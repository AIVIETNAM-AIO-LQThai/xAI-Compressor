from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml.detection.alerts import causal_ewma
from ml.detection.regime_pca import (
    REGIME_HIGH_CURRENT,
    REGIME_LOW_LOW,
)
from ml.detection.regime_smoothing_diagnostics import (
    transition_carryover_table,
    transition_flags,
    transition_reset_ewma,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_has_no_test_or_incidents() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "regime_ewma_carryover_audit.yaml"
        ).read_text(encoding="utf-8")
    )
    dataset = config["dataset"]
    assert "test_file" not in dataset
    assert "incidents_config" not in dataset
    assert "reported_incidents" not in dataset


def test_transition_reset_ewma_matches_baseline_without_switch() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=4,
        freq="5min",
    )
    scores = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=index,
    )
    regimes = pd.Series(
        [REGIME_LOW_LOW] * 4,
        index=index,
    )

    expected = causal_ewma(
        scores,
        alpha=0.2,
        reset_gap_minutes=10,
    )
    observed = transition_reset_ewma(
        scores,
        regimes,
        alpha=0.2,
        reset_gap_minutes=10,
    )

    pd.testing.assert_series_equal(
        observed.rename("smoothed_score"),
        expected,
    )


def test_transition_reset_uses_raw_score_on_switch() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=3,
        freq="5min",
    )
    scores = pd.Series(
        [10.0, 10.0, 1.0],
        index=index,
    )
    regimes = pd.Series(
        [
            REGIME_LOW_LOW,
            REGIME_LOW_LOW,
            REGIME_HIGH_CURRENT,
        ],
        index=index,
    )

    baseline = causal_ewma(
        scores,
        alpha=0.2,
        reset_gap_minutes=10,
    )
    reset = transition_reset_ewma(
        scores,
        regimes,
        alpha=0.2,
        reset_gap_minutes=10,
    )

    assert np.isclose(baseline.iloc[2], 8.2)
    assert np.isclose(reset.iloc[2], 1.0)


def test_transition_flags_do_not_cross_time_gap() -> None:
    index = pd.DatetimeIndex(
        [
            "2020-01-01 00:00",
            "2020-01-01 00:05",
            "2020-01-01 01:00",
        ]
    )
    regimes = pd.Series(
        [
            REGIME_LOW_LOW,
            REGIME_HIGH_CURRENT,
            REGIME_LOW_LOW,
        ],
        index=index,
    )

    flags = transition_flags(
        regimes,
        reset_gap_minutes=10,
    )

    assert flags.tolist() == [False, True, False]


def test_carryover_table_contains_only_transition_rows() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=3,
        freq="5min",
    )
    raw = pd.Series([2.0, 2.0, 1.0], index=index)
    regimes = pd.Series(
        [
            REGIME_LOW_LOW,
            REGIME_LOW_LOW,
            REGIME_HIGH_CURRENT,
        ],
        index=index,
    )
    baseline = causal_ewma(
        raw,
        alpha=0.2,
        reset_gap_minutes=10,
    )
    reset = transition_reset_ewma(
        raw,
        regimes,
        alpha=0.2,
        reset_gap_minutes=10,
    )

    table = transition_carryover_table(
        raw,
        baseline,
        reset,
        regimes,
        reset_gap_minutes=10,
    )

    assert len(table) == 1
    assert table.iloc[0]["previous_regime"] == REGIME_LOW_LOW
    assert table.iloc[0]["current_regime"] == REGIME_HIGH_CURRENT
