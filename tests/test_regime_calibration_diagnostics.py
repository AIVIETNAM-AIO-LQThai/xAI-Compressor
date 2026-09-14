from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml.detection.regime_calibration_diagnostics import (
    bootstrap_threshold_composition,
    gap_aware_autocorrelation,
    near_transition_flags,
    occupancy_comparison,
    pooled_tail_composition,
    regime_run_summary,
    transition_flags,
)
from ml.detection.regime_pca import (
    REGIME_HIGH_BOTH,
    REGIME_HIGH_CURRENT,
    REGIME_HIGH_PRESSURE,
    REGIME_LOW_LOW,
)

ROOT = Path(__file__).resolve().parents[1]


def _index(n: int) -> pd.DatetimeIndex:
    return pd.date_range(
        "2020-01-01",
        periods=n,
        freq="5min",
    )


def test_audit_config_contains_no_test_or_incident_input() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "regime_calibration_audit.yaml"
        ).read_text(encoding="utf-8")
    )
    dataset = config["dataset"]

    assert "test_file" not in dataset
    assert "incidents_config" not in dataset
    assert "reported_incidents" not in dataset


def test_occupancy_counts_sum_to_population() -> None:
    train = pd.Series(
        [
            REGIME_LOW_LOW,
            REGIME_LOW_LOW,
            REGIME_HIGH_CURRENT,
            REGIME_HIGH_PRESSURE,
            REGIME_HIGH_BOTH,
        ],
        index=_index(5),
    )
    calibration = pd.Series(
        [
            REGIME_LOW_LOW,
            REGIME_HIGH_CURRENT,
            REGIME_HIGH_BOTH,
            REGIME_HIGH_BOTH,
        ],
        index=_index(4),
    )

    result = occupancy_comparison(train, calibration)

    assert sum(
        item["train_count"] for item in result.values()
    ) == 5
    assert sum(
        item["calibration_count"] for item in result.values()
    ) == 4


def test_tail_composition_fractions_sum_to_one() -> None:
    scores = pd.Series(
        [0.0, 1.0, 2.0, 3.0],
        index=_index(4),
    )
    regimes = pd.Series(
        [
            REGIME_LOW_LOW,
            REGIME_LOW_LOW,
            REGIME_HIGH_CURRENT,
            REGIME_HIGH_CURRENT,
        ],
        index=_index(4),
    )

    result = pooled_tail_composition(
        scores,
        regimes,
        quantiles=[0.50],
    )["0.5"]

    total = sum(
        item["tail_fraction"]
        for item in result["composition"].values()
    )
    assert np.isclose(total, 1.0)


def test_transition_detection_does_not_cross_large_gap() -> None:
    index = pd.DatetimeIndex(
        [
            "2020-01-01 00:00",
            "2020-01-01 00:05",
            "2020-01-01 01:00",
            "2020-01-01 01:05",
        ]
    )
    regimes = pd.Series(
        [
            REGIME_LOW_LOW,
            REGIME_HIGH_CURRENT,
            REGIME_HIGH_PRESSURE,
            REGIME_HIGH_BOTH,
        ],
        index=index,
    )

    flags = transition_flags(
        regimes,
        reset_gap_minutes=10,
    )

    assert flags.tolist() == [False, True, False, True]


def test_near_transition_flags_clip_to_segment() -> None:
    index = pd.DatetimeIndex(
        [
            "2020-01-01 00:00",
            "2020-01-01 00:05",
            "2020-01-01 01:00",
            "2020-01-01 01:05",
        ]
    )
    transitions = pd.Series(
        [False, True, False, False],
        index=index,
    )

    near = near_transition_flags(
        transitions,
        radius=2,
        reset_gap_minutes=10,
    )

    assert near.tolist() == [True, True, False, False]


def test_regime_run_summary_splits_label_changes() -> None:
    regimes = pd.Series(
        [
            REGIME_LOW_LOW,
            REGIME_LOW_LOW,
            REGIME_HIGH_CURRENT,
            REGIME_HIGH_CURRENT,
            REGIME_LOW_LOW,
        ],
        index=_index(5),
    )

    result = regime_run_summary(
        regimes,
        reset_gap_minutes=10,
    )

    assert result[REGIME_LOW_LOW]["number_of_runs"] == 2
    assert result[REGIME_LOW_LOW]["max_run_length"] == 2
    assert result[REGIME_HIGH_CURRENT]["number_of_runs"] == 1
    assert result[REGIME_HIGH_CURRENT]["max_run_length"] == 2


def test_gap_aware_autocorrelation_counts_pairs() -> None:
    series = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=_index(4),
    )

    result = gap_aware_autocorrelation(
        series,
        lags=[1],
        reset_gap_minutes=10,
    )

    assert result["pooled"]["1"]["pair_count"] == 3
    assert np.isclose(
        result["pooled"]["1"]["correlation"],
        1.0,
    )


def test_bootstrap_is_deterministic_and_preserves_pairing() -> None:
    scores = pd.Series(
        np.linspace(0.0, 1.0, 20),
        index=_index(20),
    )
    regimes = pd.Series(
        [
            REGIME_LOW_LOW
            if i < 10
            else REGIME_HIGH_CURRENT
            for i in range(20)
        ],
        index=_index(20),
    )

    first_summary, first_table = bootstrap_threshold_composition(
        scores,
        regimes,
        threshold_quantile=0.90,
        bootstrap_replicates=25,
        block_lengths=[1, 3],
        seed=7,
        confidence=0.95,
    )
    second_summary, second_table = bootstrap_threshold_composition(
        scores,
        regimes,
        threshold_quantile=0.90,
        bootstrap_replicates=25,
        block_lengths=[1, 3],
        seed=7,
        confidence=0.95,
    )

    assert first_summary == second_summary
    pd.testing.assert_frame_equal(first_table, second_table)

    fraction_columns = [
        column
        for column in first_table.columns
        if column.startswith("fraction__")
    ]
    assert np.allclose(
        first_table[fraction_columns].sum(axis=1),
        1.0,
    )
