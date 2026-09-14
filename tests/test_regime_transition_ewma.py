from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml.detection.pca_research_benchmark import (
    AlertProtocol,
)
from ml.detection.regime_pca import (
    REGIME_HIGH_CURRENT,
    REGIME_LOW_LOW,
)
from ml.detection.regime_transition_ewma import (
    calibrate_transition_reset_pipeline,
    transition_reset_ewma,
    transition_reset_ewma_frame,
)

ROOT = Path(__file__).resolve().parents[1]


def _protocol() -> AlertProtocol:
    return AlertProtocol(
        ewma_alpha=0.2,
        threshold_quantile=0.995,
        persistence_hits=3,
        persistence_window=4,
        reset_gap_minutes=10,
        merge_minutes=30,
        early_warning_hours=24,
        late_tolerance_hours=2,
        bin_minutes=5,
    )


def test_config_freezes_only_one_candidate_change() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "regime_transition_ewma_benchmark.yaml"
        ).read_text(encoding="utf-8")
    )

    assert (
        config["alerting"]["candidate_extra_reset"]
        == "regime_change"
    )
    assert (
        config["alerting"][
            "persistence_resets_on_regime_change"
        ]
        is False
    )


def test_transition_reset_ewma_resets_on_switch() -> None:
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

    smoothed = transition_reset_ewma(
        scores,
        regimes,
        alpha=0.2,
        reset_gap_minutes=10,
    )

    assert np.isclose(smoothed.iloc[0], 10.0)
    assert np.isclose(smoothed.iloc[1], 10.0)
    assert np.isclose(smoothed.iloc[2], 1.0)


def test_contribution_smoothing_is_additive() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=4,
        freq="5min",
    )
    contributions = pd.DataFrame(
        {
            "a": [1.0, 2.0, 4.0, 1.0],
            "b": [2.0, 1.0, 2.0, 3.0],
        },
        index=index,
    )
    scores = contributions.sum(axis=1)
    regimes = pd.Series(
        [
            REGIME_LOW_LOW,
            REGIME_LOW_LOW,
            REGIME_HIGH_CURRENT,
            REGIME_HIGH_CURRENT,
        ],
        index=index,
    )

    smoothed_score = transition_reset_ewma(
        scores,
        regimes,
        alpha=0.2,
        reset_gap_minutes=10,
    )
    smoothed_contributions = transition_reset_ewma_frame(
        contributions,
        regimes,
        alpha=0.2,
        reset_gap_minutes=10,
    )

    assert np.allclose(
        smoothed_contributions.sum(axis=1),
        smoothed_score,
    )


def test_calibration_and_test_states_are_independent() -> None:
    calibration_index = pd.date_range(
        "2020-01-01",
        periods=4,
        freq="5min",
    )
    test_index = pd.date_range(
        "2020-02-01",
        periods=4,
        freq="5min",
    )

    calibration_scores = pd.Series(
        [10.0, 10.0, 10.0, 10.0],
        index=calibration_index,
    )
    test_scores = pd.Series(
        [1.0, 1.0, 1.0, 1.0],
        index=test_index,
    )
    calibration_regimes = pd.Series(
        [REGIME_LOW_LOW] * 4,
        index=calibration_index,
    )
    test_regimes = pd.Series(
        [REGIME_LOW_LOW] * 4,
        index=test_index,
    )

    result = calibrate_transition_reset_pipeline(
        calibration_scores=calibration_scores,
        calibration_regimes=calibration_regimes,
        test_scores=test_scores,
        test_regimes=test_regimes,
        protocol=_protocol(),
    )

    assert np.isclose(
        result.test_smoothed.iloc[0],
        1.0,
    )
