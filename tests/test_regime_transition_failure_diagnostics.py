from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from ml.detection.alerts import AlertEpisode
from ml.detection.regime_pca import (
    REGIME_HIGH_CURRENT,
    REGIME_LOW_LOW,
)
from ml.detection.regime_transition_failure_diagnostics import (
    persistence_window_frame,
    relevant_episode_flags,
    transition_flags,
)

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_changes_no_parameters() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "regime_transition_failure_audit.yaml"
        ).read_text(encoding="utf-8")
    )
    assert (
        config["study"]["role"]
        == "posthoc_reused_test_failure_diagnostic"
    )
    assert any(
        "No alternative alert rule is simulated"
        in item
        for item in config["guardrails"]
    )


def test_persistence_window_can_span_regimes() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=4,
        freq="5min",
    )
    scores = pd.Series(
        [2.0, 2.0, 0.0, 2.0],
        index=index,
    )
    regimes = pd.Series(
        [
            REGIME_LOW_LOW,
            REGIME_LOW_LOW,
            REGIME_HIGH_CURRENT,
            REGIME_HIGH_CURRENT,
        ],
        index=index,
    )

    frame = persistence_window_frame(
        scores,
        regimes,
        threshold=1.0,
        required_hits=3,
        window_bins=4,
        reset_gap_minutes=10,
    )

    assert bool(frame.iloc[-1]["alert"])
    assert int(
        frame.iloc[-1]["history_unique_regimes"]
    ) == 2
    assert bool(
        frame.iloc[-1]["history_spans_regimes"]
    )


def test_persistence_history_resets_on_time_gap() -> None:
    index = pd.DatetimeIndex(
        [
            "2020-01-01 00:00",
            "2020-01-01 00:05",
            "2020-01-01 01:00",
        ]
    )
    scores = pd.Series(
        [2.0, 2.0, 2.0],
        index=index,
    )
    regimes = pd.Series(
        [REGIME_LOW_LOW] * 3,
        index=index,
    )

    frame = persistence_window_frame(
        scores,
        regimes,
        threshold=1.0,
        required_hits=3,
        window_bins=4,
        reset_gap_minutes=10,
    )

    assert int(frame.iloc[-1]["history_length"]) == 1
    assert not bool(frame.iloc[-1]["alert"])


def test_transition_flags_do_not_cross_gap() -> None:
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


def test_relevant_episode_definition_matches_evaluator() -> None:
    episodes = [
        AlertEpisode(
            start=pd.Timestamp("2020-01-01 12:00"),
            end=pd.Timestamp("2020-01-01 12:10"),
            alert_bins=3,
        ),
        AlertEpisode(
            start=pd.Timestamp("2019-12-30 00:00"),
            end=pd.Timestamp("2019-12-30 00:10"),
            alert_bins=3,
        ),
    ]
    incidents = [
        {
            "id": 1,
            "start": "2020-01-02 00:00",
            "end": "2020-01-02 02:00",
        }
    ]

    relevant = relevant_episode_flags(
        episodes,
        incidents,
        early_warning_hours=24,
    )

    assert relevant == [True, False]
