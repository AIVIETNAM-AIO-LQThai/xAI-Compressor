from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml.detection.operating_cycle_support_diagnostics import (
    joint_pattern_summary,
    make_support_flags,
    run_summary,
    score_association_summary,
)

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_is_diagnostic_only() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "operating_cycle_support_audit.yaml"
        ).read_text(encoding="utf-8")
    )
    assert (
        config["study"]["role"]
        == "posthoc_reused_test_operating_cycle_support_diagnostic"
    )
    assert any(
        "No new router"
        in item
        for item in config["guardrails"]
    )


def test_support_flags_use_train_support() -> None:
    train = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0],
            "y": [10.0, 11.0, 12.0],
        }
    )
    frame = pd.DataFrame(
        {
            "x": [2.0, 3.0],
            "y": [9.0, 10.0],
        }
    )

    atomic = {
        "x_high": {
            "feature": "x",
            "direction": "above_train_max",
        },
        "y_low": {
            "feature": "y",
            "direction": "below_train_min",
        },
    }
    composite = {
        "any_novel": {
            "any_of": ["x_high", "y_low"],
        }
    }

    flags = make_support_flags(
        frame,
        train,
        flag_config=atomic,
        composite_config=composite,
    )

    assert flags["x_high"].tolist() == [False, True]
    assert flags["y_low"].tolist() == [True, False]
    assert flags["any_novel"].tolist() == [True, True]


def test_score_association_reports_high_score_coverage() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=4,
        freq="5min",
    )
    flags = pd.DataFrame(
        {"novel": [False, True, True, False]},
        index=index,
    )
    scores = pd.Series(
        [0.0, 10.0, 8.0, 1.0],
        index=index,
    )

    result = score_association_summary(
        flags,
        scores,
        threshold=5.0,
    )["novel"]

    assert np.isclose(
        result["fraction_of_all_high_score_rows_with_flag"],
        1.0,
    )
    assert np.isclose(
        result["flagged_threshold_exceedance_fraction"],
        1.0,
    )


def test_joint_pattern_summary() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=3,
        freq="5min",
    )
    flags = pd.DataFrame(
        {
            "a": [True, True, False],
            "b": [False, True, False],
        },
        index=index,
    )
    scores = pd.Series(
        [2.0, 3.0, 0.0],
        index=index,
    )
    regimes = pd.Series(
        ["low_current_low_pressure"] * 3,
        index=index,
    )

    result = joint_pattern_summary(
        flags,
        scores,
        regimes,
        threshold=1.0,
        top_n=10,
    )

    patterns = {
        item["pattern"]
        for item in result["all_background"]
    }
    assert patterns == {"a", "a|b", "none"}


def test_run_summary_detects_long_run() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=5,
        freq="5min",
    )
    flag = pd.Series(
        [True, True, True, False, True],
        index=index,
    )

    result = run_summary(
        flag,
        gap_minutes=10,
    )

    assert result["run_count"] == 2
    assert result["run_rows"]["max"] == 3.0
