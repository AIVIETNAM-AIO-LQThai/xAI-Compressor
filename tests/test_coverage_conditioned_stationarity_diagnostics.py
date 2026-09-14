from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml.detection.coverage_conditioned_stationarity_diagnostics import (
    association_summary,
    coverage_strata,
    make_support_flags,
    quality_novelty_cells,
    sample_count_strata,
)

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_is_diagnostic_only() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "coverage_conditioned_stationarity_audit.yaml"
        ).read_text(encoding="utf-8")
    )
    assert (
        config["study"]["role"]
        == "posthoc_reused_test_coverage_conditioned_stationarity_diagnostic"
    )
    assert any(
        "No row is removed"
        in item
        for item in config["guardrails"]
    )


def test_quality_strata_are_frozen() -> None:
    frame = pd.DataFrame(
        {
            "coverage_ratio": [1.0, 0.95, 0.85],
            "sample_count": [30, 28, 25],
        }
    )

    assert coverage_strata(frame).tolist() == [
        "full",
        "partial_high",
        "partial_low",
    ]
    assert sample_count_strata(frame).tolist() == [
        "complete",
        "mild_underfill",
        "severe_underfill",
    ]


def test_support_flags_use_train_support() -> None:
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    frame = pd.DataFrame({"x": [2.0, 3.0]})

    flags = make_support_flags(
        frame,
        train,
        flag_config={
            "x_high": {
                "feature": "x",
                "direction": "above_train_max",
            }
        },
        composite_config={
            "any_cycle_novel": {
                "any_of": ["x_high"],
            }
        },
    )

    assert flags["x_high"].tolist() == [False, True]
    assert flags["any_cycle_novel"].tolist() == [False, True]


def test_quality_novelty_cells_partition_rows() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=4,
        freq="5min",
    )
    frame = pd.DataFrame(
        {
            "coverage_ratio": [1.0, 1.0, 0.8, 0.8],
            "sample_count": [30, 30, 24, 24],
        },
        index=index,
    )
    flags = pd.DataFrame(
        {
            "any_cycle_novel": [False, True, False, True],
        },
        index=index,
    )
    scores = pd.Series(
        [0.0, 10.0, 1.0, 20.0],
        index=index,
    )
    regimes = pd.Series(
        ["low_current_low_pressure"] * 4,
        index=index,
    )

    result = quality_novelty_cells(
        frame,
        flags,
        scores,
        regimes,
        threshold=5.0,
    )

    assert sum(
        item["rows"] for item in result.values()
    ) == 4
    assert result["full_novel"]["rows"] == 1
    assert result["partial_novel"]["rows"] == 1


def test_association_detects_partial_coverage_link() -> None:
    index = pd.RangeIndex(4)
    frame = pd.DataFrame(
        {
            "coverage_ratio": [1.0, 1.0, 0.8, 0.8],
            "sample_count": [30, 30, 24, 24],
        },
        index=index,
    )
    flags = pd.DataFrame(
        {
            "novel": [False, False, True, True],
        },
        index=index,
    )

    result = association_summary(
        flags,
        frame,
    )["novel"]["partial_vs_full_coverage"]

    assert np.isclose(
        result["flag_prevalence_exposed"],
        1.0,
    )
    assert np.isclose(
        result["flag_prevalence_unexposed"],
        0.0,
    )
