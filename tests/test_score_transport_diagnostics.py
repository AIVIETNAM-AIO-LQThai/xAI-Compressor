from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from ml.detection.score_transport_diagnostics import (
    feature_drift_summary,
    high_score_run_summary,
    incident_window_mask,
    transport_summary,
)

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_is_diagnostic_only() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "regime_score_transport_audit.yaml"
        ).read_text(encoding="utf-8")
    )
    assert (
        config["study"]["role"]
        == "posthoc_reused_test_score_transport_diagnostic"
    )
    assert any(
        "No candidate is promoted or tuned"
        in item
        for item in config["guardrails"]
    )


def test_incident_window_mask_uses_24h_pre_window() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=60,
        freq="1h",
    )
    incidents = [
        {
            "id": 1,
            "start": "2020-01-02 12:00",
            "end": "2020-01-02 14:00",
        }
    ]

    mask = incident_window_mask(
        index,
        incidents,
        hours_before=24,
    )

    assert bool(mask.loc["2020-01-01 12:00"])
    assert bool(mask.loc["2020-01-02 14:00"])
    assert not bool(mask.loc["2020-01-01 11:00"])


def test_transport_summary_uses_calibration_threshold() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=5,
        freq="5min",
    )
    calibration = pd.Series(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        index=index,
    )
    test = pd.Series(
        [0.0, 0.0, 4.0, 5.0, 6.0],
        index=index,
    )

    result = transport_summary(
        calibration,
        test,
        threshold_quantile=0.80,
    )

    assert np.isclose(
        result["calibration_raw_threshold"],
        4.0,
    )
    assert np.isclose(
        result["test"]["threshold_exceedance_fraction"],
        3 / 5,
    )


def test_feature_drift_detects_large_test_z() -> None:
    calibration = pd.DataFrame(
        {"x": [0.0, 0.5, -0.5, 1.0]}
    )
    background = pd.DataFrame(
        {"x": [10.0, 11.0, 12.0, 13.0]}
    )
    scaler = StandardScaler().fit(
        pd.DataFrame({"x": [-1.0, 0.0, 1.0]})
    )

    result = feature_drift_summary(
        calibration,
        background,
        features=["x"],
        scaler=scaler,
        abs_z_thresholds=[5.0],
    )

    item = result["by_feature"]["x"]
    assert (
        item["background_test_q99_abs_z"]
        > item["calibration_q99_abs_z"]
    )
    assert (
        item["background_test_fraction_abs_z_ge_5_0"]
        == 1.0
    )


def test_high_score_runs_split_on_non_high_row() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=5,
        freq="5min",
    )
    scores = pd.Series(
        [2.0, 2.0, 0.0, 2.0, 2.0],
        index=index,
    )
    regimes = pd.Series(
        ["a", "a", "a", "b", "b"],
        index=index,
    )

    summary, table = high_score_run_summary(
        scores,
        regimes,
        threshold=1.0,
        gap_minutes=10,
    )

    assert summary["run_count"] == 2
    assert len(table) == 2
    assert table["rows"].tolist() == [2, 2]
