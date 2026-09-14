from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from ml.detection.feature_stationarity_diagnostics import (
    analog_min_coherence,
    contribution_share,
    digital_support_summary,
    extreme_run_summary,
    support_and_scaler_summary,
)

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_is_diagnostic_only() -> None:
    config = yaml.safe_load(
        (
            ROOT
            / "configs"
            / "feature_stationarity_shortcut_audit.yaml"
        ).read_text(encoding="utf-8")
    )
    assert (
        config["study"]["role"]
        == "posthoc_reused_test_feature_stationarity_shortcut_diagnostic"
    )
    assert any(
        "No feature is removed"
        in item
        for item in config["guardrails"]
    )


def test_support_summary_detects_out_of_train_support() -> None:
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    calibration = pd.DataFrame({"x": [0.5, 1.5]})
    background = pd.DataFrame({"x": [-1.0, 3.0]})
    scaler = StandardScaler().fit(train[["x"]])

    result = support_and_scaler_summary(
        train,
        calibration,
        background,
        features=["x"],
        all_model_features=["x"],
        scaler=scaler,
        abs_z_thresholds=[1.0],
    )["x"]

    assert np.isclose(
        result["background_test_below_train_min_fraction"],
        0.5,
    )
    assert np.isclose(
        result["background_test_above_train_max_fraction"],
        0.5,
    )


def test_analog_min_coherence_detects_min_only_excursion() -> None:
    frame = pd.DataFrame(
        {
            "s__mean": [0.0, 0.0],
            "s__std": [0.0, 0.0],
            "s__min": [-100.0, 0.0],
            "s__max": [0.0, 0.0],
            "s__last": [0.0, 0.0],
        }
    )
    features = list(frame.columns)
    scaler = StandardScaler().fit(
        pd.DataFrame(
            {
                column: [-1.0, 0.0, 1.0]
                for column in features
            }
        )
    )

    result = analog_min_coherence(
        frame,
        sensor="s",
        all_model_features=features,
        scaler=scaler,
        min_z_threshold=10.0,
        context_z_threshold=5.0,
    )

    assert result["extreme_min_count"] == 1
    assert np.isclose(
        result[
            "among_extreme_min_mean_and_last_nonextreme_fraction"
        ],
        1.0,
    )


def test_digital_support_summary() -> None:
    frame = pd.DataFrame(
        {
            "d__last": [0.0, 1.0, 1.0, 0.0],
            "d__active_ratio": [0.0, 0.5, 1.0, 0.0],
            "d__transitions": [0.0, 1.0, 2.0, 0.0],
        }
    )

    result = digital_support_summary(
        frame,
        sensor="d",
    )

    assert np.isclose(
        result["last"]["active_fraction_gt_0_5"],
        0.5,
    )
    assert np.isclose(
        result["transitions"]["any_transition_fraction"],
        0.5,
    )


def test_contribution_share_sums_to_one() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, 1.0],
            "b": [3.0, 1.0],
        }
    )

    result = contribution_share(frame)
    assert np.isclose(
        sum(result["feature_share"].values()),
        1.0,
    )


def test_extreme_run_summary_splits_runs() -> None:
    index = pd.date_range(
        "2020-01-01",
        periods=5,
        freq="5min",
    )
    series = pd.Series(
        [6.0, 7.0, 0.0, 8.0, 9.0],
        index=index,
    )

    result = extreme_run_summary(
        series,
        threshold=5.0,
        gap_minutes=10,
    )

    assert result["extreme_row_count"] == 4
    assert result["run_count"] == 2
    assert result["run_rows"]["max"] == 2.0
