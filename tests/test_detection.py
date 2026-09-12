import numpy as np
import pandas as pd
import pytest

from ml.detection.alerts import (
    causal_ewma,
    extract_alert_episodes,
    persistent_alerts,
)
from ml.detection.isolation_forest import fit_isolation_forest, score_isolation_forest
from ml.detection.pca_detector import fit_pca_detector, score_pca_detector
from ml.detection.robust_z import fit_robust_z, score_robust_z
from ml.explainability.pca import (
    aggregate_contributions,
    causal_ewma_contributions,
    explain_pca_detector,
)


def test_constant_feature_is_dropped():
    index = pd.date_range(
        "2020-01-01",
        periods=7,
        freq="5min",
    )

    train = pd.DataFrame(
        {
            "varying": [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ],
            "constant": [5.0] * 7,
        },
        index=index,
    )

    model = fit_robust_z(
        train, ["varying", "constant"], top_k=1, min_scale=1.0e-9,
    )

    assert "varying" in model.features
    assert "constant" in (model.dropped_features)


def test_larger_deviation_gets_larger_score():
    index = pd.date_range(
        "2020-01-01",
        periods=7,
        freq="5min",
    )

    train = pd.DataFrame(
        {
            "x": [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
            ]
        },
        index=index,
    )

    model = fit_robust_z(
        train, ["x"], top_k=1, min_scale=1.0e-9,
    )

    test = pd.DataFrame(
        {"x": [3.0, 30.0]},
        index=pd.date_range(
            "2020-02-01",
            periods=2,
            freq="5min",
        ),
    )

    scores, _ = score_robust_z(
        test,
        model,
    )

    assert scores.iloc[1] > scores.iloc[0]

def test_ewma_resets_after_large_gap():
    scores = pd.Series(
        [0.0, 10.0, 0.0],
        index=pd.to_datetime(
            [
                "2020-01-01 00:00",
                "2020-01-01 00:05",
                "2020-01-01 01:00",
            ]
        ),
    )

    result = causal_ewma(
        scores,
        alpha=0.2,
        reset_gap_minutes=10,
    )

    assert result.iloc[-1] == 0.0

def test_three_of_four_persistence():
    scores = pd.Series(
        [2.0, 2.0, 0.0, 2.0],
        index=pd.date_range(
            "2020-01-01",
            periods=4,
            freq="5min",
        ),
    )

    _, alerts = persistent_alerts(
        scores,
        threshold=1.0,
        required_hits=3,
        window_bins=4,
        reset_gap_minutes=10,
    )

    assert not alerts.iloc[0]
    assert not alerts.iloc[1]
    assert not alerts.iloc[2]
    assert alerts.iloc[3]


def test_episode_merge_does_not_cross_data_gap():
    alerts = pd.Series(
        [True, False, True, True],
        index=pd.to_datetime(
            [
                "2020-01-01 00:00",
                "2020-01-01 00:05",
                "2020-01-01 00:10",
                "2020-01-01 01:00",
            ]
        ),
    )

    episodes = extract_alert_episodes(
        alerts,
        merge_minutes=30,
        reset_gap_minutes=10,
    )

    assert len(episodes) == 2
    assert episodes[0].start == pd.Timestamp(
        "2020-01-01 00:00"
    )
    assert episodes[0].end == pd.Timestamp(
        "2020-01-01 00:10"
    )
    assert episodes[1].start == pd.Timestamp(
        "2020-01-01 01:00"
    )

def test_ewma_rejects_non_datetime_index():
    scores = pd.Series(
        [1.0, 2.0, 3.0],
        index=[0, 1, 2],
    )

    with pytest.raises(
        TypeError,
        match="DatetimeIndex",
    ):
        causal_ewma(
            scores,
            alpha=0.2,
            reset_gap_minutes=10,
        )

def test_pca_scores_outlier_higher():
    index = pd.date_range(
        "2020-01-01",
        periods=20,
        freq="5min",
    )

    train = pd.DataFrame(
        {
            "x": range(20),
            "y": range(20),
        },
        index=index, dtype=float,
    )

    detector = fit_pca_detector(
        train,
        ["x", "y"],
        variance_retained=0.95,
    )

    test = pd.DataFrame(
        {
            "x": [10.0, 100.0],
            "y": [10.0, -100.0],
        },
        index=pd.date_range(
            "2020-02-01",
            periods=2,
            freq="5min",
        ),
    )

    scores = score_pca_detector(
        test, detector,
    )

    assert scores.iloc[1] > scores.iloc[0]


def test_isolation_forest_returns_finite_scores():
    index = pd.date_range(
        "2020-01-01",
        periods=50,
        freq="5min",
    )

    train = pd.DataFrame(
        {
            "x": range(50),
            "y": range(50),
        },
        index=index, dtype=float,
    )

    detector = fit_isolation_forest(
        train,
        ["x", "y"],
        n_estimators=50,
        random_state=42,
    )

    scores = score_isolation_forest(
        train, detector,
    )

    assert scores.notna().all()
    assert len(scores) == len(train)

def test_pca_explanation_matches_score():
    index = pd.date_range(
        "2020-01-01",
        periods=20,
        freq="5min",
    )

    train = pd.DataFrame(
        {
            "x": range(20),
            "y": range(20),
        },
        index=index, dtype=float,
    )

    detector = fit_pca_detector(
        train,
        ["x", "y"],
        variance_retained=0.95,
    )

    explanation = explain_pca_detector(
        train, detector,
    )

    score = score_pca_detector(
        train, detector,
    )

    np.testing.assert_allclose(
        explanation.score.to_numpy(),
        score.to_numpy(),
    )

def test_pca_normalized_contributions_sum_to_one():
    index = pd.date_range(
        "2020-01-01",
        periods=20,
        freq="5min",
    )

    train = pd.DataFrame(
        {
            "x": range(20),
            "y": range(20),
        },
        index=index,
        dtype=float,
    )

    detector = fit_pca_detector(
        train,
        ["x", "y"],
        variance_retained=0.95,
    )

    test = pd.DataFrame(
        {
            "x": [100.0],
            "y": [-100.0],
        },
        index=[
            pd.Timestamp("2020-02-01")
        ],
    )

    explanation = explain_pca_detector(
        test, detector,
    )

    total = (
        explanation
        .normalized_contributions
        .sum(axis=1)
        .iloc[0]
    )

    assert np.isclose(total, 1.0,)

def test_pca_grouped_contributions_preserve_total():
    contributions = pd.DataFrame(
        {
            "tp2__mean": [1.0],
            "tp2__max": [2.0],
            "motor_current__mean": [3.0],
        },
        index=[
            pd.Timestamp(
                "2020-01-01"
            )
        ],
    )

    grouped = aggregate_contributions(
        contributions
    )

    assert np.isclose(
        grouped.sum(axis=1).iloc[0],
        contributions.sum(axis=1).iloc[0],
    )

    assert grouped["tp2"].iloc[0] == 3.0

    assert (
        grouped[
            "motor_current"
        ].iloc[0]
        == 3.0
    )

def test_smoothed_contributions_match_ewma_score():
    index = pd.to_datetime(
        [
            "2020-01-01 00:00:00",
            "2020-01-01 00:05:00",
            "2020-01-01 00:10:00",
            "2020-01-01 00:30:00",
        ]
    )

    contributions = pd.DataFrame(
        {
            "feature_a": [
                1.0,
                3.0,
                2.0,
                10.0,
            ],
            "feature_b": [
                2.0,
                1.0,
                6.0,
                20.0,
            ],
        },
        index=index,
    )

    smoothed_contributions = (
        causal_ewma_contributions(
            contributions,
            alpha=0.2,
            reset_gap_minutes=10,
        )
    )

    raw_score = contributions.sum(
        axis=1
    )

    smoothed_score = causal_ewma(
        raw_score,
        alpha=0.2,
        reset_gap_minutes=10,
    )

    np.testing.assert_allclose(
        smoothed_contributions
        .sum(axis=1)
        .to_numpy(),
        smoothed_score.to_numpy(),
    )

def test_pca_supports_standard_scaler():
    index = pd.date_range(
        "2020-01-01",
        periods=30,
        freq="5min",
    )

    train = pd.DataFrame(
        {
            "x": range(30),
            "y": range(30),
        },
        index=index,
        dtype=float,
    )

    detector = fit_pca_detector(
        train,
        ["x", "y"],
        variance_retained=0.95,
        scaler_name="standard",
    )

    scores = score_pca_detector(
        train,
        detector,
    )

    assert (
        detector.scaler_name
        == "standard"
    )

    assert scores.notna().all()

    assert len(scores) == len(train)


def test_pca_rejects_unknown_scaler():
    index = pd.date_range(
        "2020-01-01",
        periods=10,
        freq="5min",
    )

    train = pd.DataFrame(
        {
            "x": range(10),
            "y": range(10),
        },
        index=index,
        dtype=float,
    )

    with pytest.raises(
        ValueError,
        match="Unknown scaler method",
    ):
        fit_pca_detector(
            train,
            ["x", "y"],
            scaler_name="invalid",
        )