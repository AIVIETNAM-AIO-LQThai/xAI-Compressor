import pandas as pd
import pytest

from ml.detection.alerts import (
    causal_ewma,
    extract_alert_episodes,
    persistent_alerts,
)
from ml.detection.robust_z import (
    fit_robust_z,
    score_robust_z,
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
        train,
        ["varying", "constant"],
        top_k=1,
        min_scale=1.0e-9,
    )

    assert "varying" in model.features
    assert "constant" in (
        model.dropped_features
    )


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
        train,
        ["x"],
        top_k=1,
        min_scale=1.0e-9,
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