import numpy as np
import pandas as pd
import pytest

from ml.detection.pca_detector import (
    fit_pca_detector,
    score_pca_detector,
)
from ml.explainability.verification import (
    contribution_percentiles,
    fit_contribution_reference,
    score_with_feature_group_repair,
    temporal_evidence_window,
    verify_counterfactual_alert,
)


def make_training_frame() -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-01",
        periods=120,
        freq="5min",
    )

    base = np.linspace(
        -3.0,
        3.0,
        len(index),
    )

    return pd.DataFrame(
        {
            "tp2__mean":
                base,
            "dv_pressure__mean":
                (
                    2.0 * base
                    + 0.02
                    * np.sin(
                        np.arange(
                            len(index)
                        )
                    )
                ),
        },
        index=index,
    )


def make_detector():
    frame = make_training_frame()

    return fit_pca_detector(
        frame,
        [
            "tp2__mean",
            "dv_pressure__mean",
        ],
        variance_retained=0.95,
        scaler_name="robust",
    )


def test_contribution_percentiles_use_reference_distribution():
    calibration = pd.DataFrame(
        {
            "tp2": [
                1.0,
                2.0,
                3.0,
            ],
            "dv_pressure": [
                10.0,
                20.0,
                30.0,
            ],
        }
    )

    reference = (
        fit_contribution_reference(
            calibration
        )
    )

    observed = pd.DataFrame(
        {
            "tp2": [
                2.0,
                100.0,
            ],
            "dv_pressure": [
                5.0,
                25.0,
            ],
        }
    )

    percentiles = (
        contribution_percentiles(
            observed,
            reference,
        )
    )

    assert (percentiles["tp2"].iloc[0] == pytest.approx(0.5))
    assert (percentiles["tp2"].iloc[1] == pytest.approx(1.0))
    assert (percentiles["dv_pressure"].iloc[0] == pytest.approx(0.0))
    assert (percentiles["dv_pressure"].iloc[1] == pytest.approx(2.0 / 3.0))

def test_contribution_percentiles_handle_ties_with_midrank():
    calibration = pd.DataFrame(
        {
            "inactive_signal": [
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        }
    )

    reference = fit_contribution_reference(calibration)

    observed = pd.DataFrame(
        {
            "inactive_signal": [
                0.0,
                1.0,
            ]
        }
    )

    percentiles = (
        contribution_percentiles(
            observed,
            reference,
        )
    )

    assert (percentiles["inactive_signal"].iloc[0] == pytest.approx(0.5))
    assert (percentiles["inactive_signal"].iloc[1] == pytest.approx(1.0))

def test_temporal_window_uses_only_past_and_current_bins():
    index = pd.date_range(
        "2020-01-01",
        periods=20,
        freq="5min",
    )

    frame = pd.DataFrame(
        {
            "value":
                np.arange(
                    len(index)
                )
        },
        index=index,
    )

    end = index[10]

    window = temporal_evidence_window(
        frame,
        end_time=end,
        window_bins=4,
        bin_minutes=5,
    )

    assert len(window) == 4
    assert (window.index[-1] == end)

    assert (
        window.index[0]
        == end
        - pd.Timedelta(
            minutes=15
        )
    )

    assert (window.index.max() <= end)


def test_empty_repair_reproduces_original_pca_score():
    detector = make_detector()

    frame = make_training_frame().iloc[
        20:30
    ]

    original = score_pca_detector(
        frame,
        detector,
    )

    repaired = (
        score_with_feature_group_repair(
            frame,
            detector,
            groups=(),
        )
    )

    np.testing.assert_allclose(
        original.to_numpy(),
        repaired.to_numpy(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_repairing_all_groups_drives_score_near_zero():
    detector = make_detector()

    frame = pd.DataFrame(
        {
            "tp2__mean": [
                0.0,
                1.0,
            ],
            "dv_pressure__mean": [
                10.0,
                -8.0,
            ],
        },
        index=pd.date_range(
            "2020-02-01",
            periods=2,
            freq="5min",
        ),
    )

    repaired = (
        score_with_feature_group_repair(
            frame,
            detector,
            groups={
                "tp2",
                "dv_pressure",
            },
        )
    )

    assert (
        repaired.max()
        < 1.0e-20
    )


def test_unknown_group_is_rejected():
    detector = make_detector()

    frame = make_training_frame().iloc[
        :2
    ]

    with pytest.raises(
        ValueError,
        match="Unknown feature groups",
    ):
        score_with_feature_group_repair(
            frame,
            detector,
            groups={
                "does_not_exist"
            },
        )


def test_counterfactual_reuses_alert_pipeline():
    detector = make_detector()

    index = pd.date_range(
        "2020-03-01",
        periods=8,
        freq="5min",
    )

    frame = pd.DataFrame(
        {
            "tp2__mean": [
                -1.0,
                -0.5,
                0.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            "dv_pressure__mean": [
                -2.0,
                -1.0,
                0.0,
                1.0,
                10.0,
                10.0,
                10.0,
                10.0,
            ],
        },
        index=index,
    )

    raw_scores = score_pca_detector(
        frame,
        detector,
    )

    anomalous_scores = (
        raw_scores.iloc[-4:]
    )

    threshold = float(
        anomalous_scores.min()
        * 0.5
    )

    result = verify_counterfactual_alert(
        frame,
        detector,
        raw_scores,
        alert_timestamp=index[-1],
        groups={
            "tp2",
            "dv_pressure",
        },
        window_bins=4,
        bin_minutes=5,
        ewma_alpha=1.0,
        threshold=threshold,
        persistence_hits=3,
        persistence_window=4,
        reset_gap_minutes=10,
    )

    assert result.original_alert

    assert not result.repaired_alert

    assert result.alert_cleared

    assert (
        result.repaired_smoothed_score
        < result.original_smoothed_score
    )


def test_empty_counterfactual_preserves_alert_state():
    detector = make_detector()

    index = pd.date_range(
        "2020-04-01",
        periods=8,
        freq="5min",
    )

    frame = pd.DataFrame(
        {
            "tp2__mean": [
                -1.0,
                -0.5,
                0.0,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            "dv_pressure__mean": [
                -2.0,
                -1.0,
                0.0,
                1.0,
                10.0,
                10.0,
                10.0,
                10.0,
            ],
        },
        index=index,
    )

    raw_scores = score_pca_detector(
        frame,
        detector,
    )

    threshold = float(
        raw_scores.iloc[-4:].min()
        * 0.5
    )

    result = verify_counterfactual_alert(
        frame,
        detector,
        raw_scores,
        alert_timestamp=index[-1],
        groups=(),
        window_bins=4,
        bin_minutes=5,
        ewma_alpha=1.0,
        threshold=threshold,
        persistence_hits=3,
        persistence_window=4,
        reset_gap_minutes=10,
    )

    assert result.original_alert
    assert result.repaired_alert

    assert not result.alert_cleared

    assert (
        result.repaired_smoothed_score
        == pytest.approx(
            result.original_smoothed_score
        )
    )