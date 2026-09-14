from __future__ import annotations

import pandas as pd
import pytest

from ml.detection.feature_energy import (
    explain_feature_energy_detector,
    fit_feature_energy_detector,
)


def test_feature_energy_score_equals_contribution_sum():
    train = pd.DataFrame(
        {
            "a": [
                0.0,
                1.0,
                2.0,
                3.0,
            ],
            "b": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        }
    )

    detector = (
        fit_feature_energy_detector(
            train,
            ["a", "b"],
            scaler_name="robust",
        )
    )

    explained = (
        explain_feature_energy_detector(
            train,
            detector,
        )
    )

    assert (
        explained.contributions
        >= 0.0
    ).all().all()

    assert (
        explained.score.to_numpy()
        == pytest.approx(
            explained.contributions
            .sum(axis=1)
            .to_numpy()
        )
    )


def test_feature_energy_normalized_rows_sum_to_one_when_positive():
    train = pd.DataFrame(
        {
            "a": [
                0.0,
                1.0,
                2.0,
                4.0,
            ],
            "b": [
                0.0,
                2.0,
                3.0,
                5.0,
            ],
        }
    )

    detector = (
        fit_feature_energy_detector(
            train,
            ["a", "b"],
            scaler_name="standard",
        )
    )

    explained = (
        explain_feature_energy_detector(
            train,
            detector,
        )
    )

    positive = (
        explained.score
        > 0.0
    )

    assert (
        explained.normalized_contributions
        .loc[positive]
        .sum(axis=1)
        .to_numpy()
        == pytest.approx(
            [1.0]
            * int(
                positive.sum()
            )
        )
    )
