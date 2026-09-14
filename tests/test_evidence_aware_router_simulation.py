from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.energy.evidence_aware_simulation import (
    counterfactual_mixture,
    frozen_router_probability,
    location_shift,
    monotonic_nondecreasing,
    operating_regimes,
    route_summary,
    scale_shift,
    tail_inflation,
)


def test_frozen_router_probability() -> None:
    frame = pd.DataFrame(
        {
            "a": [0.0, 1.0],
            "b": [0.0, 0.0],
        }
    )

    probabilities = frozen_router_probability(
        frame,
        feature_names=["a", "b"],
        scaler_mean=np.asarray([0.0, 0.0]),
        scaler_scale=np.asarray([1.0, 1.0]),
        coef=np.asarray([1.0, 0.0]),
        intercept=0.0,
    )

    assert probabilities[0] == pytest.approx(0.5)
    assert probabilities[1] > probabilities[0]


def test_route_summary() -> None:
    result = route_summary(
        np.asarray([0.1, 0.4, 0.9]),
        threshold=0.3,
    )

    assert result["rows"] == 3
    assert result["tcn_invocations"] == 2
    assert result["tcn_invocation_fraction"] == pytest.approx(2 / 3)


def test_location_and_scale_shift() -> None:
    train = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    frame = pd.DataFrame({"x": [1.0, 2.0]})

    located = location_shift(
        frame,
        train,
        features=["x"],
        severity_sigma=1.0,
    )
    scaled = scale_shift(
        frame,
        train,
        features=["x"],
        factor=2.0,
    )

    assert not located.equals(frame)
    assert not scaled.equals(frame)


def test_tail_inflation_changes_only_masked_rows() -> None:
    train = pd.DataFrame(
        {
            "x": np.asarray(
                [0.0, 1.0, 2.0],
                dtype=np.float32,
            )
        }
    )
    frame = pd.DataFrame(
        {
            "x": np.asarray(
                [1.0, 2.0, 3.0],
                dtype=np.float32,
            )
        }
    )
    mask = pd.Series(
        [False, True, False],
        index=frame.index,
    )

    shifted = tail_inflation(
        frame,
        train,
        features=["x"],
        tail_mask=mask,
        factor=2.0,
    )

    assert shifted["x"].dtype == np.float64
    assert shifted.loc[0, "x"] == frame.loc[0, "x"]
    assert shifted.loc[2, "x"] == frame.loc[2, "x"]
    assert shifted.loc[1, "x"] != frame.loc[1, "x"]


def test_monotonic_nondecreasing() -> None:
    assert monotonic_nondecreasing([0.1, 0.2, 0.2, 0.4])
    assert not monotonic_nondecreasing([0.1, 0.3, 0.2])


def test_operating_mix_counterfactual() -> None:
    index = pd.RangeIndex(6)
    regimes = pd.Series(
        ["low", "low", "middle", "middle", "high", "high"],
        index=index,
    )
    route = pd.Series(
        [False, False, False, True, True, True],
        index=index,
    )

    result = counterfactual_mixture(
        route=route,
        regimes=regimes,
        weights={
            "low": 0.2,
            "middle": 0.3,
            "high": 0.5,
        },
    )

    # low=0, middle=0.5, high=1
    assert result["counterfactual_invocation_fraction"] == pytest.approx(
        0.65
    )


def test_operating_regimes() -> None:
    series = pd.Series([1.0, 5.0, 9.0])
    regimes = operating_regimes(
        series,
        low_upper=2.0,
        high_lower=8.0,
    )

    assert regimes.tolist() == ["low", "middle", "high"]
