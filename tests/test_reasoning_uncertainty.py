from __future__ import annotations

import pytest

from ml.reasoning import (
    infer_state_uncertainty,
    infer_total_outflow,
)
from ml.twin.physics import TwinParameters

PARAMETERS = TwinParameters(
    volume_m3=10.0,
    temperature_k=298.15,
)


def _constant_outflow_inference():
    return infer_total_outflow(
        [7.0, 7.0],
        [0.110],
        interval_seconds=60.0,
        parameters=PARAMETERS,
    )


def test_total_outflow_interval_without_demand():
    inference = (
        _constant_outflow_inference()
    )

    state = infer_state_uncertainty(
        inference,
        outflow_abs_error_kg_s=0.004,
    )

    assert (
        state.total_outflow_kg_s.lower
        == pytest.approx(0.106)
    )
    assert (
        state.total_outflow_kg_s.center
        == pytest.approx(0.110)
    )
    assert (
        state.total_outflow_kg_s.upper
        == pytest.approx(0.114)
    )

    assert state.demand_kg_s is None
    assert state.leak_kg_s is None
    assert state.leak_identifiable is False
    assert state.causal_claim is False


def test_demand_measurement_produces_leak_interval():
    inference = (
        _constant_outflow_inference()
    )

    state = infer_state_uncertainty(
        inference,
        outflow_abs_error_kg_s=0.004,
        independent_demand_kg_s=[
            0.090
        ],
        demand_abs_error_kg_s=0.002,
    )

    assert state.leak_identifiable is True

    assert (
        state.demand_kg_s is not None
    )
    assert (
        state.leak_kg_s is not None
    )

    assert (
        state.demand_kg_s.lower
        == pytest.approx(0.088)
    )
    assert (
        state.demand_kg_s.upper
        == pytest.approx(0.092)
    )

    assert (
        state.leak_kg_s.lower
        == pytest.approx(0.014)
    )
    assert (
        state.leak_kg_s.center
        == pytest.approx(0.020)
    )
    assert (
        state.leak_kg_s.upper
        == pytest.approx(0.026)
    )


def test_uncertainty_rejects_negative_error():
    inference = (
        _constant_outflow_inference()
    )

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        infer_state_uncertainty(
            inference,
            outflow_abs_error_kg_s=-0.001,
        )


def test_uncertainty_rejects_impossible_demand():
    inference = (
        _constant_outflow_inference()
    )

    with pytest.raises(
        ValueError,
        match=(
            "physically impossible"
        ),
    ):
        infer_state_uncertainty(
            inference,
            outflow_abs_error_kg_s=0.001,
            independent_demand_kg_s=[
                0.130
            ],
            demand_abs_error_kg_s=0.001,
        )