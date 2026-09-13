from __future__ import annotations

import pytest

from ml.reasoning import (
    generate_physical_scenarios,
    infer_state_uncertainty,
    infer_total_outflow,
)
from ml.twin.physics import TwinParameters

PARAMETERS = TwinParameters(
    volume_m3=10.0,
    temperature_k=298.15,
)


def _inference():
    return infer_total_outflow(
        [7.0, 7.0],
        [0.110],
        interval_seconds=60.0,
        parameters=PARAMETERS,
    )


def test_nonidentifiable_state_keeps_allocation_unknown():
    state = infer_state_uncertainty(
        _inference(),
        outflow_abs_error_kg_s=0.004,
    )

    scenario_set = (
        generate_physical_scenarios(
            state
        )
    )

    assert (
        scenario_set
        .allocation_identifiable
        is False
    )

    assert len(
        scenario_set.scenarios
    ) == 3

    values = [
        item.total_outflow_kg_s
        for item in scenario_set.scenarios
    ]

    assert values == pytest.approx(
        [
            0.106,
            0.110,
            0.114,
        ]
    )

    for scenario in (
        scenario_set.scenarios
    ):
        assert scenario.demand_kg_s is None
        assert scenario.leak_kg_s is None
        assert (
            scenario
            .allocation_identifiable
            is False
        )
        assert scenario.causal_claim is False


def test_identifiable_state_generates_consistent_splits():
    state = infer_state_uncertainty(
        _inference(),
        outflow_abs_error_kg_s=0.004,
        independent_demand_kg_s=[
            0.090
        ],
        demand_abs_error_kg_s=0.002,
    )

    scenario_set = (
        generate_physical_scenarios(
            state
        )
    )

    assert (
        scenario_set
        .allocation_identifiable
        is True
    )

    assert len(
        scenario_set.scenarios
    ) >= 3

    totals = [
        item.total_outflow_kg_s
        for item in scenario_set.scenarios
    ]

    assert min(totals) == pytest.approx(
        0.106
    )

    assert max(totals) == pytest.approx(
        0.114
    )

    assert any(
        item.total_outflow_kg_s
        == pytest.approx(0.110)
        for item in scenario_set.scenarios
    )

    for scenario in (
        scenario_set.scenarios
    ):
        assert (
            scenario.demand_kg_s
            is not None
        )
        assert (
            scenario.leak_kg_s
            is not None
        )

        assert (
            scenario.total_outflow_kg_s
            == pytest.approx(
                scenario.demand_kg_s
                + scenario.leak_kg_s
            )
        )

        assert (
            0.106 - 1.0e-9
            <= scenario.total_outflow_kg_s
            <= 0.114 + 1.0e-9
        )


def test_zero_width_state_generates_one_scenario():
    state = infer_state_uncertainty(
        _inference(),
        outflow_abs_error_kg_s=0.0,
    )

    scenario_set = (
        generate_physical_scenarios(
            state
        )
    )

    assert len(
        scenario_set.scenarios
    ) == 1

    assert (
        scenario_set
        .scenarios[0]
        .total_outflow_kg_s
        == pytest.approx(0.110)
    )


def test_negative_scenario_tolerance_is_rejected():
    state = infer_state_uncertainty(
        _inference(),
        outflow_abs_error_kg_s=0.004,
    )

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        generate_physical_scenarios(
            state,
            tolerance_kg_s=-1.0,
        )