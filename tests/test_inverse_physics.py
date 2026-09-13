from __future__ import annotations

import numpy as np
import pytest

from ml.reasoning import (
    infer_total_outflow,
    verify_leak_vs_demand,
)
from ml.twin.physics import (
    TwinParameters,
    bar_g_to_absolute_pa,
    step_pressure,
)

PARAMETERS = TwinParameters(
    volume_m3=10.0,
    temperature_k=298.15,
)

INTERVAL_SECONDS = 60.0


def _pressure_trace(
    *,
    initial_pressure_bar_g: float,
    inflow_kg_s: list[float],
    total_outflow_kg_s: list[float],
) -> list[float]:
    if len(inflow_kg_s) != len(
        total_outflow_kg_s
    ):
        raise ValueError(
            "Synthetic flows must have "
            "matching lengths."
        )

    pressure_pa = (
        bar_g_to_absolute_pa(
            initial_pressure_bar_g,
            ambient_pressure_pa=(
                PARAMETERS
                .ambient_pressure_pa
            ),
        )
    )

    trace = [
        initial_pressure_bar_g
    ]

    for inflow, outflow in zip(
        inflow_kg_s,
        total_outflow_kg_s,
        strict=True,
    ):
        pressure_pa = step_pressure(
            pressure_pa=pressure_pa,
            mass_flow_in_kg_s=inflow,
            demand_mass_flow_kg_s=outflow,
            leak_mass_flow_kg_s=0.0,
            timestep_seconds=(
                INTERVAL_SECONDS
            ),
            parameters=PARAMETERS,
        )

        pressure_bar_g = (
            (
                pressure_pa
                - PARAMETERS
                .ambient_pressure_pa
            )
            / 100_000.0
        )

        trace.append(
            float(pressure_bar_g)
        )

    return trace


def test_inverse_physics_recovers_total_outflow():
    inflow = [0.095] * 10
    total_outflow = [0.110] * 10

    pressure = _pressure_trace(
        initial_pressure_bar_g=7.0,
        inflow_kg_s=inflow,
        total_outflow_kg_s=total_outflow,
    )

    inferred = infer_total_outflow(
        pressure,
        inflow,
        interval_seconds=(
            INTERVAL_SECONDS
        ),
        parameters=PARAMETERS,
    )

    assert (
        inferred.mean_total_outflow_kg_s
        == pytest.approx(
            0.110,
            abs=1.0e-12,
        )
    )

    assert (
        inferred.std_total_outflow_kg_s
        == pytest.approx(
            0.0,
            abs=1.0e-12,
        )
    )

    assert (
        inferred.physically_consistent
        is True
    )

    assert inferred.causal_claim is False


def test_leak_and_demand_can_be_observationally_equivalent():
    inflow = [0.095] * 10

    leak_case_total = [
        0.090 + 0.020
    ] * 10

    demand_case_total = [
        0.105 + 0.005
    ] * 10

    leak_pressure = _pressure_trace(
        initial_pressure_bar_g=7.0,
        inflow_kg_s=inflow,
        total_outflow_kg_s=(
            leak_case_total
        ),
    )

    demand_pressure = _pressure_trace(
        initial_pressure_bar_g=7.0,
        inflow_kg_s=inflow,
        total_outflow_kg_s=(
            demand_case_total
        ),
    )

    assert np.allclose(
        leak_pressure,
        demand_pressure,
        atol=1.0e-12,
        rtol=0.0,
    )


def test_pressure_only_does_not_identify_leak_vs_demand():
    inflow = [0.095] * 10
    total_outflow = [0.110] * 10

    pressure = _pressure_trace(
        initial_pressure_bar_g=7.0,
        inflow_kg_s=inflow,
        total_outflow_kg_s=total_outflow,
    )

    inferred = infer_total_outflow(
        pressure,
        inflow,
        interval_seconds=(
            INTERVAL_SECONDS
        ),
        parameters=PARAMETERS,
    )

    leak, demand = (
        verify_leak_vs_demand(
            inferred,
            reference_total_outflow_kg_s=(
                0.095
            ),
        )
    )

    assert (
        leak.status
        == "observationally_equivalent"
    )

    assert (
        demand.status
        == "observationally_equivalent"
    )

    assert leak.identifiable is False
    assert demand.identifiable is False

    assert (
        leak.estimated_change_kg_s
        == pytest.approx(
            0.015,
            abs=1.0e-12,
        )
    )


def test_independent_demand_identifies_leak_increase():
    inflow = [0.095] * 10

    demand_observed = [0.090] * 10
    leak_actual = [0.020] * 10

    total_outflow = [
        demand + leak
        for demand, leak in zip(
            demand_observed,
            leak_actual,
            strict=True,
        )
    ]

    pressure = _pressure_trace(
        initial_pressure_bar_g=7.0,
        inflow_kg_s=inflow,
        total_outflow_kg_s=total_outflow,
    )

    inferred = infer_total_outflow(
        pressure,
        inflow,
        interval_seconds=(
            INTERVAL_SECONDS
        ),
        parameters=PARAMETERS,
    )

    leak, demand = (
        verify_leak_vs_demand(
            inferred,
            reference_total_outflow_kg_s=(
                0.095
            ),
            independent_demand_kg_s=(
                demand_observed
            ),
            reference_demand_kg_s=0.090,
            nominal_leak_kg_s=0.005,
        )
    )

    assert leak.identifiable is True

    assert leak.status == "supported"

    assert (
        leak.estimated_level_kg_s
        == pytest.approx(
            0.020,
            abs=1.0e-12,
        )
    )

    assert (
        leak.estimated_change_kg_s
        == pytest.approx(
            0.015,
            abs=1.0e-12,
        )
    )

    assert demand.status == "not_supported"

    assert (
        demand.estimated_change_kg_s
        == pytest.approx(
            0.0,
            abs=1.0e-12,
        )
    )


def test_independent_demand_identifies_demand_surge():
    inflow = [0.095] * 10

    demand_observed = [0.105] * 10
    leak_actual = [0.005] * 10

    total_outflow = [
        demand + leak
        for demand, leak in zip(
            demand_observed,
            leak_actual,
            strict=True,
        )
    ]

    pressure = _pressure_trace(
        initial_pressure_bar_g=7.0,
        inflow_kg_s=inflow,
        total_outflow_kg_s=total_outflow,
    )

    inferred = infer_total_outflow(
        pressure,
        inflow,
        interval_seconds=(
            INTERVAL_SECONDS
        ),
        parameters=PARAMETERS,
    )

    leak, demand = (
        verify_leak_vs_demand(
            inferred,
            reference_total_outflow_kg_s=(
                0.095
            ),
            independent_demand_kg_s=(
                demand_observed
            ),
            reference_demand_kg_s=0.090,
            nominal_leak_kg_s=0.005,
        )
    )

    assert leak.status == "not_supported"

    assert (
        leak.estimated_change_kg_s
        == pytest.approx(
            0.0,
            abs=1.0e-12,
        )
    )

    assert demand.status == "supported"

    assert (
        demand.estimated_change_kg_s
        == pytest.approx(
            0.015,
            abs=1.0e-12,
        )
    )


def test_independent_demand_length_must_match():
    inferred = infer_total_outflow(
        [7.0, 7.0],
        [0.095],
        interval_seconds=(
            INTERVAL_SECONDS
        ),
        parameters=PARAMETERS,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Independent demand length"
        ),
    ):
        verify_leak_vs_demand(
            inferred,
            reference_total_outflow_kg_s=(
                0.095
            ),
            independent_demand_kg_s=[
                0.090,
                0.090,
            ],
            reference_demand_kg_s=0.090,
            nominal_leak_kg_s=0.005,
        )

def test_verifier_rejects_physically_inconsistent_outflow():
    inferred = infer_total_outflow(
        [7.0, 8.0],
        [0.0],
        interval_seconds=(
            INTERVAL_SECONDS
        ),
        parameters=PARAMETERS,
    )

    assert (
        inferred.physically_consistent
        is False
    )

    with pytest.raises(
        ValueError,
        match="not physically consistent",
    ):
        verify_leak_vs_demand(
            inferred,
            reference_total_outflow_kg_s=(
                0.095
            ),
        )


def test_verifier_rejects_negative_implied_leak():
    inferred = infer_total_outflow(
        [7.0, 7.0],
        [0.095],
        interval_seconds=(
            INTERVAL_SECONDS
        ),
        parameters=PARAMETERS,
    )

    with pytest.raises(
        ValueError,
        match="negative leakage",
    ):
        verify_leak_vs_demand(
            inferred,
            reference_total_outflow_kg_s=(
                0.095
            ),
            independent_demand_kg_s=[
                0.110
            ],
            reference_demand_kg_s=0.090,
            nominal_leak_kg_s=0.005,
        )