import pytest

from ml.twin.physics import (
    TwinParameters,
    absolute_pa_to_bar_g,
    bar_g_to_absolute_pa,
    mass_in_receiver,
    pressure_rate_pa_per_s,
    step_pressure,
)


def test_pressure_conversion_round_trip():
    pressure_bar_g = 7.0

    pressure_pa = bar_g_to_absolute_pa(
        pressure_bar_g
    )

    recovered = absolute_pa_to_bar_g(
        pressure_pa
    )

    assert recovered == pytest.approx(
        pressure_bar_g
    )


def test_balanced_mass_flow_keeps_pressure():
    parameters = TwinParameters(
        volume_m3=10.0,
        temperature_k=298.15,
    )

    initial_pressure = (
        bar_g_to_absolute_pa(7.0)
    )

    next_pressure = step_pressure(
        pressure_pa=initial_pressure,
        mass_flow_in_kg_s=1.0,
        demand_mass_flow_kg_s=0.8,
        leak_mass_flow_kg_s=0.2,
        timestep_seconds=10.0,
        parameters=parameters,
    )

    assert next_pressure == pytest.approx(
        initial_pressure
    )


def test_positive_net_flow_increases_pressure():
    parameters = TwinParameters(
        volume_m3=10.0,
        temperature_k=298.15,
    )

    rate = pressure_rate_pa_per_s(
        mass_flow_in_kg_s=1.0,
        demand_mass_flow_kg_s=0.5,
        leak_mass_flow_kg_s=0.1,
        parameters=parameters,
    )

    assert rate > 0.0


def test_negative_net_flow_decreases_pressure():
    parameters = TwinParameters(
        volume_m3=10.0,
        temperature_k=298.15,
    )

    initial_pressure = (
        bar_g_to_absolute_pa(7.0)
    )

    next_pressure = step_pressure(
        pressure_pa=initial_pressure,
        mass_flow_in_kg_s=0.5,
        demand_mass_flow_kg_s=0.8,
        leak_mass_flow_kg_s=0.1,
        timestep_seconds=5.0,
        parameters=parameters,
    )

    assert next_pressure < initial_pressure


def test_pressure_change_matches_mass_balance():
    parameters = TwinParameters(
        volume_m3=10.0,
        temperature_k=298.15,
    )

    initial_pressure = (
        bar_g_to_absolute_pa(7.0)
    )

    initial_mass = mass_in_receiver(
        initial_pressure,
        parameters,
    )

    timestep = 4.0
    net_mass_flow = 0.25

    next_pressure = step_pressure(
        pressure_pa=initial_pressure,
        mass_flow_in_kg_s=0.75,
        demand_mass_flow_kg_s=0.4,
        leak_mass_flow_kg_s=0.1,
        timestep_seconds=timestep,
        parameters=parameters,
    )

    final_mass = mass_in_receiver(
        next_pressure,
        parameters,
    )

    expected_mass = (
        initial_mass
        + net_mass_flow * timestep
    )

    assert final_mass == pytest.approx(
        expected_mass
    )


def test_negative_flow_is_rejected():
    parameters = TwinParameters(
        volume_m3=10.0,
        temperature_k=298.15,
    )

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        pressure_rate_pa_per_s(
            mass_flow_in_kg_s=-1.0,
            demand_mass_flow_kg_s=0.5,
            leak_mass_flow_kg_s=0.0,
            parameters=parameters,
        )