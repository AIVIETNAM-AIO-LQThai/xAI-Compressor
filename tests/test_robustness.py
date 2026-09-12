import pandas as pd
import pytest

from ml.optimization.robustness import (
    RobustnessScenario,
    evaluate_scenarios,
    replay_schedule,
)
from ml.twin.compressor import CompressorSpec
from ml.twin.physics import TwinParameters


def compressors() -> list[CompressorSpec]:
    return [
        CompressorSpec(
            id="fixed_1",
            kind="fixed",
            max_mass_flow_kg_s=0.060,
            rated_power_kw=13.0,
            idle_power_kw=0.0,
            min_load_fraction=1.0,
            min_on_seconds=0.0,
            min_off_seconds=0.0,
        ),
        CompressorSpec(
            id="vsd_1",
            kind="vsd",
            max_mass_flow_kg_s=0.080,
            rated_power_kw=19.0,
            idle_power_kw=2.0,
            min_load_fraction=0.20,
            min_on_seconds=0.0,
            min_off_seconds=0.0,
        ),
    ]


def schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fixed_1_on": [
                True,
                True,
            ],
            "vsd_1_fraction": [
                0.50,
                0.50,
            ],
        }
    )


def parameters() -> TwinParameters:
    return TwinParameters(
        volume_m3=10.0,
        temperature_k=298.15,
    )


def replay(
    scenario: RobustnessScenario,
    *,
    required_reserve_kg_s: float = 0.01,
):
    return replay_schedule(
        schedule(),
        [0.095, 0.095],
        nominal_leak_mass_flow_kg_s=(
            0.005
        ),
        initial_pressure_bar_g=7.0,
        nominal_parameters=parameters(),
        compressors=compressors(),
        interval_seconds=60.0,
        safety_min_bar_g=6.5,
        safety_max_bar_g=7.5,
        required_reserve_kg_s=(
            required_reserve_kg_s
        ),
        terminal_pressure_min_bar_g=7.0,
        scenario=scenario,
    )


def test_nominal_replay_is_balanced():
    result = replay(
        RobustnessScenario(
            id="nominal"
        )
    )

    assert (
        result.terminal_pressure_bar_g
        == pytest.approx(7.0)
    )

    assert (
        result.minimum_pressure_bar_g
        == pytest.approx(7.0)
    )

    assert result.safety_violation_intervals == 0
    assert result.reserve_violation_intervals == 0
    assert result.terminal_requirement_met


def test_higher_demand_lowers_pressure():
    nominal = replay(
        RobustnessScenario(
            id="nominal"
        )
    )

    stressed = replay(
        RobustnessScenario(
            id="demand_high",
            demand_multiplier=1.10,
        )
    )

    assert (
        stressed.terminal_pressure_bar_g
        < nominal.terminal_pressure_bar_g
    )


def test_higher_leak_lowers_pressure():
    nominal = replay(
        RobustnessScenario(
            id="nominal"
        )
    )

    stressed = replay(
        RobustnessScenario(
            id="leak_high",
            leak_delta_kg_s=0.010,
        )
    )

    assert (
        stressed.terminal_pressure_bar_g
        < nominal.terminal_pressure_bar_g
    )


def test_smaller_receiver_magnifies_drop():
    normal_volume = replay(
        RobustnessScenario(
            id="demand_high",
            demand_multiplier=1.10,
            volume_multiplier=1.0,
        )
    )

    smaller_volume = replay(
        RobustnessScenario(
            id="demand_high_small_volume",
            demand_multiplier=1.10,
            volume_multiplier=0.9,
        )
    )

    assert (
        smaller_volume.terminal_pressure_bar_g
        < normal_volume.terminal_pressure_bar_g
    )


def test_reserve_violation_is_detected():
    result = replay(
        RobustnessScenario(
            id="demand_high",
            demand_multiplier=1.10,
        ),
        required_reserve_kg_s=0.04,
    )

    assert result.reserve_violation_intervals > 0
    assert (
        result.first_reserve_violation_time_seconds
        is not None
    )


def test_duplicate_scenario_ids_rejected():
    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        evaluate_scenarios(
            schedule(),
            [0.095, 0.095],
            nominal_leak_mass_flow_kg_s=0.005,
            initial_pressure_bar_g=7.0,
            nominal_parameters=parameters(),
            compressors=compressors(),
            interval_seconds=60.0,
            safety_min_bar_g=6.5,
            safety_max_bar_g=7.5,
            required_reserve_kg_s=0.01,
            terminal_pressure_min_bar_g=7.0,
            scenarios=[
                RobustnessScenario(
                    id="same"
                ),
                RobustnessScenario(
                    id="same"
                ),
            ],
        )


def test_missing_command_column_rejected():
    bad_schedule = schedule().drop(
        columns=["vsd_1_fraction"]
    )

    with pytest.raises(
        ValueError,
        match="vsd_1_fraction",
    ):
        replay_schedule(
            bad_schedule,
            [0.095, 0.095],
            nominal_leak_mass_flow_kg_s=0.005,
            initial_pressure_bar_g=7.0,
            nominal_parameters=parameters(),
            compressors=compressors(),
            interval_seconds=60.0,
            safety_min_bar_g=6.5,
            safety_max_bar_g=7.5,
            required_reserve_kg_s=0.01,
            terminal_pressure_min_bar_g=7.0,
            scenario=RobustnessScenario(
                id="nominal"
            ),
        )
