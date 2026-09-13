from __future__ import annotations

import pytest

from ml.optimization import (
    OptimizationConfig,
    optimize_schedule,
)
from ml.optimization.robust_scheduler import (
    optimize_robust_schedule,
)
from ml.optimization.runtime_state import CompressorRuntimeState
from ml.reasoning.scenarios import (
    PhysicalScenario,
    PhysicalScenarioSet,
)
from ml.twin.compressor import CompressorSpec
from ml.twin.physics import TwinParameters

PARAMETERS = TwinParameters(
    volume_m3=10.0,
    temperature_k=298.15,
)

COMPRESSORS = [
    CompressorSpec(
        id="fixed_1",
        kind="fixed",
        max_mass_flow_kg_s=0.060,
        rated_power_kw=13.0,
        idle_power_kw=0.0,
        min_load_fraction=1.0,
        min_on_seconds=120.0,
        min_off_seconds=60.0,
    ),
    CompressorSpec(
        id="fixed_2",
        kind="fixed",
        max_mass_flow_kg_s=0.060,
        rated_power_kw=13.0,
        idle_power_kw=0.0,
        min_load_fraction=1.0,
        min_on_seconds=120.0,
        min_off_seconds=60.0,
    ),
    CompressorSpec(
        id="vsd_1",
        kind="vsd",
        max_mass_flow_kg_s=0.080,
        rated_power_kw=19.0,
        idle_power_kw=2.0,
        min_load_fraction=0.20,
        min_on_seconds=60.0,
        min_off_seconds=30.0,
    ),
]

CONFIG = OptimizationConfig(
    interval_seconds=60.0,
    reserve_mass_flow_kg_s=0.010,
    startup_penalty_kwh=0.050,
    overpressure_penalty_kwh_per_bar_hour=0.10,
    terminal_pressure_min_bar_g=7.0,
    time_limit_seconds=30.0,
    mip_rel_gap=0.001,
)


def _scenario_set(
    values: list[float],
) -> PhysicalScenarioSet:
    scenarios = tuple(
        PhysicalScenario(
            id=f"scenario_{index}",
            total_outflow_kg_s=value,
            demand_kg_s=None,
            leak_kg_s=None,
            allocation_identifiable=False,
            source="test",
            evidence_class=(
                "PHYSICS_MODEL_INFERENCE"
            ),
            causal_claim=False,
        )
        for index, value in enumerate(
            values
        )
    )

    return PhysicalScenarioSet(
        scenarios=scenarios,
        allocation_identifiable=False,
        evidence_class=(
            "PHYSICS_MODEL_INFERENCE"
        ),
        causal_claim=False,
    )


def test_shared_schedule_is_safe_for_all_scenarios():
    result = optimize_robust_schedule(
        _scenario_set(
            [0.106, 0.110, 0.114]
        ),
        horizon_intervals=10,
        initial_pressure_bar_g=7.0,
        parameters=PARAMETERS,
        compressors=COMPRESSORS,
        target_bar_g=7.0,
        safety_min_bar_g=6.5,
        safety_max_bar_g=7.5,
        config=CONFIG,
    )

    assert result.scenario_count == 3
    assert (
        result.method
        == "shared_action_robust_milp"
    )
    assert result.causal_claim is False

    assert not result.schedule[
        "robust_safety_violation"
    ].any()

    assert (
        result.schedule[
            "minimum_reserve_kg_s"
        ].min()
        >= CONFIG.reserve_mass_flow_kg_s
        - 1.0e-8
    )

    pressure_values = (
        result.scenario_pressures
        .drop(
            columns=["time_seconds"]
        )
    )

    assert (
        pressure_values
        .min()
        .min()
        >= 6.5 - 1.0e-8
    )

    assert (
        pressure_values
        .max()
        .max()
        <= 7.5 + 1.0e-8
    )

    terminal = (
        pressure_values.iloc[-1]
    )

    assert (
        terminal.min()
        >= CONFIG
        .terminal_pressure_min_bar_g
        - 1.0e-8
    )


def test_single_scenario_matches_nominal_energy():
    robust = optimize_robust_schedule(
        _scenario_set([0.110]),
        horizon_intervals=10,
        initial_pressure_bar_g=7.0,
        parameters=PARAMETERS,
        compressors=COMPRESSORS,
        target_bar_g=7.0,
        safety_min_bar_g=6.5,
        safety_max_bar_g=7.5,
        config=CONFIG,
    )

    nominal = optimize_schedule(
        [0.110] * 10,
        leak_mass_flow_kg_s=0.0,
        initial_pressure_bar_g=7.0,
        parameters=PARAMETERS,
        compressors=COMPRESSORS,
        target_bar_g=7.0,
        safety_min_bar_g=6.5,
        safety_max_bar_g=7.5,
        config=CONFIG,
    )

    assert robust.energy_kwh == pytest.approx(
        nominal.energy_kwh,
        abs=1.0e-7,
    )

    assert (
        robust.startup_count
        == nominal.startup_count
    )


def test_shared_schedule_uses_same_commands_for_every_scenario():
    result = optimize_robust_schedule(
        _scenario_set(
            [0.106, 0.110, 0.114]
        ),
        horizon_intervals=10,
        initial_pressure_bar_g=7.0,
        parameters=PARAMETERS,
        compressors=COMPRESSORS,
        target_bar_g=7.0,
        safety_min_bar_g=6.5,
        safety_max_bar_g=7.5,
        config=CONFIG,
    )

    assert len(
        result.schedule
    ) == 10

    for column in (
        "fixed_1_on",
        "fixed_2_on",
        "vsd_1_fraction",
    ):
        assert column in (
            result.schedule.columns
        )

    assert set(
        result.scenario_pressures.columns
    ) == {
        "time_seconds",
        "scenario_0",
        "scenario_1",
        "scenario_2",
    }


def test_impossible_shared_scenario_set_is_rejected():
    with pytest.raises(
        RuntimeError,
        match=(
            "Robust MILP optimization failed"
        ),
    ):
        optimize_robust_schedule(
            _scenario_set([0.210]),
            horizon_intervals=5,
            initial_pressure_bar_g=7.0,
            parameters=PARAMETERS,
            compressors=COMPRESSORS,
            target_bar_g=7.0,
            safety_min_bar_g=6.5,
            safety_max_bar_g=7.5,
            config=CONFIG,
        )

def test_remaining_min_on_is_preserved():
    runtime_state = (
        CompressorRuntimeState(
            compressor_id="fixed_1",
            is_on=True,
            seconds_in_state=60.0,
        ),
        CompressorRuntimeState(
            compressor_id="fixed_2",
            is_on=False,
            seconds_in_state=60.0,
        ),
        CompressorRuntimeState(
            compressor_id="vsd_1",
            is_on=False,
            seconds_in_state=30.0,
        ),
    )

    result = optimize_robust_schedule(
        _scenario_set([0.030]),
        horizon_intervals=5,
        initial_runtime_state=(
            runtime_state
        ),
        initial_pressure_bar_g=7.0,
        parameters=PARAMETERS,
        compressors=COMPRESSORS,
        target_bar_g=7.0,
        safety_min_bar_g=6.5,
        safety_max_bar_g=7.5,
        config=CONFIG,
    )

    first = result.schedule.iloc[0]

    assert bool(
        first["fixed_1_on"]
    ) is True

    assert bool(
        first["fixed_1_start"]
    ) is False


def test_remaining_min_off_is_preserved():
    runtime_state = (
        CompressorRuntimeState(
            compressor_id="fixed_1",
            is_on=False,
            seconds_in_state=0.0,
        ),
        CompressorRuntimeState(
            compressor_id="fixed_2",
            is_on=False,
            seconds_in_state=60.0,
        ),
        CompressorRuntimeState(
            compressor_id="vsd_1",
            is_on=False,
            seconds_in_state=30.0,
        ),
    )

    result = optimize_robust_schedule(
        _scenario_set([0.110]),
        horizon_intervals=5,
        initial_runtime_state=(
            runtime_state
        ),
        initial_pressure_bar_g=7.0,
        parameters=PARAMETERS,
        compressors=COMPRESSORS,
        target_bar_g=7.0,
        safety_min_bar_g=6.5,
        safety_max_bar_g=7.5,
        config=CONFIG,
    )

    first = result.schedule.iloc[0]

    assert bool(
        first["fixed_1_on"]
    ) is False