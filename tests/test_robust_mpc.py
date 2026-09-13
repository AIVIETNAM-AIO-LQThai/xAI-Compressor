from __future__ import annotations

import pytest

from ml.optimization import OptimizationConfig
from ml.optimization.robust_mpc import (
    select_robust_first_action,
)
from ml.optimization.robust_scheduler import (
    optimize_robust_schedule,
)
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
        for index, value in enumerate(values)
    )

    return PhysicalScenarioSet(
        scenarios=scenarios,
        allocation_identifiable=False,
        evidence_class=(
            "PHYSICS_MODEL_INFERENCE"
        ),
        causal_claim=False,
    )


def test_robust_action_is_safe_across_scenarios():
    result = select_robust_first_action(
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

    selected = result.selected

    assert selected.robust_safe is True
    assert (
        selected.worst_case_min_pressure_bar_g
        >= 6.5
    )
    assert (
        selected.worst_case_max_pressure_bar_g
        <= 7.5
    )
    assert (
        selected.minimum_reserve_kg_s
        >= CONFIG.reserve_mass_flow_kg_s
        - 1.0e-9
    )
    assert result.scenario_count == 3
    assert result.valid_for_seconds == 60.0
    assert result.causal_claim is False
    assert (
        result.method
        == (
            "shared_action_robust_milp_"
            "with_one_step_safety_gate"
        )
    )


def test_nonidentifiable_allocation_is_accepted():
    scenario_set = _scenario_set(
        [0.106, 0.110, 0.114]
    )

    assert (
        scenario_set.allocation_identifiable
        is False
    )

    result = select_robust_first_action(
        scenario_set,
        horizon_intervals=10,
        initial_pressure_bar_g=7.0,
        parameters=PARAMETERS,
        compressors=COMPRESSORS,
        target_bar_g=7.0,
        safety_min_bar_g=6.5,
        safety_max_bar_g=7.5,
        config=CONFIG,
    )

    assert result.selected.robust_safe is True


def test_gate_extracts_robust_milp_first_action():
    scenario_set = _scenario_set(
        [0.106, 0.110, 0.114]
    )

    robust = optimize_robust_schedule(
        scenario_set,
        horizon_intervals=10,
        initial_pressure_bar_g=7.0,
        parameters=PARAMETERS,
        compressors=COMPRESSORS,
        target_bar_g=7.0,
        safety_min_bar_g=6.5,
        safety_max_bar_g=7.5,
        config=CONFIG,
    )

    gated = select_robust_first_action(
        scenario_set,
        horizon_intervals=10,
        initial_pressure_bar_g=7.0,
        parameters=PARAMETERS,
        compressors=COMPRESSORS,
        target_bar_g=7.0,
        safety_min_bar_g=6.5,
        safety_max_bar_g=7.5,
        config=CONFIG,
    )

    first = robust.schedule.iloc[0]

    expected = (
        (
            "fixed_1",
            1.0
            if bool(first["fixed_1_on"])
            else 0.0,
        ),
        (
            "fixed_2",
            1.0
            if bool(first["fixed_2_on"])
            else 0.0,
        ),
        (
            "vsd_1",
            round(
                float(
                    first["vsd_1_fraction"]
                ),
                12,
            ),
        ),
    )

    assert gated.selected.commands == expected

    assert (
        gated.robust_horizon_energy_kwh
        == pytest.approx(
            robust.energy_kwh
        )
    )

    assert (
        gated.robust_horizon_startup_count
        == robust.startup_count
    )


def test_impossible_scenarios_are_rejected():
    with pytest.raises(
        RuntimeError,
        match=(
            "Robust MILP optimization failed"
        ),
    ):
        select_robust_first_action(
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
