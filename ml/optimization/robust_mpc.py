from __future__ import annotations

from dataclasses import dataclass

from ml.optimization.robust_scheduler import (
    optimize_robust_schedule,
)
from ml.optimization.runtime_state import (
    CompressorRuntimeState,
    advance_runtime_state,
    resolve_runtime_state,
)
from ml.optimization.scheduler import OptimizationConfig
from ml.reasoning.scenarios import PhysicalScenarioSet
from ml.twin.compressor import (
    CompressorSpec,
    fleet_operating_point,
)
from ml.twin.physics import (
    TwinParameters,
    absolute_pa_to_bar_g,
    bar_g_to_absolute_pa,
    step_pressure,
)


@dataclass(frozen=True)
class RobustActionEvaluation:
    commands: tuple[tuple[str, float], ...]
    source_scenarios: tuple[str, ...]
    interval_energy_kwh: float
    startup_count: int
    worst_case_min_pressure_bar_g: float
    worst_case_max_pressure_bar_g: float
    minimum_reserve_kg_s: float
    objective_value: float
    robust_safe: bool


@dataclass(frozen=True)
class RobustFirstActionResult:
    selected: RobustActionEvaluation
    candidates: tuple[RobustActionEvaluation, ...]
    scenario_count: int
    horizon_intervals: int
    valid_for_seconds: float
    robust_horizon_objective_value: float
    robust_horizon_energy_kwh: float
    robust_horizon_startup_count: int
    method: str
    evidence_class: str
    causal_claim: bool
    next_runtime_state: tuple[CompressorRuntimeState, ...]


def _commands_from_first_row(
    row,
    compressors: list[CompressorSpec],
) -> tuple[tuple[str, float], ...]:
    commands: list[tuple[str, float]] = []

    for spec in compressors:
        if spec.kind == "fixed":
            value = 1.0 if bool(
                row[f"{spec.id}_on"]
            ) else 0.0
        else:
            value = float(
                row[f"{spec.id}_fraction"]
            )

            if value <= 1.0e-10:
                value = 0.0
            elif (
                value < spec.min_load_fraction
                and spec.min_load_fraction - value
                <= 1.0e-8
            ):
                value = spec.min_load_fraction
            elif (
                value > 1.0
                and value - 1.0 <= 1.0e-8
            ):
                value = 1.0

        commands.append(
            (spec.id, round(value, 12))
        )

    return tuple(commands)


def _startup_count_from_first_row(
    row,
    compressors: list[CompressorSpec],
) -> int:
    return sum(
        int(bool(row[f"{spec.id}_start"]))
        for spec in compressors
    )


def _online_capacity(
    commands: dict[str, float],
    compressors: list[CompressorSpec],
) -> float:
    return sum(
        spec.max_mass_flow_kg_s
        for spec in compressors
        if commands[spec.id] > 0.0
    )


def _evaluate_first_action(
    commands_tuple: tuple[tuple[str, float], ...],
    *,
    startup_count: int,
    scenario_set: PhysicalScenarioSet,
    initial_pressure_bar_g: float,
    parameters: TwinParameters,
    compressors: list[CompressorSpec],
    target_bar_g: float,
    safety_min_bar_g: float,
    safety_max_bar_g: float,
    config: OptimizationConfig,
) -> RobustActionEvaluation:
    commands = dict(commands_tuple)

    operating_point = fleet_operating_point(
        compressors,
        load_commands=commands,
    )

    online_capacity = _online_capacity(
        commands,
        compressors,
    )

    initial_pressure_pa = (
        bar_g_to_absolute_pa(
            initial_pressure_bar_g,
            ambient_pressure_pa=(
                parameters.ambient_pressure_pa
            ),
        )
    )

    pressures: list[float] = []
    reserves: list[float] = []
    robust_safe = True

    for scenario in scenario_set.scenarios:
        next_pressure_pa = step_pressure(
            pressure_pa=initial_pressure_pa,
            mass_flow_in_kg_s=(
                operating_point.total_mass_flow_kg_s
            ),
            demand_mass_flow_kg_s=(
                scenario.total_outflow_kg_s
            ),
            leak_mass_flow_kg_s=0.0,
            timestep_seconds=config.interval_seconds,
            parameters=parameters,
        )

        next_pressure = absolute_pa_to_bar_g(
            next_pressure_pa,
            ambient_pressure_pa=(
                parameters.ambient_pressure_pa
            ),
        )

        reserve = (
            online_capacity
            - scenario.total_outflow_kg_s
        )

        pressures.append(float(next_pressure))
        reserves.append(float(reserve))

        if not (
            safety_min_bar_g
            <= next_pressure
            <= safety_max_bar_g
        ):
            robust_safe = False

        if (
            reserve
            < config.reserve_mass_flow_kg_s
            - 1.0e-9
        ):
            robust_safe = False

    interval_hours = (
        config.interval_seconds / 3600.0
    )

    interval_energy = (
        operating_point.total_power_kw
        * interval_hours
    )

    worst_overpressure = max(
        0.0,
        max(pressures) - target_bar_g,
    )

    objective = (
        interval_energy
        + startup_count
        * config.startup_penalty_kwh
        + worst_overpressure
        * config.overpressure_penalty_kwh_per_bar_hour
        * interval_hours
    )

    return RobustActionEvaluation(
        commands=commands_tuple,
        source_scenarios=tuple(
            scenario.id
            for scenario in scenario_set.scenarios
        ),
        interval_energy_kwh=float(interval_energy),
        startup_count=int(startup_count),
        worst_case_min_pressure_bar_g=float(
            min(pressures)
        ),
        worst_case_max_pressure_bar_g=float(
            max(pressures)
        ),
        minimum_reserve_kg_s=float(
            min(reserves)
        ),
        objective_value=float(objective),
        robust_safe=robust_safe,
    )


def select_robust_first_action(
    scenario_set: PhysicalScenarioSet,
    *,
    horizon_intervals: int,
    initial_runtime_state: (
        tuple[
            CompressorRuntimeState,
            ...
        ]
        | None
    ) = None,
    initial_pressure_bar_g: float,
    parameters: TwinParameters,
    compressors: list[CompressorSpec],
    target_bar_g: float,
    safety_min_bar_g: float,
    safety_max_bar_g: float,
    config: OptimizationConfig,
    
) -> RobustFirstActionResult:
    if horizon_intervals <= 0:
        raise ValueError(
            "horizon_intervals must be positive."
        )

    resolved_runtime_state = (
        resolve_runtime_state(
            compressors,
            initial_runtime_state,
        )
    )

    robust_result = optimize_robust_schedule(
        scenario_set,
        horizon_intervals=horizon_intervals,
        initial_runtime_state=resolved_runtime_state,
        initial_pressure_bar_g=(
            initial_pressure_bar_g
        ),
        parameters=parameters,
        compressors=compressors,
        target_bar_g=target_bar_g,
        safety_min_bar_g=safety_min_bar_g,
        safety_max_bar_g=safety_max_bar_g,
        config=config,
    )

    first_row = robust_result.schedule.iloc[0]

    commands = _commands_from_first_row(
        first_row,
        compressors,
    )

    startup_count = (
        _startup_count_from_first_row(
            first_row,
            compressors,
        )
    )

    evaluation = _evaluate_first_action(
        commands,
        startup_count=startup_count,
        scenario_set=scenario_set,
        initial_pressure_bar_g=(
            initial_pressure_bar_g
        ),
        parameters=parameters,
        compressors=compressors,
        target_bar_g=target_bar_g,
        safety_min_bar_g=safety_min_bar_g,
        safety_max_bar_g=safety_max_bar_g,
        config=config,
    )

    if not evaluation.robust_safe:
        raise RuntimeError(
            "Robust MILP first action failed "
            "independent one-step safety gate."
        )

    next_runtime_state = (
        advance_runtime_state(
            resolved_runtime_state,
            commands,
            interval_seconds=(
                config.interval_seconds
            ),
            compressors=compressors,
        )
    )

    return RobustFirstActionResult(
        selected=evaluation,
        candidates=(evaluation,),
        scenario_count=len(
            scenario_set.scenarios
        ),
        horizon_intervals=horizon_intervals,
        valid_for_seconds=config.interval_seconds,
        robust_horizon_objective_value=(
            robust_result.objective_value
        ),
        robust_horizon_energy_kwh=(
            robust_result.energy_kwh
        ),
        robust_horizon_startup_count=(
            robust_result.startup_count
        ),
        method=(
            "shared_action_robust_milp_"
            "with_one_step_safety_gate"
        ),
        evidence_class="SIMULATED",
        causal_claim=False,
        next_runtime_state=next_runtime_state,
    )
