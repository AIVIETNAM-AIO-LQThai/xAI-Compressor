from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

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
class RobustnessScenario:
    id: str
    demand_multiplier: float = 1.0
    leak_delta_kg_s: float = 0.0
    volume_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError(
                "Scenario id cannot be empty."
            )

        if self.demand_multiplier <= 0.0:
            raise ValueError(
                "demand_multiplier must be positive."
            )

        if self.volume_multiplier <= 0.0:
            raise ValueError(
                "volume_multiplier must be positive."
            )


@dataclass(frozen=True)
class ScheduleReplayResult:
    scenario: RobustnessScenario
    trajectory: pd.DataFrame
    energy_kwh: float
    minimum_pressure_bar_g: float
    maximum_pressure_bar_g: float
    terminal_pressure_bar_g: float
    minimum_safety_margin_bar: float
    safety_violation_intervals: int
    reserve_violation_intervals: int
    minimum_reserve_available_kg_s: float
    first_safety_violation_time_seconds: (
        float | None
    )
    first_reserve_violation_time_seconds: (
        float | None
    )
    terminal_requirement_met: bool

    @property
    def safe(self) -> bool:
        return (
            self.safety_violation_intervals == 0
            and self.reserve_violation_intervals == 0
            and self.terminal_requirement_met
        )

    def summary_dict(self) -> dict:
        return {
            "scenario": self.scenario.id,
            "demand_multiplier": (
                self.scenario.demand_multiplier
            ),
            "leak_delta_kg_s": (
                self.scenario.leak_delta_kg_s
            ),
            "volume_multiplier": (
                self.scenario.volume_multiplier
            ),
            "energy_kwh": self.energy_kwh,
            "minimum_pressure_bar_g": (
                self.minimum_pressure_bar_g
            ),
            "maximum_pressure_bar_g": (
                self.maximum_pressure_bar_g
            ),
            "terminal_pressure_bar_g": (
                self.terminal_pressure_bar_g
            ),
            "minimum_safety_margin_bar": (
                self.minimum_safety_margin_bar
            ),
            "safety_violation_intervals": (
                self.safety_violation_intervals
            ),
            "reserve_violation_intervals": (
                self.reserve_violation_intervals
            ),
            "minimum_reserve_available_kg_s": (
                self.minimum_reserve_available_kg_s
            ),
            "first_safety_violation_time_seconds": (
                self.first_safety_violation_time_seconds
            ),
            "first_reserve_violation_time_seconds": (
                self.first_reserve_violation_time_seconds
            ),
            "terminal_requirement_met": (
                self.terminal_requirement_met
            ),
            "safe": self.safe,
            "evidence_class": "simulated",
        }


def _command_for_spec(
    row: pd.Series,
    spec: CompressorSpec,
    *,
    tolerance: float,
) -> float:
    if spec.kind == "fixed":
        column = f"{spec.id}_on"

        if column not in row.index:
            raise ValueError(
                "Schedule is missing compressor "
                f"column: {column}"
            )

        return (
            1.0
            if bool(row[column])
            else 0.0
        )

    column = f"{spec.id}_fraction"

    if column not in row.index:
        raise ValueError(
            "Schedule is missing compressor "
            f"column: {column}"
        )

    command = float(row[column])

    if abs(command) <= tolerance:
        return 0.0

    if abs(command - 1.0) <= tolerance:
        return 1.0

    if (
        command < spec.min_load_fraction
        and (
            spec.min_load_fraction - command
            <= tolerance
        )
    ):
        return spec.min_load_fraction

    return command


def replay_schedule(
    schedule: pd.DataFrame,
    nominal_demand_profile_kg_s: Sequence[float],
    *,
    nominal_leak_mass_flow_kg_s: float,
    initial_pressure_bar_g: float,
    nominal_parameters: TwinParameters,
    compressors: list[CompressorSpec],
    interval_seconds: float,
    safety_min_bar_g: float,
    safety_max_bar_g: float,
    required_reserve_kg_s: float,
    terminal_pressure_min_bar_g: float,
    scenario: RobustnessScenario,
    numerical_tolerance: float = 1.0e-9,
) -> ScheduleReplayResult:
    if schedule.empty:
        raise ValueError(
            "schedule cannot be empty."
        )

    if interval_seconds <= 0.0:
        raise ValueError(
            "interval_seconds must be positive."
        )

    if numerical_tolerance < 0.0:
        raise ValueError(
            "numerical_tolerance "
            "cannot be negative."
        )

    if required_reserve_kg_s < 0.0:
        raise ValueError(
            "required_reserve_kg_s "
            "cannot be negative."
        )

    if nominal_leak_mass_flow_kg_s < 0.0:
        raise ValueError(
            "nominal_leak_mass_flow_kg_s "
            "cannot be negative."
        )

    demand = np.asarray(
        list(nominal_demand_profile_kg_s),
        dtype=float,
    )

    if demand.size != len(schedule):
        raise ValueError(
            "Demand profile length must "
            "match schedule length."
        )

    if not np.isfinite(demand).all():
        raise ValueError(
            "Demand profile contains "
            "non-finite values."
        )

    if (demand < 0.0).any():
        raise ValueError(
            "Demand values cannot be negative."
        )

    actual_demand = (
        demand
        * scenario.demand_multiplier
    )

    actual_leak = (
        nominal_leak_mass_flow_kg_s
        + scenario.leak_delta_kg_s
    )

    if actual_leak < 0.0:
        raise ValueError(
            "Scenario produces a negative "
            "actual leak rate."
        )

    parameters = replace(
        nominal_parameters,
        volume_m3=(
            nominal_parameters.volume_m3
            * scenario.volume_multiplier
        ),
    )

    pressure_pa = bar_g_to_absolute_pa(
        initial_pressure_bar_g,
        ambient_pressure_pa=(
            parameters.ambient_pressure_pa
        ),
    )

    rows: list[dict] = []
    cumulative_energy_kwh = 0.0
    pressure_values = [
        float(initial_pressure_bar_g)
    ]

    first_safety_violation = None
    first_reserve_violation = None

    for interval_index, (
        _,
        schedule_row,
    ) in enumerate(
        schedule.iterrows()
    ):
        commands = {
            spec.id: _command_for_spec(
                schedule_row,
                spec,
                tolerance=numerical_tolerance,
            )
            for spec in compressors
        }

        operating_point = (
            fleet_operating_point(
                compressors,
                load_commands=commands,
            )
        )

        pressure_start_bar_g = (
            absolute_pa_to_bar_g(
                pressure_pa,
                ambient_pressure_pa=(
                    parameters
                    .ambient_pressure_pa
                ),
            )
        )

        next_pressure_pa = step_pressure(
            pressure_pa=pressure_pa,
            mass_flow_in_kg_s=(
                operating_point
                .total_mass_flow_kg_s
            ),
            demand_mass_flow_kg_s=float(
                actual_demand[
                    interval_index
                ]
            ),
            leak_mass_flow_kg_s=(
                actual_leak
            ),
            timestep_seconds=(
                interval_seconds
            ),
            parameters=parameters,
        )

        pressure_end_bar_g = (
            absolute_pa_to_bar_g(
                next_pressure_pa,
                ambient_pressure_pa=(
                    parameters
                    .ambient_pressure_pa
                ),
            )
        )

        online_capacity = sum(
            spec.max_mass_flow_kg_s
            for spec in compressors
            if commands[spec.id] > 0.0
        )

        reserve_available = (
            online_capacity
            - float(
                actual_demand[
                    interval_index
                ]
            )
            - actual_leak
        )

        lower_violation = (
            pressure_end_bar_g
            < (
                safety_min_bar_g
                - numerical_tolerance
            )
        )

        upper_violation = (
            pressure_end_bar_g
            > (
                safety_max_bar_g
                + numerical_tolerance
            )
        )

        safety_violation = (
            lower_violation
            or upper_violation
        )

        reserve_violation = (
            reserve_available
            < (
                required_reserve_kg_s
                - numerical_tolerance
            )
        )

        time_seconds = (
            (interval_index + 1)
            * interval_seconds
        )

        if (
            safety_violation
            and first_safety_violation is None
        ):
            first_safety_violation = float(
                time_seconds
            )

        if (
            reserve_violation
            and first_reserve_violation is None
        ):
            first_reserve_violation = float(
                time_seconds
            )

        interval_energy_kwh = (
            operating_point.total_power_kw
            * interval_seconds
            / 3600.0
        )

        cumulative_energy_kwh += (
            interval_energy_kwh
        )

        rows.append(
            {
                "interval_index":
                    interval_index,
                "time_seconds":
                    float(time_seconds),
                "demand_kg_s":
                    float(
                        actual_demand[
                            interval_index
                        ]
                    ),
                "leak_kg_s":
                    float(actual_leak),
                "volume_m3":
                    float(
                        parameters.volume_m3
                    ),
                "mass_flow_in_kg_s":
                    float(
                        operating_point
                        .total_mass_flow_kg_s
                    ),
                "power_kw":
                    float(
                        operating_point
                        .total_power_kw
                    ),
                "energy_kwh":
                    float(
                        interval_energy_kwh
                    ),
                "pressure_start_bar_g":
                    float(
                        pressure_start_bar_g
                    ),
                "pressure_end_bar_g":
                    float(
                        pressure_end_bar_g
                    ),
                "online_capacity_kg_s":
                    float(online_capacity),
                "reserve_available_kg_s":
                    float(reserve_available),
                "safety_violation":
                    bool(safety_violation),
                "reserve_violation":
                    bool(reserve_violation),
            }
        )

        pressure_pa = next_pressure_pa
        pressure_values.append(
            float(pressure_end_bar_g)
        )

    trajectory = pd.DataFrame(rows)

    safety_violation_intervals = int(
        trajectory[
            "safety_violation"
        ].sum()
    )

    reserve_violation_intervals = int(
        trajectory[
            "reserve_violation"
        ].sum()
    )

    terminal_pressure = float(
        pressure_values[-1]
    )

    return ScheduleReplayResult(
        scenario=scenario,
        trajectory=trajectory,
        energy_kwh=float(
            cumulative_energy_kwh
        ),
        minimum_pressure_bar_g=float(
            min(pressure_values)
        ),
        maximum_pressure_bar_g=float(
            max(pressure_values)
        ),
        terminal_pressure_bar_g=(
            terminal_pressure
        ),
        minimum_safety_margin_bar=float(
            min(pressure_values)
            - safety_min_bar_g
        ),
        safety_violation_intervals=(
            safety_violation_intervals
        ),
        reserve_violation_intervals=(
            reserve_violation_intervals
        ),
        minimum_reserve_available_kg_s=float(
            trajectory[
                "reserve_available_kg_s"
            ].min()
        ),
        first_safety_violation_time_seconds=(
            first_safety_violation
        ),
        first_reserve_violation_time_seconds=(
            first_reserve_violation
        ),
        terminal_requirement_met=(
            terminal_pressure
            >= (
                terminal_pressure_min_bar_g
                - numerical_tolerance
            )
        ),
    )


def evaluate_scenarios(
    schedule: pd.DataFrame,
    nominal_demand_profile_kg_s: Sequence[float],
    *,
    nominal_leak_mass_flow_kg_s: float,
    initial_pressure_bar_g: float,
    nominal_parameters: TwinParameters,
    compressors: list[CompressorSpec],
    interval_seconds: float,
    safety_min_bar_g: float,
    safety_max_bar_g: float,
    required_reserve_kg_s: float,
    terminal_pressure_min_bar_g: float,
    scenarios: Sequence[
        RobustnessScenario
    ],
    numerical_tolerance: float = 1.0e-9,
) -> list[ScheduleReplayResult]:
    scenario_list = list(scenarios)

    ids = [
        scenario.id
        for scenario in scenario_list
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Robustness scenario ids "
            "must be unique."
        )

    return [
        replay_schedule(
            schedule,
            nominal_demand_profile_kg_s,
            nominal_leak_mass_flow_kg_s=(
                nominal_leak_mass_flow_kg_s
            ),
            initial_pressure_bar_g=(
                initial_pressure_bar_g
            ),
            nominal_parameters=(
                nominal_parameters
            ),
            compressors=compressors,
            interval_seconds=(
                interval_seconds
            ),
            safety_min_bar_g=(
                safety_min_bar_g
            ),
            safety_max_bar_g=(
                safety_max_bar_g
            ),
            required_reserve_kg_s=(
                required_reserve_kg_s
            ),
            terminal_pressure_min_bar_g=(
                terminal_pressure_min_bar_g
            ),
            scenario=scenario,
            numerical_tolerance=(
                numerical_tolerance
            ),
        )
        for scenario in scenario_list
    ]
