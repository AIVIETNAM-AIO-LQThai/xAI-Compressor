from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from ml.twin.compressor import CompressorSpec, fleet_operating_point
from ml.twin.physics import (
    TwinParameters,
    absolute_pa_to_bar_g,
    bar_g_to_absolute_pa,
    step_pressure,
)


@dataclass(frozen=True)
class BaselineControllerConfig:
    target_bar_g: float
    lower_band_bar_g: float
    upper_band_bar_g: float
    safety_min_bar_g: float
    safety_max_bar_g: float
    pressure_gain_kg_s_per_bar: float

    def __post_init__(self) -> None:
        valid_order = (
            self.safety_min_bar_g
            < self.lower_band_bar_g
            < self.target_bar_g
            < self.upper_band_bar_g
            < self.safety_max_bar_g
        )

        if not valid_order:
            raise ValueError(
                "Pressure limits must satisfy "
                "safety_min < lower < target "
                "< upper < safety_max."
            )

        if (
            self.pressure_gain_kg_s_per_bar
            < 0.0
        ):
            raise ValueError(
                "pressure_gain_kg_s_per_bar "
                "cannot be negative."
            )


@dataclass
class _RuntimeState:
    is_on: bool
    seconds_in_state: float


def _can_turn_on(
    state: _RuntimeState,
    spec: CompressorSpec,
) -> bool:
    return (
        not state.is_on
        and state.seconds_in_state
        >= spec.min_off_seconds
    )


def _can_turn_off(
    state: _RuntimeState,
    spec: CompressorSpec,
) -> bool:
    return (
        state.is_on
        and state.seconds_in_state
        >= spec.min_on_seconds
    )


def simulate_baseline(
    demand_profile_kg_s: Sequence[float],
    *,
    leak_mass_flow_kg_s: float,
    initial_pressure_bar_g: float,
    parameters: TwinParameters,
    compressors: list[CompressorSpec],
    controller_config: (
        BaselineControllerConfig
    ),
    timestep_seconds: float,
) -> pd.DataFrame:
    if timestep_seconds <= 0.0:
        raise ValueError(
            "timestep_seconds "
            "must be positive."
        )

    if leak_mass_flow_kg_s < 0.0:
        raise ValueError(
            "leak_mass_flow_kg_s "
            "cannot be negative."
        )

    demand_values = [
        float(value)
        for value in demand_profile_kg_s
    ]

    if not demand_values:
        raise ValueError(
            "demand_profile_kg_s "
            "cannot be empty."
        )

    if any(
        value < 0.0
        for value in demand_values
    ):
        raise ValueError(
            "Demand values "
            "cannot be negative."
        )

    fixed_specs = [
        spec
        for spec in compressors
        if spec.kind == "fixed"
    ]

    vsd_specs = [
        spec
        for spec in compressors
        if spec.kind == "vsd"
    ]

    if not fixed_specs:
        raise ValueError(
            "Baseline controller requires "
            "at least one fixed compressor."
        )

    if len(vsd_specs) != 1:
        raise ValueError(
            "Baseline controller requires "
            "exactly one VSD compressor."
        )

    ids = {
        spec.id
        for spec in compressors
    }

    if len(ids) != len(compressors):
        raise ValueError(
            "Compressor ids must be unique."
        )

    vsd_spec = vsd_specs[0]

    states = {
        spec.id: _RuntimeState(
            is_on=False,
            seconds_in_state=float(
                spec.min_off_seconds
            ),
        )
        for spec in compressors
    }

    pressure_pa = (
        bar_g_to_absolute_pa(
            initial_pressure_bar_g,
            ambient_pressure_pa=(
                parameters
                .ambient_pressure_pa
            ),
        )
    )

    cumulative_energy_kwh = 0.0
    rows = []

    for step_index, demand in enumerate(
        demand_values
    ):
        pressure_start_bar_g = (
            absolute_pa_to_bar_g(
                pressure_pa,
                ambient_pressure_pa=(
                    parameters
                    .ambient_pressure_pa
                ),
            )
        )

        desired_flow = max(
            0.0,
            demand
            + leak_mass_flow_kg_s
            + (
                controller_config
                .pressure_gain_kg_s_per_bar
                * (
                    controller_config
                    .target_bar_g
                    - pressure_start_bar_g
                )
            ),
        )

        started_ids: set[str] = set()

        if (
            pressure_start_bar_g
            < controller_config
            .lower_band_bar_g
        ):
            for spec in fixed_specs:
                active_fixed_flow = sum(
                    item.max_mass_flow_kg_s
                    for item in fixed_specs
                    if states[
                        item.id
                    ].is_on
                )

                available_with_vsd = (
                    active_fixed_flow
                    + vsd_spec
                    .max_mass_flow_kg_s
                )

                if (
                    desired_flow
                    <= available_with_vsd
                ):
                    break

                state = states[spec.id]

                if _can_turn_on(
                    state,
                    spec,
                ):
                    state.is_on = True
                    state.seconds_in_state = 0.0
                    started_ids.add(
                        spec.id
                    )

        if (
            pressure_start_bar_g
            > controller_config
            .upper_band_bar_g
        ):
            for spec in reversed(
                fixed_specs
            ):
                state = states[spec.id]

                if not _can_turn_off(
                    state,
                    spec,
                ):
                    continue

                remaining_fixed_flow = sum(
                    item.max_mass_flow_kg_s
                    for item in fixed_specs
                    if (
                        states[
                            item.id
                        ].is_on
                        and item.id
                        != spec.id
                    )
                )

                if (
                    desired_flow
                    <= (
                        remaining_fixed_flow
                        + vsd_spec
                        .max_mass_flow_kg_s
                    )
                ):
                    state.is_on = False
                    state.seconds_in_state = 0.0

        fixed_flow = sum(
            spec.max_mass_flow_kg_s
            for spec in fixed_specs
            if states[spec.id].is_on
        )

        residual_flow = max(
            0.0,
            desired_flow - fixed_flow,
        )

        desired_vsd_fraction = min(
            1.0,
            residual_flow
            / vsd_spec.max_mass_flow_kg_s,
        )

        if (
            0.0
            < desired_vsd_fraction
            < vsd_spec.min_load_fraction
        ):
            desired_vsd_fraction = (
                vsd_spec.min_load_fraction
            )

        vsd_state = states[
            vsd_spec.id
        ]

        if desired_vsd_fraction == 0.0:
            if (
                vsd_state.is_on
                and _can_turn_off(
                    vsd_state,
                    vsd_spec,
                )
            ):
                vsd_state.is_on = False
                vsd_state.seconds_in_state = 0.0
                vsd_fraction = 0.0

            elif vsd_state.is_on:
                vsd_fraction = (
                    vsd_spec
                    .min_load_fraction
                )

            else:
                vsd_fraction = 0.0

        else:
            if vsd_state.is_on:
                vsd_fraction = (
                    desired_vsd_fraction
                )

            elif _can_turn_on(
                vsd_state,
                vsd_spec,
            ):
                vsd_state.is_on = True
                vsd_state.seconds_in_state = 0.0

                started_ids.add(
                    vsd_spec.id
                )

                vsd_fraction = (
                    desired_vsd_fraction
                )

            else:
                vsd_fraction = 0.0

        load_commands = {
            spec.id: (
                1.0
                if states[
                    spec.id
                ].is_on
                else 0.0
            )
            for spec in fixed_specs
        }

        load_commands[
            vsd_spec.id
        ] = vsd_fraction

        fleet = fleet_operating_point(
            compressors,
            load_commands=(
                load_commands
            ),
        )

        pressure_pa = step_pressure(
            pressure_pa=pressure_pa,
            mass_flow_in_kg_s=(
                fleet
                .total_mass_flow_kg_s
            ),
            demand_mass_flow_kg_s=(
                demand
            ),
            leak_mass_flow_kg_s=(
                leak_mass_flow_kg_s
            ),
            timestep_seconds=(
                timestep_seconds
            ),
            parameters=parameters,
        )

        pressure_end_bar_g = (
            absolute_pa_to_bar_g(
                pressure_pa,
                ambient_pressure_pa=(
                    parameters
                    .ambient_pressure_pa
                ),
            )
        )

        interval_energy_kwh = (
            fleet.total_power_kw
            * timestep_seconds
            / 3600.0
        )

        cumulative_energy_kwh += (
            interval_energy_kwh
        )

        row = {
            "time_seconds":
                (
                    step_index + 1
                )
                * timestep_seconds,
            "demand_kg_s":
                demand,
            "leak_kg_s":
                leak_mass_flow_kg_s,
            "desired_flow_kg_s":
                desired_flow,
            "pressure_start_bar_g":
                pressure_start_bar_g,
            "pressure_bar_g":
                pressure_end_bar_g,
            "compressor_flow_kg_s":
                fleet
                .total_mass_flow_kg_s,
            "power_kw":
                fleet.total_power_kw,
            "energy_kwh":
                interval_energy_kwh,
            "cumulative_energy_kwh":
                cumulative_energy_kwh,
            "safety_violation":
                (
                    pressure_end_bar_g
                    < controller_config
                    .safety_min_bar_g
                    or pressure_end_bar_g
                    > controller_config
                    .safety_max_bar_g
                ),
        }

        for spec in fixed_specs:
            row[
                f"{spec.id}_on"
            ] = states[
                spec.id
            ].is_on

            row[
                f"{spec.id}_start"
            ] = (
                spec.id
                in started_ids
            )

        row[
            f"{vsd_spec.id}_on"
        ] = vsd_state.is_on

        row[
            f"{vsd_spec.id}_fraction"
        ] = vsd_fraction

        row[
            f"{vsd_spec.id}_start"
        ] = (
            vsd_spec.id
            in started_ids
        )

        rows.append(row)

        for state in states.values():
            state.seconds_in_state += (
                timestep_seconds
            )

    return pd.DataFrame(rows)