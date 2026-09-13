from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from ml.optimization.runtime_state import CompressorRuntimeState, resolve_runtime_state
from ml.optimization.scheduler import OptimizationConfig
from ml.reasoning.scenarios import PhysicalScenarioSet
from ml.twin.compressor import CompressorSpec
from ml.twin.physics import TwinParameters


@dataclass(frozen=True)
class RobustOptimizationResult:
    schedule: pd.DataFrame
    scenario_pressures: pd.DataFrame
    objective_value: float
    energy_kwh: float
    startup_count: int
    worst_overpressure_bar_hours: float
    solver_status: int
    solver_message: str
    scenario_count: int
    initial_compressor_state_assumption: str
    method: str
    evidence_class: str
    causal_claim: bool


def _minimum_intervals(
    duration_seconds: float,
    interval_seconds: float,
) -> int:
    return max(
        1,
        math.ceil(
            duration_seconds
            / interval_seconds
        ),
    )

def _remaining_lock_intervals(
    required_seconds: float,
    elapsed_seconds: float,
    interval_seconds: float,
) -> int:
    remaining_seconds = max(
        0.0,
        required_seconds
        - elapsed_seconds,
    )

    if remaining_seconds <= 0.0:
        return 0

    return math.ceil(
        remaining_seconds
        / interval_seconds
    )


def optimize_robust_schedule(
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
) -> RobustOptimizationResult:
    if horizon_intervals <= 0:
        raise ValueError(
            "horizon_intervals must be positive."
        )

    if not scenario_set.scenarios:
        raise ValueError(
            "scenario_set cannot be empty."
        )

    if not (
        safety_min_bar_g
        < target_bar_g
        < safety_max_bar_g
    ):
        raise ValueError(
            "Pressure limits must satisfy "
            "safety_min < target < safety_max."
        )

    if not (
        safety_min_bar_g
        <= initial_pressure_bar_g
        <= safety_max_bar_g
    ):
        raise ValueError(
            "Initial pressure is outside "
            "the safety bounds."
        )

    if not (
        safety_min_bar_g
        <= config.terminal_pressure_min_bar_g
        <= safety_max_bar_g
    ):
        raise ValueError(
            "terminal_pressure_min_bar_g "
            "is outside the safety bounds."
        )

    scenario_outflows = np.asarray(
        [
            scenario.total_outflow_kg_s
            for scenario in scenario_set.scenarios
        ],
        dtype=float,
    )

    if not np.isfinite(
        scenario_outflows
    ).all():
        raise ValueError(
            "Scenario outflow contains "
            "non-finite values."
        )

    if (
        scenario_outflows < 0.0
    ).any():
        raise ValueError(
            "Scenario outflow cannot "
            "be negative."
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
            "Optimizer requires at least "
            "one fixed compressor."
        )

    if len(vsd_specs) != 1:
        raise ValueError(
            "Optimizer requires exactly "
            "one VSD compressor."
        )

    compressor_ids = {
        spec.id
        for spec in compressors
    }

    if len(compressor_ids) != len(compressors):
        raise ValueError("Compressor ids must be unique.")

    runtime_states = resolve_runtime_state(
        compressors,
        initial_runtime_state,
    )

    runtime_by_id = {
        state.compressor_id: state
        for state in runtime_states
    }

    vsd_spec = vsd_specs[0]
    horizon = horizon_intervals
    interval_seconds = (
        config.interval_seconds
    )
    scenario_count = len(
        scenario_set.scenarios
    )

    blocks: dict[
        str,
        np.ndarray,
    ] = {}
    next_index = 0

    def add_block(
        name: str,
        size: int,
    ) -> np.ndarray:
        nonlocal next_index

        indices = np.arange(
            next_index,
            next_index + size,
            dtype=int,
        )

        blocks[name] = indices
        next_index += size
        return indices

    for spec in fixed_specs:
        add_block(
            f"fixed_on:{spec.id}",
            horizon,
        )
        add_block(
            f"fixed_start:{spec.id}",
            horizon,
        )
        add_block(
            f"fixed_stop:{spec.id}",
            horizon,
        )

    vsd_on = add_block(
        "vsd_on",
        horizon,
    )
    vsd_start = add_block(
        "vsd_start",
        horizon,
    )
    vsd_stop = add_block(
        "vsd_stop",
        horizon,
    )
    vsd_fraction = add_block(
        "vsd_fraction",
        horizon,
    )

    pressure_blocks: list[
        np.ndarray
    ] = []

    for scenario_index in range(
        scenario_count
    ):
        pressure_blocks.append(
            add_block(
                (
                    "pressure:"
                    f"{scenario_index}"
                ),
                horizon + 1,
            )
        )

    worst_overpressure = add_block(
        "worst_overpressure",
        horizon,
    )

    variable_count = next_index

    objective = np.zeros(
        variable_count,
        dtype=float,
    )

    integrality = np.zeros(
        variable_count,
        dtype=int,
    )

    lower_bounds = np.zeros(
        variable_count,
        dtype=float,
    )

    upper_bounds = np.full(
        variable_count,
        np.inf,
        dtype=float,
    )

    interval_hours = (
        interval_seconds / 3600.0
    )

    for spec in fixed_specs:
        on = blocks[
            f"fixed_on:{spec.id}"
        ]
        start = blocks[
            f"fixed_start:{spec.id}"
        ]
        stop = blocks[
            f"fixed_stop:{spec.id}"
        ]

        for indices in (
            on,
            start,
            stop,
        ):
            upper_bounds[
                indices
            ] = 1.0
            integrality[
                indices
            ] = 1

        objective[on] += (
            spec.rated_power_kw
            * interval_hours
        )

        objective[start] += (
            config.startup_penalty_kwh
        )

    for indices in (
        vsd_on,
        vsd_start,
        vsd_stop,
    ):
        upper_bounds[
            indices
        ] = 1.0
        integrality[
            indices
        ] = 1

    upper_bounds[
        vsd_fraction
    ] = 1.0

    objective[vsd_on] += (
        vsd_spec.idle_power_kw
        * interval_hours
    )

    objective[
        vsd_fraction
    ] += (
        (
            vsd_spec.rated_power_kw
            - vsd_spec.idle_power_kw
        )
        * interval_hours
    )

    objective[
        vsd_start
    ] += (
        config.startup_penalty_kwh
    )

    for pressure in pressure_blocks:
        lower_bounds[
            pressure
        ] = safety_min_bar_g

        upper_bounds[
            pressure
        ] = safety_max_bar_g

        lower_bounds[
            pressure[0]
        ] = initial_pressure_bar_g

        upper_bounds[
            pressure[0]
        ] = initial_pressure_bar_g

        lower_bounds[
            pressure[-1]
        ] = max(
            safety_min_bar_g,
            config
            .terminal_pressure_min_bar_g,
        )

    objective[
        worst_overpressure
    ] += (
        config
        .overpressure_penalty_kwh_per_bar_hour
        * interval_hours
    )

    rows: list[
        dict[int, float]
    ] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    def add_constraint(
        coefficients: dict[int, float],
        lower: float = -np.inf,
        upper: float = np.inf,
    ) -> None:
        rows.append(coefficients)
        row_lower.append(lower)
        row_upper.append(upper)

    for spec in fixed_specs:
        on = blocks[
            f"fixed_on:{spec.id}"
        ]
        start = blocks[
            f"fixed_start:{spec.id}"
        ]
        stop = blocks[
            f"fixed_stop:{spec.id}"
        ]

        minimum_on = _minimum_intervals(
            spec.min_on_seconds,
            interval_seconds,
        )

        minimum_off = _minimum_intervals(
            spec.min_off_seconds,
            interval_seconds,
        )

        initial_state = runtime_by_id[spec.id]
        locked_on_intervals = 0
        locked_off_intervals = 0

        if initial_state.is_on:
            locked_on_intervals = (
                _remaining_lock_intervals(
                    spec.min_on_seconds,
                    initial_state.seconds_in_state,
                    interval_seconds,
                )
            )
        else:
            locked_off_intervals = (
                _remaining_lock_intervals(
                    spec.min_off_seconds,
                    initial_state.seconds_in_state,
                    interval_seconds,
                )
            )

        for k in range(
            min(horizon, locked_on_intervals)
        ):
            add_constraint(
                {
                    int(on[k]): 1.0,
                },
                lower=1.0,
                upper=1.0,
            )

        for k in range(
            min(horizon, locked_off_intervals)
        ):
            add_constraint(
                {
                    int(on[k]): 1.0,
                },
                lower=0.0,
                upper=0.0,
            )

        for t in range(horizon):
            transition = {
                int(on[t]): 1.0,
                int(start[t]): -1.0,
                int(stop[t]): 1.0,
            }

            if t > 0:
                transition[
                    int(on[t - 1])
                ] = -1.0

                transition_rhs = 0.0
            else:
                transition_rhs = (
                    1.0
                    if initial_state.is_on
                    else 0.0
                )

            add_constraint(
                transition,
                lower=transition_rhs,
                upper=transition_rhs,
            )

            add_constraint(
                {
                    int(start[t]): 1.0,
                    int(stop[t]): 1.0,
                },
                upper=1.0,
            )

            if (
                t + minimum_on > horizon
            ):
                add_constraint(
                    {
                        int(start[t]): 1.0,
                    },
                    lower=0.0,
                    upper=0.0,
                )
            else:
                for k in range(
                    t,
                    t + minimum_on,
                ):
                    add_constraint(
                        {
                            int(on[k]): 1.0,
                            int(start[t]): -1.0,
                        },
                        lower=0.0,
                    )

            for k in range(
                t,
                min(
                    horizon,
                    t + minimum_off,
                ),
            ):
                add_constraint(
                    {
                        int(on[k]): 1.0,
                        int(stop[t]): 1.0,
                    },
                    upper=1.0,
                )

    minimum_vsd_on = _minimum_intervals(
        vsd_spec.min_on_seconds,
        interval_seconds,
    )

    minimum_vsd_off = _minimum_intervals(
        vsd_spec.min_off_seconds,
        interval_seconds,
    )

    vsd_initial_state = runtime_by_id[vsd_spec.id]

    vsd_locked_on_intervals = 0
    vsd_locked_off_intervals = 0

    if vsd_initial_state.is_on:
        vsd_locked_on_intervals = (
            _remaining_lock_intervals(
                vsd_spec.min_on_seconds,
                vsd_initial_state.seconds_in_state,
                interval_seconds,
            )
        )
    else:
        vsd_locked_off_intervals = (
            _remaining_lock_intervals(
                vsd_spec.min_off_seconds,
                vsd_initial_state.seconds_in_state,
                interval_seconds,
            )
        )

    for k in range(
        min(horizon, vsd_locked_on_intervals,)
    ):
        add_constraint(
            {
                int(vsd_on[k]): 1.0,
            },
            lower=1.0,
            upper=1.0,
        )

    for k in range(
        min(horizon, vsd_locked_off_intervals)
    ):
        add_constraint(
            {
                int(vsd_on[k]): 1.0,
            },
            lower=0.0,
            upper=0.0,
        )

    for t in range(horizon):
        transition = {
            int(vsd_on[t]): 1.0,
            int(vsd_start[t]): -1.0,
            int(vsd_stop[t]): 1.0,
        }

        if t > 0:
            transition[int(vsd_on[t - 1])] = -1.0
            transition_rhs = 0.0
        else:
            transition_rhs = (
                1.0
                if vsd_initial_state.is_on
                else 0.0
            )

        add_constraint(
            transition,
            lower=transition_rhs,
            upper=transition_rhs,
        )

        add_constraint(
            {
                int(vsd_start[t]): 1.0,
                int(vsd_stop[t]): 1.0,
            },
            upper=1.0,
        )

        if (t + minimum_vsd_on > horizon):
            add_constraint(
                {
                    int(vsd_start[t]): 1.0,
                },
                lower=0.0,
                upper=0.0,
            )
        else:
            for k in range(
                t,
                t + minimum_vsd_on,
            ):
                add_constraint(
                    {
                        int(vsd_on[k]): 1.0,
                        int(vsd_start[t]): -1.0,
                    },
                    lower=0.0,
                )

        for k in range(
            t,
            min(horizon, t + minimum_vsd_off,
            ),
        ):
            add_constraint(
                {
                    int(vsd_on[k]): 1.0,
                    int(vsd_stop[t]): 1.0,
                },
                upper=1.0,
            )

        add_constraint(
            {
                int(vsd_fraction[t]): 1.0,
                int(vsd_on[t]): -1.0,
            },
            upper=0.0,
        )

        add_constraint(
            {
                int(vsd_fraction[t]): 1.0,
                int(vsd_on[t]): (
                    -vsd_spec
                    .min_load_fraction
                ),
            },
            lower=0.0,
        )

    pressure_gain = (
        parameters
        .gas_constant_j_per_kg_k
        * parameters.temperature_k
        / parameters.volume_m3
        * interval_seconds
        / 100_000.0
    )

    for scenario_index, outflow in enumerate(
        scenario_outflows
    ):
        pressure = pressure_blocks[
            scenario_index
        ]

        for t in range(horizon):
            pressure_row = {
                int(
                    pressure[t + 1]
                ): 1.0,
                int(
                    pressure[t]
                ): -1.0,
                int(
                    vsd_fraction[t]
                ): (
                    -pressure_gain
                    * vsd_spec
                    .max_mass_flow_kg_s
                ),
            }

            for spec in fixed_specs:
                on = blocks[
                    f"fixed_on:{spec.id}"
                ]

                pressure_row[
                    int(on[t])
                ] = (
                    -pressure_gain
                    * spec.max_mass_flow_kg_s
                )

            pressure_rhs = (
                -pressure_gain
                * float(outflow)
            )

            add_constraint(
                pressure_row,
                lower=pressure_rhs,
                upper=pressure_rhs,
            )

            reserve_row = {
                int(vsd_on[t]): (
                    vsd_spec
                    .max_mass_flow_kg_s
                )
            }

            for spec in fixed_specs:
                on = blocks[
                    f"fixed_on:{spec.id}"
                ]

                reserve_row[
                    int(on[t])
                ] = (
                    spec.max_mass_flow_kg_s
                )

            add_constraint(
                reserve_row,
                lower=(
                    float(outflow)
                    + config
                    .reserve_mass_flow_kg_s
                ),
            )

            add_constraint(
                {
                    int(
                        pressure[t + 1]
                    ): 1.0,
                    int(
                        worst_overpressure[t]
                    ): -1.0,
                },
                upper=target_bar_g,
            )

    constraint_matrix = lil_matrix(
        (
            len(rows),
            variable_count,
        ),
        dtype=float,
    )

    for row_index, coefficients in enumerate(
        rows
    ):
        for column_index, value in (
            coefficients.items()
        ):
            constraint_matrix[
                row_index,
                column_index,
            ] = value

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(
            lower_bounds,
            upper_bounds,
        ),
        constraints=LinearConstraint(
            constraint_matrix.tocsr(),
            np.asarray(
                row_lower,
                dtype=float,
            ),
            np.asarray(
                row_upper,
                dtype=float,
            ),
        ),
        options={
            "time_limit": (
                config.time_limit_seconds
            ),
            "mip_rel_gap": (
                config.mip_rel_gap
            ),
        },
    )

    if (
        not result.success
        or result.x is None
    ):
        raise RuntimeError(
            "Robust MILP optimization failed: "
            f"{result.message}"
        )

    solution = result.x
    schedule_rows: list[
        dict[str, object]
    ] = []
    cumulative_energy_kwh = 0.0
    startup_count = 0

    for t in range(horizon):
        fixed_flow = 0.0
        fixed_power = 0.0

        row: dict[str, object] = {
            "time_seconds": (
                (t + 1)
                * interval_seconds
            ),
            "outflow_min_kg_s": float(
                np.min(scenario_outflows)
            ),
            "outflow_max_kg_s": float(
                np.max(scenario_outflows)
            ),
        }

        for spec in fixed_specs:
            on = blocks[
                f"fixed_on:{spec.id}"
            ]
            start = blocks[
                f"fixed_start:{spec.id}"
            ]

            is_on = bool(
                round(
                    float(
                        solution[
                            on[t]
                        ]
                    )
                )
            )

            did_start = bool(
                round(
                    float(
                        solution[
                            start[t]
                        ]
                    )
                )
            )

            row[
                f"{spec.id}_on"
            ] = is_on

            row[
                f"{spec.id}_start"
            ] = did_start

            if is_on:
                fixed_flow += (
                    spec.max_mass_flow_kg_s
                )
                fixed_power += (
                    spec.rated_power_kw
                )

            startup_count += int(
                did_start
            )

        vsd_is_on = bool(
            round(
                float(
                    solution[
                        vsd_on[t]
                    ]
                )
            )
        )

        vsd_did_start = bool(
            round(
                float(
                    solution[
                        vsd_start[t]
                    ]
                )
            )
        )

        fraction = float(
            solution[
                vsd_fraction[t]
            ]
        )

        if not vsd_is_on:
            fraction = 0.0

        vsd_flow = (
            fraction
            * vsd_spec
            .max_mass_flow_kg_s
        )

        vsd_power = (
            0.0
            if not vsd_is_on
            else (
                vsd_spec.idle_power_kw
                + fraction
                * (
                    vsd_spec
                    .rated_power_kw
                    - vsd_spec
                    .idle_power_kw
                )
            )
        )

        startup_count += int(
            vsd_did_start
        )

        row[
            f"{vsd_spec.id}_on"
        ] = vsd_is_on
        row[
            f"{vsd_spec.id}_fraction"
        ] = fraction
        row[
            f"{vsd_spec.id}_start"
        ] = vsd_did_start

        total_flow = (
            fixed_flow
            + vsd_flow
        )

        total_power = (
            fixed_power
            + vsd_power
        )

        interval_energy_kwh = (
            total_power
            * interval_hours
        )

        cumulative_energy_kwh += (
            interval_energy_kwh
        )

        online_capacity = (
            fixed_flow
            + (
                vsd_spec
                .max_mass_flow_kg_s
                if vsd_is_on
                else 0.0
            )
        )

        minimum_reserve = float(
            np.min(
                online_capacity
                - scenario_outflows
            )
        )

        interval_pressures = [
            float(
                solution[
                    pressure_blocks[
                        scenario_index
                    ][t + 1]
                ]
            )
            for scenario_index in range(
                scenario_count
            )
        ]

        row[
            "compressor_flow_kg_s"
        ] = total_flow
        row[
            "online_capacity_kg_s"
        ] = online_capacity
        row[
            "minimum_reserve_kg_s"
        ] = minimum_reserve
        row[
            "power_kw"
        ] = total_power
        row[
            "energy_kwh"
        ] = interval_energy_kwh
        row[
            "cumulative_energy_kwh"
        ] = cumulative_energy_kwh
        row[
            "worst_case_min_pressure_bar_g"
        ] = min(interval_pressures)
        row[
            "worst_case_max_pressure_bar_g"
        ] = max(interval_pressures)
        row[
            "worst_overpressure_bar"
        ] = max(
            0.0,
            float(
                solution[
                    worst_overpressure[t]
                ]
            ),
        )
        row[
            "robust_safety_violation"
        ] = not (
            min(interval_pressures)
            >= safety_min_bar_g
            - 1.0e-8
            and max(interval_pressures)
            <= safety_max_bar_g
            + 1.0e-8
            and minimum_reserve
            >= config
            .reserve_mass_flow_kg_s
            - 1.0e-8
        )

        schedule_rows.append(row)

    schedule = pd.DataFrame(
        schedule_rows
    )

    pressure_rows: list[
        dict[str, object]
    ] = []

    for t in range(
        horizon + 1
    ):
        pressure_row: dict[
            str,
            object,
        ] = {
            "time_seconds": (
                t * interval_seconds
            ),
        }

        for scenario_index, scenario in enumerate(
            scenario_set.scenarios
        ):
            pressure_row[
                scenario.id
            ] = float(
                solution[
                    pressure_blocks[
                        scenario_index
                    ][t]
                ]
            )

        pressure_rows.append(
            pressure_row
        )

    scenario_pressures = pd.DataFrame(
        pressure_rows
    )

    worst_overpressure_bar_hours = (
        float(
            schedule[
                "worst_overpressure_bar"
            ].sum()
        )
        * interval_hours
    )

    return RobustOptimizationResult(
        schedule=schedule,
        scenario_pressures=(
            scenario_pressures
        ),
        objective_value=float(
            result.fun
        ),
        energy_kwh=float(
            cumulative_energy_kwh
        ),
        startup_count=int(
            startup_count
        ),
        worst_overpressure_bar_hours=(
            worst_overpressure_bar_hours
        ),
        solver_status=int(
            result.status
        ),
        solver_message=str(
            result.message
        ),
        scenario_count=scenario_count,
        initial_compressor_state_assumption=(
            
                "provided_runtime_state"
                if initial_runtime_state
                is not None
                else "default_all_off_min_off_satisfied"
            
        ),
        method=(
            "shared_action_robust_milp"
        ),
        evidence_class="SIMULATED",
        causal_claim=False,
    )
