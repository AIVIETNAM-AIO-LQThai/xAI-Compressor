from fastapi import APIRouter

from backend.app.models import (
    TwinSimulationRequest,
    TwinSimulationResponse,
)
from backend.app.runtime import (
    load_runtime_config,
)
from ml.twin.physics import (
    absolute_pa_to_bar_g,
    bar_g_to_absolute_pa,
    step_pressure,
)

router = APIRouter(
    prefix="/twin",
    tags=["digital-twin"],
)


@router.post(
    "/simulate",
    response_model=TwinSimulationResponse,
)
def simulate_twin(
    request: TwinSimulationRequest,
) -> dict:
    runtime = load_runtime_config()
    parameters = (
        runtime.twin_parameters
    )

    pressure_pa = (
        bar_g_to_absolute_pa(
            request.initial_pressure_bar_g,
            ambient_pressure_pa=(
                parameters
                .ambient_pressure_pa
            ),
        )
    )

    trajectory = [
        {
            "time_seconds": 0.0,
            "pressure_bar_g":
                request.initial_pressure_bar_g,
        }
    ]

    pressure_values = [
        request.initial_pressure_bar_g
    ]

    steps = (
        request.duration_seconds
        // request.timestep_seconds
    )

    for step_index in range(steps):
        pressure_pa = step_pressure(
            pressure_pa=pressure_pa,
            mass_flow_in_kg_s=(
                request.mass_flow_in_kg_s
            ),
            demand_mass_flow_kg_s=(
                request
                .demand_mass_flow_kg_s
            ),
            leak_mass_flow_kg_s=(
                request
                .leak_mass_flow_kg_s
            ),
            timestep_seconds=float(
                request.timestep_seconds
            ),
            parameters=parameters,
        )

        pressure_bar_g = (
            absolute_pa_to_bar_g(
                pressure_pa,
                ambient_pressure_pa=(
                    parameters
                    .ambient_pressure_pa
                ),
            )
        )

        pressure_values.append(
            pressure_bar_g
        )

        trajectory.append(
            {
                "time_seconds": float(
                    (
                        step_index
                        + 1
                    )
                    * request
                    .timestep_seconds
                ),
                "pressure_bar_g":
                    pressure_bar_g,
            }
        )

    return {
        "evidence_class": "SIMULATED",
        "initial_pressure_bar_g":
            request.initial_pressure_bar_g,
        "final_pressure_bar_g":
            pressure_values[-1],
        "minimum_pressure_bar_g":
            min(pressure_values),
        "maximum_pressure_bar_g":
            max(pressure_values),
        "duration_seconds":
            request.duration_seconds,
        "timestep_seconds":
            request.timestep_seconds,
        "trajectory": trajectory,
        "limitations": [
            (
                "The digital twin is a "
                "lumped isothermal ideal-gas "
                "receiver model."
            ),
            (
                "Inputs are scenario "
                "assumptions unless supplied "
                "from validated telemetry."
            ),
            (
                "This endpoint provides "
                "simulated evidence only."
            ),
        ],
    }
