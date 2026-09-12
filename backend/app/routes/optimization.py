from fastapi import (
    APIRouter,
    HTTPException,
)

from backend.app.evidence import (
    load_json_report,
)
from backend.app.models import (
    OptimizationRecommendRequest,
    OptimizationRecommendResponse,
)
from backend.app.runtime import (
    load_runtime_config,
)
from ml.explainability.actions import (
    explain_schedule,
)
from ml.optimization.scheduler import (
    optimize_schedule,
)

router = APIRouter(
    prefix="/optimization",
    tags=["optimization"],
)


@router.post(
    "/recommend",
    response_model=OptimizationRecommendResponse,
)
def recommend_action(
    request: OptimizationRecommendRequest,
) -> dict:
    runtime = load_runtime_config()

    robustness = load_json_report(
        "robustness_report.json"
    )

    try:
        result = optimize_schedule(
            request.demand_profile_kg_s,
            leak_mass_flow_kg_s=(
                request
                .leak_mass_flow_kg_s
            ),
            initial_pressure_bar_g=(
                request
                .initial_pressure_bar_g
            ),
            parameters=(
                runtime
                .twin_parameters
            ),
            compressors=(
                runtime.compressors
            ),
            target_bar_g=(
                runtime.pressure[
                    "target_bar_g"
                ]
            ),
            safety_min_bar_g=(
                runtime.pressure[
                    "safety_min_bar_g"
                ]
            ),
            safety_max_bar_g=(
                runtime.pressure[
                    "safety_max_bar_g"
                ]
            ),
            config=runtime.optimizer,
        )
    except (
        ValueError,
        RuntimeError,
    ) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    explanations = explain_schedule(
        result.schedule,
        target_bar_g=(
            runtime.pressure[
                "target_bar_g"
            ]
        ),
        safety_min_bar_g=(
            runtime.pressure[
                "safety_min_bar_g"
            ]
        ),
        safety_max_bar_g=(
            runtime.pressure[
                "safety_max_bar_g"
            ]
        ),
        required_reserve_kg_s=(
            runtime.optimizer
            .reserve_mass_flow_kg_s
        ),
    )

    current_row = (
        result.schedule.iloc[0]
    )

    current_explanation = (
        explanations[0]
    )

    current_actions = []

    for spec in runtime.compressors:
        if spec.kind == "fixed":
            is_on = bool(
                current_row[
                    f"{spec.id}_on"
                ]
            )

            fraction = (
                1.0
                if is_on
                else 0.0
            )
        else:
            fraction = float(
                current_row[
                    f"{spec.id}_fraction"
                ]
            )

            is_on = (
                fraction > 1.0e-10
            )

        current_actions.append(
            {
                "compressor_id":
                    spec.id,
                "kind": spec.kind,
                "state": (
                    "on"
                    if is_on
                    else "off"
                ),
                "load_fraction":
                    fraction,
            }
        )

    robustness_status = (
        robustness["robustness"][
            "status"
        ]
    )

    open_loop_approved = bool(
        robustness["robustness"][
            "all_tested_perturbations_safe"
        ]
    )

    project_safety = runtime.project[
        "safety"
    ]

    return {
        "evidence_class": "SIMULATED",
        "solver_status":
            result.solver_status,
        "solver_message":
            result.solver_message,
        "horizon_intervals": len(
            result.schedule
        ),
        "predicted_horizon_energy_kwh":
            result.energy_kwh,
        "current_actions":
            current_actions,
        "explanation": {
            "reason":
                current_explanation.reason,
            "predicted_pressure_end_bar_g":
                current_explanation
                .pressure_end_bar_g,
            "safety_margin_bar":
                current_explanation
                .safety_margin_bar,
            "reserve_available_kg_s":
                current_explanation
                .reserve_available_kg_s,
            "required_reserve_kg_s":
                current_explanation
                .required_reserve_kg_s,
            "binding_constraints":
                list(
                    current_explanation
                    .binding_constraints
                ),
            "causal_claim": False,
        },
        "safety": {
            "deployment_mode":
                project_safety[
                    "deployment_mode"
                ],
            "override_equipment_ctrl":
                False,
            "valid_for_seconds":
                runtime.optimizer
                .interval_seconds,
            "requires_reoptimization":
                True,
            "open_loop_schedule_approved":
                open_loop_approved,
            "robustness_status":
                robustness_status,
        },
        "limitations": [
            (
                "Recommendation is generated "
                "from simulated compressor "
                "and digital-twin assumptions."
            ),
            (
                "The recommendation is "
                "advisory-only and is valid "
                "for one optimizer interval."
            ),
            (
                "The frozen one-hour nominal "
                "schedule was not robust to "
                "all tested open-loop "
                "perturbations."
            ),
            (
                "The API does not write to "
                "PLC or equipment controls."
            ),
        ],
    }
