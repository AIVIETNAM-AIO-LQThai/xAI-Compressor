from __future__ import annotations

from dataclasses import dataclass

from ml.reasoning.inverse_physics import OutflowInference
from ml.twin.physics import (
    TwinParameters,
    absolute_pa_to_bar_g,
    bar_g_to_absolute_pa,
    step_pressure,
)


@dataclass(frozen=True)
class RecoveryAssessment:
    truth_total_outflow_kg_s: float
    inferred_total_outflow_kg_s: float
    absolute_error_kg_s: float
    tolerance_kg_s: float
    passed: bool
    evidence_class: str
    causal_claim: bool


def generate_synthetic_pressure_trace(
    *,
    initial_pressure_bar_g: float,
    inflow_profile_kg_s: tuple[float, ...],
    total_outflow_kg_s: float,
    interval_seconds: float,
    parameters: TwinParameters,
) -> tuple[float, ...]:
    if total_outflow_kg_s < 0.0:
        raise ValueError("total_outflow_kg_s cannot be negative.")
    if interval_seconds <= 0.0:
        raise ValueError("interval_seconds must be positive.")
    if not inflow_profile_kg_s:
        raise ValueError("inflow_profile_kg_s cannot be empty.")
    if any(value < 0.0 for value in inflow_profile_kg_s):
        raise ValueError("Synthetic inflow cannot be negative.")

    pressure_pa = bar_g_to_absolute_pa(
        initial_pressure_bar_g,
        ambient_pressure_pa=parameters.ambient_pressure_pa,
    )
    trace = [float(initial_pressure_bar_g)]

    for inflow in inflow_profile_kg_s:
        pressure_pa = step_pressure(
            pressure_pa=pressure_pa,
            mass_flow_in_kg_s=float(inflow),
            demand_mass_flow_kg_s=total_outflow_kg_s,
            leak_mass_flow_kg_s=0.0,
            timestep_seconds=interval_seconds,
            parameters=parameters,
        )
        trace.append(
            float(
                absolute_pa_to_bar_g(
                    pressure_pa,
                    ambient_pressure_pa=parameters.ambient_pressure_pa,
                )
            )
        )

    return tuple(trace)


def assess_outflow_recovery(
    inference: OutflowInference,
    *,
    truth_total_outflow_kg_s: float,
    tolerance_kg_s: float,
) -> RecoveryAssessment:
    if truth_total_outflow_kg_s < 0.0:
        raise ValueError("truth_total_outflow_kg_s cannot be negative.")
    if tolerance_kg_s < 0.0:
        raise ValueError("tolerance_kg_s cannot be negative.")
    if not inference.physically_consistent:
        raise ValueError("Inference is physically inconsistent.")

    inferred = inference.mean_total_outflow_kg_s
    error = abs(inferred - truth_total_outflow_kg_s)

    return RecoveryAssessment(
        truth_total_outflow_kg_s=truth_total_outflow_kg_s,
        inferred_total_outflow_kg_s=inferred,
        absolute_error_kg_s=float(error),
        tolerance_kg_s=tolerance_kg_s,
        passed=bool(error <= tolerance_kg_s),
        evidence_class="SIMULATED",
        causal_claim=False,
    )
