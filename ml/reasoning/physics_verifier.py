from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ml.reasoning.inverse_physics import (
    OutflowInference,
)

LEAK_HYPOTHESIS = (
    "leak_like_persistent_outflow"
)

DEMAND_HYPOTHESIS = "demand_surge"


@dataclass(frozen=True)
class PhysicsHypothesisResult:
    hypothesis_id: str
    status: str
    identifiable: bool
    estimated_level_kg_s: float | None
    estimated_change_kg_s: float | None
    required_observations: tuple[
        str,
        ...
    ]
    note: str
    evidence_class: str
    causal_claim: bool


def _increase_status(
    change_kg_s: float,
    *,
    tolerance_kg_s: float,
) -> str:
    if change_kg_s > tolerance_kg_s:
        return "supported"

    return "not_supported"


def verify_leak_vs_demand(
    inference: OutflowInference,
    *,
    reference_total_outflow_kg_s: float,
    independent_demand_kg_s: (
        Sequence[float] | None
    ) = None,
    reference_demand_kg_s: (
        float | None
    ) = None,
    nominal_leak_kg_s: (
        float | None
    ) = None,
    support_tolerance_kg_s: float = 1.0e-6,
) -> tuple[
    PhysicsHypothesisResult,
    PhysicsHypothesisResult,
]:
    if reference_total_outflow_kg_s < 0.0:
        raise ValueError("reference_total_outflow_kg_s cannot be negative.")
    if support_tolerance_kg_s < 0.0:
        raise ValueError("support_tolerance_kg_s cannot be negative.")
    if not inference.physically_consistent:
        raise ValueError("Outflow inference is not physically consistent.")

    additional_outflow = (
        inference.mean_total_outflow_kg_s
        - reference_total_outflow_kg_s
    )

    if independent_demand_kg_s is None:
        note = (
            "Receiver pressure dynamics observe "
            "only the sum of process demand and "
            "leakage. Without an independent "
            "process-demand observation, leak "
            "increase and demand surge are "
            "observationally equivalent under "
            "the current receiver model."
        )

        leak_result = (
            PhysicsHypothesisResult(
                hypothesis_id=(
                    LEAK_HYPOTHESIS
                ),
                status=(
                    "observationally_equivalent"
                ),
                identifiable=False,
                estimated_level_kg_s=None,
                estimated_change_kg_s=float(
                    additional_outflow
                ),
                required_observations=(
                    "independent_process_demand",
                ),
                note=note,
                evidence_class=(
                    "PHYSICS_MODEL_INFERENCE"
                ),
                causal_claim=False,
            )
        )

        demand_result = (
            PhysicsHypothesisResult(
                hypothesis_id=(
                    DEMAND_HYPOTHESIS
                ),
                status=(
                    "observationally_equivalent"
                ),
                identifiable=False,
                estimated_level_kg_s=None,
                estimated_change_kg_s=float(
                    additional_outflow
                ),
                required_observations=(
                    "independent_process_demand",
                ),
                note=note,
                evidence_class=(
                    "PHYSICS_MODEL_INFERENCE"
                ),
                causal_claim=False,
            )
        )

        return (
            leak_result,
            demand_result,
        )

    if reference_demand_kg_s is None:
        raise ValueError(
            "reference_demand_kg_s is required "
            "when independent demand is given."
        )

    if nominal_leak_kg_s is None:
        raise ValueError(
            "nominal_leak_kg_s is required "
            "when independent demand is given."
        )

    if reference_demand_kg_s < 0.0:
        raise ValueError(
            "reference_demand_kg_s "
            "cannot be negative."
        )

    if nominal_leak_kg_s < 0.0:
        raise ValueError(
            "nominal_leak_kg_s "
            "cannot be negative."
        )

    demand = np.asarray(
        list(independent_demand_kg_s),
        dtype=float,
    )

    if demand.size != inference.interval_count:
        raise ValueError(
            "Independent demand length must "
            "match the inferred interval count."
        )

    if not np.isfinite(demand).all():
        raise ValueError(
            "Independent demand contains "
            "non-finite values."
        )

    if (demand < 0.0).any():
        raise ValueError(
            "Independent demand cannot "
            "be negative."
        )

    total_outflow = np.asarray(
        inference.interval_total_outflow_kg_s,
        dtype=float,
    )

    implied_leak = total_outflow - demand

    if (implied_leak < -support_tolerance_kg_s).any():
        raise ValueError(
            "Independent demand implies "
            "negative leakage under the "
            "receiver mass-balance model."
        )

    implied_leak = np.maximum(
        implied_leak,
        0.0,
    )

    mean_demand = float(
        np.mean(demand)
    )

    mean_leak = float(
        np.mean(implied_leak)
    )

    demand_change = (
        mean_demand
        - reference_demand_kg_s
    )
    leak_change = (
        mean_leak
        - nominal_leak_kg_s
    )

    leak_result = PhysicsHypothesisResult(
        hypothesis_id=LEAK_HYPOTHESIS,
        status=_increase_status(
            leak_change,
            tolerance_kg_s=(
                support_tolerance_kg_s
            ),
        ),
        identifiable=True,
        estimated_level_kg_s=mean_leak,
        estimated_change_kg_s=float(
            leak_change
        ),
        required_observations=(
            "independent_process_demand",
        ),
        note=(
            "Leak level is inferred as total "
            "outflow minus independently "
            "observed process demand under the "
            "receiver mass-balance model."
        ),
        evidence_class=(
            "PHYSICS_MODEL_INFERENCE"
        ),
        causal_claim=False,
    )

    demand_result = PhysicsHypothesisResult(
        hypothesis_id=DEMAND_HYPOTHESIS,
        status=_increase_status(
            demand_change,
            tolerance_kg_s=(
                support_tolerance_kg_s
            ),
        ),
        identifiable=True,
        estimated_level_kg_s=mean_demand,
        estimated_change_kg_s=float(
            demand_change
        ),
        required_observations=(
            "independent_process_demand",
        ),
        note=(
            "Demand change is evaluated from "
            "the independent process-demand "
            "observation against its reference."
        ),
        evidence_class=(
            "PHYSICS_MODEL_INFERENCE"
        ),
        causal_claim=False,
    )

    return (
        leak_result,
        demand_result,
    )