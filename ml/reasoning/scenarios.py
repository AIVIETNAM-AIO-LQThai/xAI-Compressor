from __future__ import annotations

from dataclasses import dataclass

from ml.reasoning.uncertainty import (
    IntervalEstimate,
    PhysicalStateUncertainty,
)


@dataclass(frozen=True)
class PhysicalScenario:
    id: str
    total_outflow_kg_s: float
    demand_kg_s: float | None
    leak_kg_s: float | None
    allocation_identifiable: bool
    source: str
    evidence_class: str
    causal_claim: bool


@dataclass(frozen=True)
class PhysicalScenarioSet:
    scenarios: tuple[
        PhysicalScenario,
        ...
    ]
    allocation_identifiable: bool
    evidence_class: str
    causal_claim: bool

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError(
                "Scenario set cannot be empty."
            )

        ids = [
            scenario.id
            for scenario in self.scenarios
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Scenario ids must be unique."
            )


def _interval_points(
    interval: IntervalEstimate,
    *,
    tolerance: float,
) -> tuple[
    tuple[str, float],
    ...
]:
    raw = (
        ("low", interval.lower),
        ("center", interval.center),
        ("high", interval.upper),
    )

    points: list[
        tuple[str, float]
    ] = []

    for label, value in raw:
        duplicate = any(
            abs(value - existing_value)
            <= tolerance
            for _, existing_value in points
        )

        if not duplicate:
            points.append(
                (
                    label,
                    float(value),
                )
            )

    return tuple(points)


def _within(
    value: float,
    interval: IntervalEstimate,
    *,
    tolerance: float,
) -> bool:
    return (
        value
        >= interval.lower - tolerance
        and value
        <= interval.upper + tolerance
    )


def generate_physical_scenarios(
    state: PhysicalStateUncertainty,
    *,
    tolerance_kg_s: float = 1.0e-9,
) -> PhysicalScenarioSet:
    if tolerance_kg_s < 0.0:
        raise ValueError(
            "tolerance_kg_s cannot be negative."
        )

    if not state.leak_identifiable:
        outflow_points = _interval_points(
            state.total_outflow_kg_s,
            tolerance=tolerance_kg_s,
        )

        scenarios = tuple(
            PhysicalScenario(
                id=f"outflow_{label}",
                total_outflow_kg_s=value,
                demand_kg_s=None,
                leak_kg_s=None,
                allocation_identifiable=False,
                source=(
                    "bounded_total_outflow"
                ),
                evidence_class=(
                    "PHYSICS_MODEL_INFERENCE"
                ),
                causal_claim=False,
            )
            for label, value in outflow_points
        )

        return PhysicalScenarioSet(
            scenarios=scenarios,
            allocation_identifiable=False,
            evidence_class=(
                "PHYSICS_MODEL_INFERENCE"
            ),
            causal_claim=False,
        )

    if (
        state.demand_kg_s is None
        or state.leak_kg_s is None
    ):
        raise ValueError(
            "Identifiable leakage requires "
            "both demand and leak intervals."
        )

    demand_points = _interval_points(
        state.demand_kg_s,
        tolerance=tolerance_kg_s,
    )

    leak_points = _interval_points(
        state.leak_kg_s,
        tolerance=tolerance_kg_s,
    )

    scenarios: list[
        PhysicalScenario
    ] = []

    seen: list[
        tuple[float, float]
    ] = []

    for demand_label, demand in (
        demand_points
    ):
        for leak_label, leak in (
            leak_points
        ):
            total_outflow = (
                demand
                + leak
            )

            if not _within(
                total_outflow,
                state.total_outflow_kg_s,
                tolerance=tolerance_kg_s,
            ):
                continue

            duplicate = any(
                (
                    abs(
                        demand
                        - existing_demand
                    )
                    <= tolerance_kg_s
                    and abs(
                        leak
                        - existing_leak
                    )
                    <= tolerance_kg_s
                )
                for (
                    existing_demand,
                    existing_leak,
                ) in seen
            )

            if duplicate:
                continue

            seen.append(
                (
                    demand,
                    leak,
                )
            )

            scenarios.append(
                PhysicalScenario(
                    id=(
                        f"demand_{demand_label}"
                        f"__leak_{leak_label}"
                    ),
                    total_outflow_kg_s=float(
                        total_outflow
                    ),
                    demand_kg_s=float(
                        demand
                    ),
                    leak_kg_s=float(
                        leak
                    ),
                    allocation_identifiable=True,
                    source=(
                        "bounded_demand_leak"
                    ),
                    evidence_class=(
                        "PHYSICS_MODEL_INFERENCE"
                    ),
                    causal_claim=False,
                )
            )

    if not scenarios:
        raise ValueError(
            "No physically consistent "
            "demand/leak scenario exists "
            "inside the uncertainty set."
        )

    scenarios.sort(
        key=lambda item: (
            item.total_outflow_kg_s,
            (
                -1.0
                if item.demand_kg_s is None
                else item.demand_kg_s
            ),
            item.id,
        )
    )

    return PhysicalScenarioSet(
        scenarios=tuple(scenarios),
        allocation_identifiable=True,
        evidence_class=(
            "PHYSICS_MODEL_INFERENCE"
        ),
        causal_claim=False,
    )