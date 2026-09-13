from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ml.reasoning.inverse_physics import (
    OutflowInference,
)


@dataclass(frozen=True)
class IntervalEstimate:
    lower: float
    center: float
    upper: float

    def __post_init__(self) -> None:
        if not (
            self.lower
            <= self.center
            <= self.upper
        ):
            raise ValueError(
                "Interval must satisfy "
                "lower <= center <= upper."
            )

    @property
    def width(self) -> float:
        return (
            self.upper
            - self.lower
        )


@dataclass(frozen=True)
class PhysicalStateUncertainty:
    total_outflow_kg_s: IntervalEstimate
    demand_kg_s: IntervalEstimate | None
    leak_kg_s: IntervalEstimate | None
    leak_identifiable: bool
    assumptions: tuple[str, ...]
    evidence_class: str
    causal_claim: bool


def _nonnegative_interval(
    *,
    center: float,
    half_width: float,
) -> IntervalEstimate:
    if half_width < 0.0:
        raise ValueError(
            "half_width cannot be negative."
        )

    return IntervalEstimate(
        lower=max(
            0.0,
            center - half_width,
        ),
        center=max(
            0.0,
            center,
        ),
        upper=max(
            0.0,
            center + half_width,
        ),
    )


def infer_state_uncertainty(
    inference: OutflowInference,
    *,
    outflow_abs_error_kg_s: float,
    independent_demand_kg_s: (
        Sequence[float] | None
    ) = None,
    demand_abs_error_kg_s: float = 0.0,
) -> PhysicalStateUncertainty:
    if not inference.physically_consistent:
        raise ValueError(
            "Outflow inference is not "
            "physically consistent."
        )

    if outflow_abs_error_kg_s < 0.0:
        raise ValueError(
            "outflow_abs_error_kg_s "
            "cannot be negative."
        )

    if demand_abs_error_kg_s < 0.0:
        raise ValueError(
            "demand_abs_error_kg_s "
            "cannot be negative."
        )

    outflow_center = (
        inference.mean_total_outflow_kg_s
    )

    outflow_interval = (
        _nonnegative_interval(
            center=outflow_center,
            half_width=(
                outflow_abs_error_kg_s
            ),
        )
    )

    if independent_demand_kg_s is None:
        return PhysicalStateUncertainty(
            total_outflow_kg_s=(
                outflow_interval
            ),
            demand_kg_s=None,
            leak_kg_s=None,
            leak_identifiable=False,
            assumptions=(
                (
                    "Receiver dynamics constrain "
                    "total outflow only."
                ),
                (
                    "Leakage and legitimate demand "
                    "remain observationally "
                    "equivalent without independent "
                    "demand measurement."
                ),
            ),
            evidence_class=(
                "PHYSICS_MODEL_INFERENCE"
            ),
            causal_claim=False,
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

    demand_center = float(
        np.mean(demand)
    )

    demand_interval = (
        _nonnegative_interval(
            center=demand_center,
            half_width=(
                demand_abs_error_kg_s
            ),
        )
    )

    leak_center = (
        outflow_center
        - demand_center
    )

    leak_lower = max(
        0.0,
        (
            outflow_interval.lower
            - demand_interval.upper
        ),
    )

    leak_upper = max(
        0.0,
        (
            outflow_interval.upper
            - demand_interval.lower
        ),
    )

    if (
        leak_center
        < -(
            outflow_abs_error_kg_s
            + demand_abs_error_kg_s
        )
    ):
        raise ValueError(
            "Demand and outflow uncertainty "
            "sets imply physically impossible "
            "negative leakage."
        )

    leak_center = max(
        0.0,
        leak_center,
    )

    leak_interval = IntervalEstimate(
        lower=leak_lower,
        center=leak_center,
        upper=max(
            leak_center,
            leak_upper,
        ),
    )

    return PhysicalStateUncertainty(
        total_outflow_kg_s=(
            outflow_interval
        ),
        demand_kg_s=demand_interval,
        leak_kg_s=leak_interval,
        leak_identifiable=True,
        assumptions=(
            (
                "Outflow uncertainty is bounded "
                "by the configured absolute error."
            ),
            (
                "Demand uncertainty is bounded "
                "independently."
            ),
            (
                "Leakage is derived from the "
                "difference between the two "
                "uncertainty sets."
            ),
        ),
        evidence_class=(
            "PHYSICS_MODEL_INFERENCE"
        ),
        causal_claim=False,
    )