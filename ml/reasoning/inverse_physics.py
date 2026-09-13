from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ml.twin.physics import (
    TwinParameters,
    bar_g_to_absolute_pa,
)


@dataclass(frozen=True)
class OutflowInference:
    interval_total_outflow_kg_s: tuple[
        float,
        ...
    ]
    mean_total_outflow_kg_s: float
    std_total_outflow_kg_s: float
    physically_consistent: bool
    evidence_class: str
    causal_claim: bool

    @property
    def interval_count(self) -> int:
        return len(
            self.interval_total_outflow_kg_s
        )


def infer_total_outflow(
    pressure_bar_g: Sequence[float],
    inflow_mass_flow_kg_s: Sequence[float],
    *,
    interval_seconds: float,
    parameters: TwinParameters,
    physical_tolerance_kg_s: float = 1.0e-9,
) -> OutflowInference:
    pressure = np.asarray(
        list(pressure_bar_g),
        dtype=float,
    )

    inflow = np.asarray(
        list(inflow_mass_flow_kg_s),
        dtype=float,
    )

    if interval_seconds <= 0.0:
        raise ValueError(
            "interval_seconds must be positive."
        )

    if physical_tolerance_kg_s < 0.0:
        raise ValueError(
            "physical_tolerance_kg_s "
            "cannot be negative."
        )

    if pressure.size < 2:
        raise ValueError(
            "At least two pressure observations "
            "are required."
        )

    if pressure.size != inflow.size + 1:
        raise ValueError(
            "pressure_bar_g must contain exactly "
            "one more value than inflow."
        )

    if not np.isfinite(pressure).all():
        raise ValueError(
            "Pressure observations contain "
            "non-finite values."
        )

    if not np.isfinite(inflow).all():
        raise ValueError(
            "Inflow observations contain "
            "non-finite values."
        )

    if (inflow < 0.0).any():
        raise ValueError(
            "Inflow cannot be negative."
        )

    pressure_pa = np.asarray(
        [
            bar_g_to_absolute_pa(
                float(value),
                ambient_pressure_pa=(
                    parameters
                    .ambient_pressure_pa
                ),
            )
            for value in pressure
        ],
        dtype=float,
    )

    pressure_rate_pa_s = (
        np.diff(pressure_pa)
        / interval_seconds
    )

    storage_mass_rate = (
        pressure_rate_pa_s
        * parameters.volume_m3
        / (
            parameters
            .gas_constant_j_per_kg_k
            * parameters.temperature_k
        )
    )

    total_outflow = (
        inflow
        - storage_mass_rate
    )

    physically_consistent = bool(
        (
            total_outflow
            >= -physical_tolerance_kg_s
        ).all()
    )

    return OutflowInference(
        interval_total_outflow_kg_s=tuple(
            float(value)
            for value in total_outflow
        ),
        mean_total_outflow_kg_s=float(
            np.mean(total_outflow)
        ),
        std_total_outflow_kg_s=float(
            np.std(total_outflow)
        ),
        physically_consistent=(
            physically_consistent
        ),
        evidence_class=(
            "PHYSICS_MODEL_INFERENCE"
        ),
        causal_claim=False,
    )