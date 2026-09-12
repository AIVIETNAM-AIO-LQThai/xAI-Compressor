from __future__ import annotations

from dataclasses import dataclass

PA_PER_BAR = 100_000.0


@dataclass(frozen=True)
class TwinParameters:
    volume_m3: float
    temperature_k: float
    gas_constant_j_per_kg_k: float = 287.05
    ambient_pressure_pa: float = 101_325.0

    def __post_init__(self) -> None:
        if self.volume_m3 <= 0.0:
            raise ValueError(
                "volume_m3 must be positive."
            )

        if self.temperature_k <= 0.0:
            raise ValueError(
                "temperature_k must be positive."
            )

        if self.gas_constant_j_per_kg_k <= 0.0:
            raise ValueError(
                "gas_constant_j_per_kg_k "
                "must be positive."
            )

        if self.ambient_pressure_pa <= 0.0:
            raise ValueError(
                "ambient_pressure_pa "
                "must be positive."
            )


def bar_g_to_absolute_pa(
    pressure_bar_g: float,
    *,
    ambient_pressure_pa: float = 101_325.0,
) -> float:
    return (
        ambient_pressure_pa
        + pressure_bar_g * PA_PER_BAR
    )


def absolute_pa_to_bar_g(
    pressure_pa: float,
    *,
    ambient_pressure_pa: float = 101_325.0,
) -> float:
    return (
        pressure_pa - ambient_pressure_pa
    ) / PA_PER_BAR


def mass_in_receiver(
    pressure_pa: float,
    parameters: TwinParameters,
) -> float:
    if pressure_pa <= 0.0:
        raise ValueError(
            "Absolute pressure must be positive."
        )

    return (
        pressure_pa
        * parameters.volume_m3
        / (
            parameters.gas_constant_j_per_kg_k
            * parameters.temperature_k
        )
    )


def pressure_rate_pa_per_s(
    *,
    mass_flow_in_kg_s: float,
    demand_mass_flow_kg_s: float,
    leak_mass_flow_kg_s: float,
    parameters: TwinParameters,
) -> float:
    if mass_flow_in_kg_s < 0.0:
        raise ValueError(
            "mass_flow_in_kg_s "
            "cannot be negative."
        )

    if demand_mass_flow_kg_s < 0.0:
        raise ValueError(
            "demand_mass_flow_kg_s "
            "cannot be negative."
        )

    if leak_mass_flow_kg_s < 0.0:
        raise ValueError(
            "leak_mass_flow_kg_s "
            "cannot be negative."
        )

    net_mass_flow = (
        mass_flow_in_kg_s
        - demand_mass_flow_kg_s
        - leak_mass_flow_kg_s
    )

    return (
        parameters.gas_constant_j_per_kg_k
        * parameters.temperature_k
        / parameters.volume_m3
        * net_mass_flow
    )


def step_pressure(
    *,
    pressure_pa: float,
    mass_flow_in_kg_s: float,
    demand_mass_flow_kg_s: float,
    leak_mass_flow_kg_s: float,
    timestep_seconds: float,
    parameters: TwinParameters,
) -> float:
    if pressure_pa <= 0.0:
        raise ValueError(
            "Absolute pressure must be positive."
        )

    if timestep_seconds <= 0.0:
        raise ValueError(
            "timestep_seconds must be positive."
        )

    pressure_rate = pressure_rate_pa_per_s(
        mass_flow_in_kg_s=mass_flow_in_kg_s,
        demand_mass_flow_kg_s=(
            demand_mass_flow_kg_s
        ),
        leak_mass_flow_kg_s=(
            leak_mass_flow_kg_s
        ),
        parameters=parameters,
    )

    next_pressure = (
        pressure_pa
        + pressure_rate * timestep_seconds
    )

    if next_pressure <= 0.0:
        raise ValueError(
            "Simulation produced non-positive "
            "absolute pressure."
        )

    return next_pressure