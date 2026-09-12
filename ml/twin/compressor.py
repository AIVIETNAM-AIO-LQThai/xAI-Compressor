from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompressorSpec:
    id: str
    kind: str
    max_mass_flow_kg_s: float
    rated_power_kw: float
    idle_power_kw: float
    min_load_fraction: float
    min_on_seconds: float
    min_off_seconds: float

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError(
                "Compressor id cannot be empty."
            )

        if self.kind not in {
            "fixed",
            "vsd",
        }:
            raise ValueError(
                "Compressor kind must be "
                "'fixed' or 'vsd'."
            )

        if self.max_mass_flow_kg_s <= 0.0:
            raise ValueError(
                "max_mass_flow_kg_s "
                "must be positive."
            )

        if self.rated_power_kw <= 0.0:
            raise ValueError(
                "rated_power_kw "
                "must be positive."
            )

        if self.idle_power_kw < 0.0:
            raise ValueError(
                "idle_power_kw "
                "cannot be negative."
            )

        if (
            self.idle_power_kw
            > self.rated_power_kw
        ):
            raise ValueError(
                "idle_power_kw cannot exceed "
                "rated_power_kw."
            )

        if not (
            0.0
            <= self.min_load_fraction
            <= 1.0
        ):
            raise ValueError(
                "min_load_fraction "
                "must be in [0, 1]."
            )

        if (
            self.kind == "fixed"
            and self.min_load_fraction != 1.0
        ):
            raise ValueError(
                "Fixed compressors must use "
                "min_load_fraction=1.0."
            )

        if self.min_on_seconds < 0.0:
            raise ValueError(
                "min_on_seconds "
                "cannot be negative."
            )

        if self.min_off_seconds < 0.0:
            raise ValueError(
                "min_off_seconds "
                "cannot be negative."
            )


@dataclass(frozen=True)
class CompressorOperatingPoint:
    compressor_id: str
    is_on: bool
    load_fraction: float
    mass_flow_kg_s: float
    power_kw: float


@dataclass(frozen=True)
class FleetOperatingPoint:
    points: tuple[
        CompressorOperatingPoint,
        ...,
    ]
    total_mass_flow_kg_s: float
    total_power_kw: float


def fixed_operating_point(
    spec: CompressorSpec,
    *,
    is_on: bool,
) -> CompressorOperatingPoint:
    if spec.kind != "fixed":
        raise ValueError(
            "Expected a fixed-speed "
            "compressor."
        )

    if not is_on:
        return CompressorOperatingPoint(
            compressor_id=spec.id,
            is_on=False,
            load_fraction=0.0,
            mass_flow_kg_s=0.0,
            power_kw=0.0,
        )

    return CompressorOperatingPoint(
        compressor_id=spec.id,
        is_on=True,
        load_fraction=1.0,
        mass_flow_kg_s=(
            spec.max_mass_flow_kg_s
        ),
        power_kw=spec.rated_power_kw,
    )


def vsd_operating_point(
    spec: CompressorSpec,
    *,
    load_fraction: float,
) -> CompressorOperatingPoint:
    if spec.kind != "vsd":
        raise ValueError(
            "Expected a VSD compressor."
        )

    if not (
        0.0 <= load_fraction <= 1.0
    ):
        raise ValueError(
            "VSD load_fraction "
            "must be in [0, 1]."
        )

    if load_fraction == 0.0:
        return CompressorOperatingPoint(
            compressor_id=spec.id,
            is_on=False,
            load_fraction=0.0,
            mass_flow_kg_s=0.0,
            power_kw=0.0,
        )

    if (
        load_fraction
        < spec.min_load_fraction
    ):
        raise ValueError(
            "Running VSD load_fraction "
            "is below min_load_fraction."
        )

    mass_flow = (
        load_fraction
        * spec.max_mass_flow_kg_s
    )

    power = (
        spec.idle_power_kw
        + load_fraction
        * (
            spec.rated_power_kw
            - spec.idle_power_kw
        )
    )

    return CompressorOperatingPoint(
        compressor_id=spec.id,
        is_on=True,
        load_fraction=load_fraction,
        mass_flow_kg_s=mass_flow,
        power_kw=power,
    )


def fleet_operating_point(
    specs: list[CompressorSpec],
    *,
    load_commands: dict[str, float],
) -> FleetOperatingPoint:
    ids = {
        spec.id
        for spec in specs
    }

    if len(ids) != len(specs):
        raise ValueError(
            "Compressor ids must be unique."
        )

    unknown_ids = (
        set(load_commands)
        - ids
    )

    if unknown_ids:
        raise ValueError(
            "Unknown compressor ids in "
            "load_commands: "
            f"{sorted(unknown_ids)}"
        )

    points = []

    for spec in specs:
        command = float(
            load_commands.get(
                spec.id,
                0.0,
            )
        )

        if spec.kind == "fixed":
            if command not in {
                0.0,
                1.0,
            }:
                raise ValueError(
                    "Fixed compressor command "
                    "must be 0.0 or 1.0."
                )

            point = fixed_operating_point(
                spec,
                is_on=bool(command),
            )

        else:
            point = vsd_operating_point(
                spec,
                load_fraction=command,
            )

        points.append(point)

    return FleetOperatingPoint(
        points=tuple(points),
        total_mass_flow_kg_s=sum(
            point.mass_flow_kg_s
            for point in points
        ),
        total_power_kw=sum(
            point.power_kw
            for point in points
        ),
    )