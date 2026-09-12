from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ml.optimization.scheduler import (
    OptimizationConfig,
)
from ml.twin.compressor import CompressorSpec
from ml.twin.physics import TwinParameters

REPO_ROOT = (
    Path(__file__).resolve().parents[2]
)


@dataclass(frozen=True)
class RuntimeConfig:
    compressors: list[
        CompressorSpec
    ]
    twin_parameters: TwinParameters
    pressure: dict[str, float]
    optimizer: OptimizationConfig
    project: dict[str, Any]


def load_yaml(
    relative_path: str,
) -> dict[str, Any]:
    path = (
        REPO_ROOT
        / relative_path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise TypeError(
            f"{relative_path} must contain "
            "a YAML mapping."
        )

    return data


def load_runtime_config() -> RuntimeConfig:
    compressor_raw = load_yaml(
        "configs/compressors.yaml"
    )

    twin_raw = load_yaml(
        "configs/twin.yaml"
    )

    optimizer_raw = load_yaml(
        "configs/optimization.yaml"
    )["optimization"]

    project_raw = load_yaml(
        "configs/project.yaml"
    )

    compressors = [
        CompressorSpec(
            id=str(item["id"]),
            kind=str(item["kind"]),
            max_mass_flow_kg_s=float(
                item[
                    "max_mass_flow_kg_s"
                ]
            ),
            rated_power_kw=float(
                item["rated_power_kw"]
            ),
            idle_power_kw=float(
                item["idle_power_kw"]
            ),
            min_load_fraction=float(
                item[
                    "min_load_fraction"
                ]
            ),
            min_on_seconds=float(
                item["min_on_seconds"]
            ),
            min_off_seconds=float(
                item["min_off_seconds"]
            ),
        )
        for item in compressor_raw[
            "compressors"
        ]
    ]

    air = twin_raw["air"]
    receiver = twin_raw["receiver"]

    twin_parameters = TwinParameters(
        volume_m3=float(
            receiver["volume_m3"]
        ),
        temperature_k=float(
            air["temperature_k"]
        ),
        gas_constant_j_per_kg_k=float(
            air[
                "gas_constant_j_per_kg_k"
            ]
        ),
        ambient_pressure_pa=float(
            air["ambient_pressure_pa"]
        ),
    )

    solver = optimizer_raw["solver"]

    optimizer = OptimizationConfig(
        interval_seconds=float(
            optimizer_raw[
                "interval_seconds"
            ]
        ),
        reserve_mass_flow_kg_s=float(
            optimizer_raw[
                "reserve_mass_flow_kg_s"
            ]
        ),
        startup_penalty_kwh=float(
            optimizer_raw[
                "startup_penalty_kwh"
            ]
        ),
        overpressure_penalty_kwh_per_bar_hour=float(
            optimizer_raw[
                "overpressure_penalty_kwh_per_bar_hour"
            ]
        ),
        terminal_pressure_min_bar_g=float(
            optimizer_raw[
                "terminal_pressure_min_bar_g"
            ]
        ),
        time_limit_seconds=float(
            solver[
                "time_limit_seconds"
            ]
        ),
        mip_rel_gap=float(
            solver["mip_rel_gap"]
        ),
    )

    pressure = {
        key: float(value)
        for key, value in (
            compressor_raw[
                "pressure"
            ]
        ).items()
    }

    return RuntimeConfig(
        compressors=compressors,
        twin_parameters=twin_parameters,
        pressure=pressure,
        optimizer=optimizer,
        project=project_raw,
    )
