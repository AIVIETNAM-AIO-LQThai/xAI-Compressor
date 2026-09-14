from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path

import pandas as pd


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()

    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def higher_quantile(
    values: pd.Series,
    quantile: float,
) -> float:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1].")

    clean = values.dropna().sort_index()

    if clean.empty:
        raise ValueError("values cannot be empty.")

    return float(
        clean.quantile(
            quantile,
            interpolation="higher",
        )
    )


def routing_mask(
    scores: pd.Series,
    threshold: float,
) -> pd.Series:
    return (scores >= threshold).rename("route_to_tcn")


def seeded_condition_orders(
    conditions: list[str],
    *,
    repetitions: int,
    seed: int,
) -> list[list[str]]:
    if not conditions:
        raise ValueError("conditions cannot be empty.")
    if len(set(conditions)) != len(conditions):
        raise ValueError("conditions must be unique.")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")

    rng = random.Random(seed)
    orders: list[list[str]] = []

    for _ in range(repetitions):
        order = list(conditions)
        rng.shuffle(order)
        orders.append(order)

    return orders


def minimum_complete_passes(
    single_pass_seconds: float,
    *,
    minimum_workload_seconds: float,
) -> int:
    if single_pass_seconds <= 0.0:
        raise ValueError("single_pass_seconds must be positive.")
    if minimum_workload_seconds <= 0.0:
        raise ValueError("minimum_workload_seconds must be positive.")

    return max(
        1,
        math.ceil(
            minimum_workload_seconds
            / single_pass_seconds
        ),
    )


def energy_reduction_fraction(
    reference_energy_j: float,
    candidate_energy_j: float,
) -> float:
    if reference_energy_j <= 0.0:
        raise ValueError(
            "reference_energy_j must be positive."
        )

    return (
        reference_energy_j
        - candidate_energy_j
    ) / reference_energy_j
