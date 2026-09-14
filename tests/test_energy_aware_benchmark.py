from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from ml.energy.hierarchical_inference import (
    energy_reduction_fraction,
    higher_quantile,
    minimum_complete_passes,
    routing_mask,
    seeded_condition_orders,
)

ROOT = Path(__file__).resolve().parents[1]


def test_higher_quantile_and_routing_mask() -> None:
    values = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )

    threshold = higher_quantile(
        values,
        0.80,
    )

    assert threshold == 5.0

    mask = routing_mask(
        values,
        threshold,
    )

    assert mask.tolist() == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_seeded_condition_orders_are_deterministic() -> None:
    conditions = [
        "always_on_tcn",
        "route_q90",
        "route_q95",
        "route_q99",
    ]

    first = seeded_condition_orders(
        conditions,
        repetitions=5,
        seed=20260915,
    )
    second = seeded_condition_orders(
        conditions,
        repetitions=5,
        seed=20260915,
    )

    assert first == second
    assert len(first) == 5

    for order in first:
        assert set(order) == set(
            conditions
        )
        assert len(order) == len(
            conditions
        )


def test_minimum_complete_passes() -> None:
    assert minimum_complete_passes(
        2.1,
        minimum_workload_seconds=20.0,
    ) == 10

    assert minimum_complete_passes(
        25.0,
        minimum_workload_seconds=20.0,
    ) == 1


def test_energy_reduction_fraction() -> None:
    assert energy_reduction_fraction(
        100.0,
        70.0,
    ) == pytest.approx(0.30)


def test_benchmark_config_is_pretest_frozen() -> None:
    path = (
        ROOT
        / "configs"
        / "energy_aware_inference_benchmark.yaml"
    )
    config = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        config["benchmark"][
            "frozen_before_primary_test"
        ]
        is True
    )

    assert (
        config["tcn"][
            "checkpoint_sha256"
        ]
        == "3a2a98fd65ae91014aba89b04b953b984d6763003351ec03724d682632bc8f60"
    )

    assert (
        config["router"][
            "quantile_interpolation"
        ]
        == "higher"
    )

    assert set(
        config["router"][
            "operating_points"
        ]
    ) == {
        "route_q90",
        "route_q95",
        "route_q99",
    }

    measurement = config[
        "energy_measurement"
    ]

    assert (
        measurement[
            "sampling_interval_seconds"
        ]
        == 0.10
    )
    assert (
        measurement[
            "warmup_runs_per_condition"
        ]
        == 2
    )
    assert (
        measurement[
            "measured_repetitions"
        ]
        == 5
    )
    assert (
        measurement[
            "paired_measurement"
        ][
            "minimum_workload_seconds"
        ]
        == 20.0
    )
    assert (
        measurement[
            "materiality"
        ][
            "minimum_mean_gpu_energy_reduction_fraction"
        ]
        == 0.03
    )
