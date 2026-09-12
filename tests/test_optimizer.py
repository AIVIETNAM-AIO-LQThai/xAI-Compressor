
import pytest

from ml.optimization.scheduler import (
    OptimizationConfig,
    optimize_schedule,
)
from ml.twin.compressor import CompressorSpec
from ml.twin.physics import TwinParameters


def compressors() -> list[CompressorSpec]:
    return [
        CompressorSpec(
            id="fixed_1",
            kind="fixed",
            max_mass_flow_kg_s=0.06,
            rated_power_kw=13.0,
            idle_power_kw=0.0,
            min_load_fraction=1.0,
            min_on_seconds=120,
            min_off_seconds=60,
        ),
        CompressorSpec(
            id="fixed_2",
            kind="fixed",
            max_mass_flow_kg_s=0.06,
            rated_power_kw=13.0,
            idle_power_kw=0.0,
            min_load_fraction=1.0,
            min_on_seconds=120,
            min_off_seconds=60,
        ),
        CompressorSpec(
            id="vsd_1",
            kind="vsd",
            max_mass_flow_kg_s=0.08,
            rated_power_kw=19.0,
            idle_power_kw=2.0,
            min_load_fraction=0.20,
            min_on_seconds=60,
            min_off_seconds=30,
        ),
    ]


def parameters() -> TwinParameters:
    return TwinParameters(
        volume_m3=10.0,
        temperature_k=298.15,
    )


def config() -> OptimizationConfig:
    return OptimizationConfig(
        interval_seconds=60.0,
        reserve_mass_flow_kg_s=0.01,
        startup_penalty_kwh=0.05,
        overpressure_penalty_kwh_per_bar_hour=0.10,
        terminal_pressure_min_bar_g=7.0,
        time_limit_seconds=30.0,
        mip_rel_gap=0.001,
    )


def solve(
    demand: list[float],
):
    return optimize_schedule(
        demand,
        leak_mass_flow_kg_s=0.005,
        initial_pressure_bar_g=7.0,
        parameters=parameters(),
        compressors=compressors(),
        target_bar_g=7.0,
        safety_min_bar_g=6.5,
        safety_max_bar_g=7.5,
        config=config(),
    )


def test_optimizer_returns_safe_schedule():
    result = solve(
        [0.07] * 10
        + [0.145] * 10
        + [0.09] * 10
    )

    assert result.energy_kwh >= 0.0

    assert not result.schedule[
        "safety_violation"
    ].any()

    assert (
        result.schedule[
            "pressure_bar_g"
        ].iloc[-1]
        >= 7.0 - 1.0e-8
    )


def test_vsd_fraction_respects_minimum_load():
    result = solve(
        [0.07] * 20
    )

    fractions = result.schedule[
        "vsd_1_fraction"
    ]

    running = fractions[
        fractions > 1.0e-8
    ]

    assert (
        running >= 0.20 - 1.0e-8
    ).all()

    assert (
        fractions <= 1.0 + 1.0e-8
    ).all()


def test_fixed_minimum_on_time_is_respected():
    result = solve(
        [0.145] * 15
        + [0.03] * 15
    )

    schedule = result.schedule

    starts = schedule.index[
        schedule["fixed_1_start"]
    ].tolist()

    assert starts

    for start in starts:
        end = min(
            len(schedule),
            start + 2,
        )

        assert schedule.loc[
            start:end - 1,
            "fixed_1_on",
        ].all()


def test_reserve_capacity_is_respected():
    result = solve(
        [0.11] * 20
    )

    assert (
        result.schedule[
            "reserve_available_kg_s"
        ]
        >= 0.01 - 1.0e-8
    ).all()


def test_higher_demand_requires_more_air():
    low = solve(
        [0.06] * 20
    )

    high = solve(
        [0.12] * 20
    )

    low_supply = low.schedule[
        "compressor_flow_kg_s"
    ].sum()

    high_supply = high.schedule[
        "compressor_flow_kg_s"
    ].sum()

    assert high_supply > low_supply


def test_infeasible_capacity_is_reported():
    with pytest.raises(
        RuntimeError,
        match="optimization failed",
    ):
        solve(
            [0.30] * 5
        )
