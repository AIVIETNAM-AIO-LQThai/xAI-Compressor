import pytest

from ml.control.baseline import (
    BaselineControllerConfig,
    simulate_baseline,
)
from ml.twin.compressor import (
    CompressorSpec,
)
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


def controller() -> BaselineControllerConfig:
    return BaselineControllerConfig(
        target_bar_g=7.0,
        lower_band_bar_g=6.8,
        upper_band_bar_g=7.2,
        safety_min_bar_g=6.5,
        safety_max_bar_g=7.5,
        pressure_gain_kg_s_per_bar=0.025,
    )


def test_low_demand_uses_vsd_only():
    result = simulate_baseline(
        [0.05] * 300,
        leak_mass_flow_kg_s=0.005,
        initial_pressure_bar_g=7.0,
        parameters=parameters(),
        compressors=compressors(),
        controller_config=controller(),
        timestep_seconds=1.0,
    )

    assert not result[
        "fixed_1_on"
    ].any()

    assert not result[
        "fixed_2_on"
    ].any()

    assert result[
        "pressure_bar_g"
    ].min() == pytest.approx(
        7.0
    )

    assert result[
        "pressure_bar_g"
    ].max() == pytest.approx(
        7.0
    )


def test_high_demand_stages_fixed_units():
    result = simulate_baseline(
        [0.145] * 600,
        leak_mass_flow_kg_s=0.005,
        initial_pressure_bar_g=7.0,
        parameters=parameters(),
        compressors=compressors(),
        controller_config=controller(),
        timestep_seconds=1.0,
    )

    assert result[
        "fixed_1_start"
    ].any()

    assert result[
        "fixed_2_start"
    ].any()

    assert not result[
        "safety_violation"
    ].any()


def test_minimum_on_time_is_respected():
    result = simulate_baseline(
        (
            [0.145] * 50
            + [0.03] * 150
        ),
        leak_mass_flow_kg_s=0.005,
        initial_pressure_bar_g=7.0,
        parameters=parameters(),
        compressors=compressors(),
        controller_config=controller(),
        timestep_seconds=1.0,
    )

    start_time = float(
        result.loc[
            result[
                "fixed_1_start"
            ],
            "time_seconds",
        ].iloc[0]
    )

    off_transition = (
        result[
            "fixed_1_on"
        ].shift(
            fill_value=False
        )
        & ~result[
            "fixed_1_on"
        ]
    )

    off_time = float(
        result.loc[
            off_transition,
            "time_seconds",
        ].iloc[0]
    )

    assert (
        off_time - start_time
        >= 120.0
    )


def test_higher_leak_uses_more_energy():
    normal = simulate_baseline(
        [0.07] * 1200,
        leak_mass_flow_kg_s=0.005,
        initial_pressure_bar_g=7.0,
        parameters=parameters(),
        compressors=compressors(),
        controller_config=controller(),
        timestep_seconds=1.0,
    )

    high_leak = simulate_baseline(
        [0.07] * 1200,
        leak_mass_flow_kg_s=0.020,
        initial_pressure_bar_g=7.0,
        parameters=parameters(),
        compressors=compressors(),
        controller_config=controller(),
        timestep_seconds=1.0,
    )

    normal_energy = normal[
        "cumulative_energy_kwh"
    ].iloc[-1]

    high_leak_energy = high_leak[
        "cumulative_energy_kwh"
    ].iloc[-1]

    assert (
        high_leak_energy
        > normal_energy
    )


def test_negative_demand_is_rejected():
    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        simulate_baseline(
            [0.05, -0.01],
            leak_mass_flow_kg_s=0.005,
            initial_pressure_bar_g=7.0,
            parameters=parameters(),
            compressors=compressors(),
            controller_config=controller(),
            timestep_seconds=1.0,
        )