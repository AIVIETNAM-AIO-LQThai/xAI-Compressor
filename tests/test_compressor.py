import pytest

from ml.twin.compressor import (
    CompressorSpec,
    fixed_operating_point,
    fleet_operating_point,
    vsd_operating_point,
)


def fixed_spec() -> CompressorSpec:
    return CompressorSpec(
        id="fixed_1",
        kind="fixed",
        max_mass_flow_kg_s=0.06,
        rated_power_kw=13.0,
        idle_power_kw=0.0,
        min_load_fraction=1.0,
        min_on_seconds=120,
        min_off_seconds=60,
    )


def vsd_spec() -> CompressorSpec:
    return CompressorSpec(
        id="vsd_1",
        kind="vsd",
        max_mass_flow_kg_s=0.08,
        rated_power_kw=19.0,
        idle_power_kw=2.0,
        min_load_fraction=0.20,
        min_on_seconds=60,
        min_off_seconds=30,
    )


def test_fixed_compressor_off():
    point = fixed_operating_point(
        fixed_spec(),
        is_on=False,
    )

    assert point.mass_flow_kg_s == 0.0
    assert point.power_kw == 0.0
    assert not point.is_on


def test_fixed_compressor_on():
    point = fixed_operating_point(
        fixed_spec(),
        is_on=True,
    )

    assert point.mass_flow_kg_s == pytest.approx(
        0.06
    )

    assert point.power_kw == pytest.approx(
        13.0
    )

    assert point.load_fraction == 1.0


def test_vsd_half_load():
    point = vsd_operating_point(
        vsd_spec(),
        load_fraction=0.5,
    )

    assert point.mass_flow_kg_s == pytest.approx(
        0.04
    )

    assert point.power_kw == pytest.approx(
        10.5
    )


def test_vsd_below_minimum_load_rejected():
    with pytest.raises(
        ValueError,
        match="below min_load_fraction",
    ):
        vsd_operating_point(
            vsd_spec(),
            load_fraction=0.10,
        )


def test_fixed_fractional_command_rejected():
    with pytest.raises(
        ValueError,
        match="0.0 or 1.0",
    ):
        fleet_operating_point(
            [fixed_spec()],
            load_commands={
                "fixed_1": 0.5,
            },
        )


def test_fleet_totals():
    fixed = fixed_spec()
    vsd = vsd_spec()

    fleet = fleet_operating_point(
        [
            fixed,
            vsd,
        ],
        load_commands={
            "fixed_1": 1.0,
            "vsd_1": 0.5,
        },
    )

    assert (
        fleet.total_mass_flow_kg_s
        == pytest.approx(0.10)
    )

    assert (
        fleet.total_power_kw
        == pytest.approx(23.5)
    )