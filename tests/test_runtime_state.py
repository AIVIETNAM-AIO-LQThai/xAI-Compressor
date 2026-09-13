from __future__ import annotations

import pytest

from ml.optimization.runtime_state import (
    CompressorRuntimeState,
    advance_runtime_state,
    resolve_runtime_state,
)
from ml.twin.compressor import CompressorSpec

COMPRESSORS = [
    CompressorSpec(
        id="fixed_1",
        kind="fixed",
        max_mass_flow_kg_s=0.060,
        rated_power_kw=13.0,
        idle_power_kw=0.0,
        min_load_fraction=1.0,
        min_on_seconds=120.0,
        min_off_seconds=60.0,
    ),
    CompressorSpec(
        id="vsd_1",
        kind="vsd",
        max_mass_flow_kg_s=0.080,
        rated_power_kw=19.0,
        idle_power_kw=2.0,
        min_load_fraction=0.20,
        min_on_seconds=60.0,
        min_off_seconds=30.0,
    ),
]


def test_default_runtime_state_is_available_off():
    states = resolve_runtime_state(
        COMPRESSORS,
        None,
    )

    assert [
        state.is_on
        for state in states
    ] == [False, False]

    assert states[0].seconds_in_state == (
        pytest.approx(60.0)
    )

    assert states[1].seconds_in_state == (
        pytest.approx(30.0)
    )


def test_runtime_state_advances_same_state():
    previous = (
        CompressorRuntimeState(
            compressor_id="fixed_1",
            is_on=True,
            seconds_in_state=60.0,
        ),
        CompressorRuntimeState(
            compressor_id="vsd_1",
            is_on=False,
            seconds_in_state=30.0,
        ),
    )

    updated = advance_runtime_state(
        previous,
        {
            "fixed_1": 1.0,
            "vsd_1": 0.0,
        },
        interval_seconds=60.0,
        compressors=COMPRESSORS,
    )

    assert updated[0].is_on is True
    assert updated[0].seconds_in_state == (
        pytest.approx(120.0)
    )

    assert updated[1].is_on is False
    assert updated[1].seconds_in_state == (
        pytest.approx(90.0)
    )


def test_runtime_state_resets_timer_on_transition():
    previous = (
        CompressorRuntimeState(
            compressor_id="fixed_1",
            is_on=False,
            seconds_in_state=300.0,
        ),
        CompressorRuntimeState(
            compressor_id="vsd_1",
            is_on=True,
            seconds_in_state=300.0,
        ),
    )

    updated = advance_runtime_state(
        previous,
        {
            "fixed_1": 1.0,
            "vsd_1": 0.0,
        },
        interval_seconds=60.0,
        compressors=COMPRESSORS,
    )

    assert updated[0].is_on is True
    assert updated[0].seconds_in_state == (
        pytest.approx(60.0)
    )

    assert updated[1].is_on is False
    assert updated[1].seconds_in_state == (
        pytest.approx(60.0)
    )


def test_runtime_state_requires_exact_fleet_ids():
    with pytest.raises(
        ValueError,
        match="exactly match",
    ):
        resolve_runtime_state(
            COMPRESSORS,
            (
                CompressorRuntimeState(
                    compressor_id=(
                        "fixed_1"
                    ),
                    is_on=False,
                    seconds_in_state=60.0,
                ),
            ),
        )
