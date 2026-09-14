from __future__ import annotations

import pytest

from ml.reasoning.inverse_physics import infer_total_outflow
from ml.reasoning.simulation_validation import (
    assess_outflow_recovery,
    generate_synthetic_pressure_trace,
)
from ml.twin.physics import TwinParameters

PARAMETERS = TwinParameters(
    volume_m3=10.0,
    temperature_k=298.15,
)


def test_synthetic_trace_recovers_total_outflow():
    inflow = (0.105, 0.115, 0.108, 0.112)
    truth = 0.110

    pressure = generate_synthetic_pressure_trace(
        initial_pressure_bar_g=7.0,
        inflow_profile_kg_s=inflow,
        total_outflow_kg_s=truth,
        interval_seconds=60.0,
        parameters=PARAMETERS,
    )

    inference = infer_total_outflow(
        pressure,
        inflow,
        interval_seconds=60.0,
        parameters=PARAMETERS,
    )

    recovery = assess_outflow_recovery(
        inference,
        truth_total_outflow_kg_s=truth,
        tolerance_kg_s=1.0e-9,
    )

    assert recovery.passed is True
    assert recovery.inferred_total_outflow_kg_s == pytest.approx(
        truth,
        abs=1.0e-9,
    )
    assert recovery.evidence_class == "SIMULATED"
    assert recovery.causal_claim is False


def test_synthetic_trace_rejects_negative_outflow():
    with pytest.raises(ValueError, match="cannot be negative"):
        generate_synthetic_pressure_trace(
            initial_pressure_bar_g=7.0,
            inflow_profile_kg_s=(0.1,),
            total_outflow_kg_s=-0.1,
            interval_seconds=60.0,
            parameters=PARAMETERS,
        )
