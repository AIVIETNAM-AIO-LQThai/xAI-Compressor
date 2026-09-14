from __future__ import annotations

import pytest

from ml.reasoning.evidence_action_bridge import (
    EvidenceActionBridgeConfig,
    assess_evidence_action_bridge,
)


def test_uncalibrated_bridge_withholds_real_advisory():
    assessment = assess_evidence_action_bridge(
        EvidenceActionBridgeConfig(
            site_calibrated=False,
            calibrated_compressor_inflow_available=False,
            receiver_pressure_trace_available=False,
        )
    )

    assert (
        assessment.may_generate_real_asset_advisory
        is False
    )
    assert assessment.status.startswith(
        "WITHHOLD_REAL_ASSET_ADVISORY"
    )
    assert (
        "site_calibrated_receiver_and_compressor_model"
        in assessment.missing_requirements
    )
    assert assessment.causal_claim is False


def test_partial_bridge_still_withholds_real_advisory():
    assessment = assess_evidence_action_bridge(
        EvidenceActionBridgeConfig(
            site_calibrated=True,
            calibrated_compressor_inflow_available=True,
            receiver_pressure_trace_available=False,
        )
    )

    assert (
        assessment.may_generate_real_asset_advisory
        is False
    )
    assert assessment.missing_requirements == (
        "receiver_pressure_trace",
    )


def test_complete_bridge_allows_advisory_reasoning_only():
    assessment = assess_evidence_action_bridge(
        EvidenceActionBridgeConfig(
            site_calibrated=True,
            calibrated_compressor_inflow_available=True,
            receiver_pressure_trace_available=True,
        )
    )

    assert (
        assessment.may_generate_real_asset_advisory
        is True
    )
    assert (
        assessment.status
        == "REAL_ASSET_PHYSICAL_BRIDGE_AVAILABLE"
    )
    assert assessment.missing_requirements == ()
    assert assessment.causal_claim is False


def test_bridge_rejects_control_override():
    with pytest.raises(
        ValueError,
        match="cannot authorize equipment control",
    ):
        EvidenceActionBridgeConfig(
            site_calibrated=True,
            calibrated_compressor_inflow_available=True,
            receiver_pressure_trace_available=True,
            override_equipment_ctrl=True,
        )
