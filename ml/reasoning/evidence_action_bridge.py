from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceActionBridgeConfig:
    site_calibrated: bool
    calibrated_compressor_inflow_available: bool
    receiver_pressure_trace_available: bool
    deployment_mode: str = "advisory"
    override_equipment_ctrl: bool = False

    def __post_init__(self) -> None:
        if self.deployment_mode != "advisory":
            raise ValueError(
                "AeroXAI evidence-to-action bridge "
                "currently supports advisory mode only."
            )

        if self.override_equipment_ctrl:
            raise ValueError(
                "The evidence-to-action bridge cannot "
                "authorize equipment control."
            )


@dataclass(frozen=True)
class EvidenceActionBridgeAssessment:
    status: str
    may_generate_real_asset_advisory: bool
    missing_requirements: tuple[str, ...]
    note: str
    causal_claim: bool


def assess_evidence_action_bridge(
    config: EvidenceActionBridgeConfig,
) -> EvidenceActionBridgeAssessment:
    missing: list[str] = []

    if not config.site_calibrated:
        missing.append(
            "site_calibrated_receiver_and_compressor_model"
        )

    if not config.calibrated_compressor_inflow_available:
        missing.append(
            "calibrated_compressor_inflow_observation"
        )

    if not config.receiver_pressure_trace_available:
        missing.append(
            "receiver_pressure_trace"
        )

    if missing:
        return EvidenceActionBridgeAssessment(
            status=(
                "WITHHOLD_REAL_ASSET_ADVISORY_"
                "MISSING_VALIDATED_PHYSICAL_BRIDGE"
            ),
            may_generate_real_asset_advisory=False,
            missing_requirements=tuple(missing),
            note=(
                "Real detector evidence can support "
                "candidate hypotheses, but AeroXAI "
                "must not map those hypotheses to a "
                "real-asset physical state or control "
                "advisory until the receiver/compressor "
                "model is site-calibrated and the "
                "required physical observations are "
                "available."
            ),
            causal_claim=False,
        )

    return EvidenceActionBridgeAssessment(
        status="REAL_ASSET_PHYSICAL_BRIDGE_AVAILABLE",
        may_generate_real_asset_advisory=True,
        missing_requirements=(),
        note=(
            "The minimum physical bridge requirements "
            "are present. This permits model-based "
            "advisory reasoning, not automatic equipment "
            "control and not a causal diagnosis."
        ),
        causal_claim=False,
    )
