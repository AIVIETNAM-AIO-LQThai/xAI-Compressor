from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ml.optimization.robust_mpc import (
    RobustFirstActionResult,
)
from ml.reasoning.evidence import IncidentEvidence
from ml.reasoning.hypotheses import CandidateHypothesis
from ml.reasoning.physics_verifier import (
    PhysicsHypothesisResult,
)
from ml.reasoning.scenarios import PhysicalScenarioSet
from ml.reasoning.uncertainty import (
    PhysicalStateUncertainty,
)


@dataclass(frozen=True)
class ProofBundleConfig:
    deployment_mode: str = "advisory"
    override_equipment_ctrl: bool = False
    site_calibrated: bool = False

    def __post_init__(self) -> None:
        if self.deployment_mode != "advisory":
            raise ValueError(
                "Proof bundles currently support "
                "advisory deployment only."
            )

        if self.override_equipment_ctrl:
            raise ValueError(
                "AeroXAI proof bundles cannot "
                "authorize equipment control."
            )


def _validate_noncausal(
    *,
    detector_evidence: IncidentEvidence,
    hypotheses: tuple[CandidateHypothesis, ...],
    physics_results: tuple[
        PhysicsHypothesisResult,
        ...
    ],
    state: PhysicalStateUncertainty,
    scenarios: PhysicalScenarioSet,
    action: RobustFirstActionResult,
) -> None:
    if detector_evidence.causal_claim:
        raise ValueError(
            "Detector evidence cannot contain "
            "a causal claim."
        )

    for hypothesis in hypotheses:
        if hypothesis.causal_claim:
            raise ValueError(
                "Candidate hypotheses cannot "
                "contain causal claims."
            )

        if (
            hypothesis.evidence_class
            != "MODEL_INFERENCE_FROM_REAL"
        ):
            raise ValueError(
                "Candidate hypotheses must use "
                "MODEL_INFERENCE_FROM_REAL."
            )

    for result in physics_results:
        if result.causal_claim:
            raise ValueError(
                "Physics verification cannot "
                "contain causal claims."
            )

        if (
            result.evidence_class
            != "PHYSICS_MODEL_INFERENCE"
        ):
            raise ValueError(
                "Physics verification must use "
                "PHYSICS_MODEL_INFERENCE."
            )

    if state.causal_claim:
        raise ValueError(
            "Physical state cannot contain "
            "a causal claim."
        )

    if (
        state.evidence_class
        != "PHYSICS_MODEL_INFERENCE"
    ):
        raise ValueError(
            "Physical state must use "
            "PHYSICS_MODEL_INFERENCE."
        )

    if scenarios.causal_claim:
        raise ValueError(
            "Scenario set cannot contain "
            "a causal claim."
        )

    if (
        scenarios.evidence_class
        != "PHYSICS_MODEL_INFERENCE"
    ):
        raise ValueError(
            "Scenario set must use "
            "PHYSICS_MODEL_INFERENCE."
        )

    if action.causal_claim:
        raise ValueError(
            "Robust action cannot contain "
            "a causal claim."
        )

    if action.evidence_class != "SIMULATED":
        raise ValueError(
            "Robust action must use SIMULATED."
        )


def _validate_action_proof(
    scenarios: PhysicalScenarioSet,
    action: RobustFirstActionResult,
) -> None:
    if not action.selected.robust_safe:
        raise ValueError(
            "Cannot issue a proof bundle for "
            "an action that failed the robust "
            "safety gate."
        )

    if (
        action.scenario_count
        != len(scenarios.scenarios)
    ):
        raise ValueError(
            "Action scenario count does not "
            "match the physical scenario set."
        )

    scenario_ids = {
        scenario.id
        for scenario in scenarios.scenarios
    }

    action_ids = set(
        action.selected.source_scenarios
    )

    if action_ids != scenario_ids:
        raise ValueError(
            "Action proof does not cover exactly "
            "the physical scenario set."
        )

    if action.valid_for_seconds <= 0.0:
        raise ValueError(
            "Action validity must be positive."
        )


def _recommendation_payload(
    action: RobustFirstActionResult,
) -> dict[str, Any]:
    selected = action.selected

    return {
        "status": (
            "ROBUST_MODEL_CHECKED_ADVISORY"
        ),
        "commands": {
            compressor_id: value
            for compressor_id, value
            in selected.commands
        },
        "valid_for_seconds": (
            action.valid_for_seconds
        ),
        "robust_safe": (
            selected.robust_safe
        ),
        "scenario_count": (
            action.scenario_count
        ),
        "source_scenarios": list(
            selected.source_scenarios
        ),
        "first_interval": {
            "energy_kwh": (
                selected
                .interval_energy_kwh
            ),
            "startup_count": (
                selected.startup_count
            ),
            "worst_case_min_pressure_bar_g": (
                selected
                .worst_case_min_pressure_bar_g
            ),
            "worst_case_max_pressure_bar_g": (
                selected
                .worst_case_max_pressure_bar_g
            ),
            "minimum_reserve_kg_s": (
                selected
                .minimum_reserve_kg_s
            ),
            "objective_value": (
                selected.objective_value
            ),
        },
        "horizon": {
            "intervals": (
                action.horizon_intervals
            ),
            "energy_kwh": (
                action
                .robust_horizon_energy_kwh
            ),
            "startup_count": (
                action
                .robust_horizon_startup_count
            ),
            "objective_value": (
                action
                .robust_horizon_objective_value
            ),
        },
        "method": action.method,
        "evidence_class": (
            action.evidence_class
        ),
    }


def _claim_boundaries(
    config: ProofBundleConfig,
) -> list[str]:
    boundaries = [
        (
            "Detector evidence identifies "
            "signals contributing to the "
            "frozen detector; it does not "
            "establish physical root cause."
        ),
        (
            "Waste hypotheses are candidate "
            "explanations inferred from real "
            "detector evidence, not diagnoses."
        ),
        (
            "Physics verification and physical "
            "state estimates are model "
            "inferences, not direct measurements."
        ),
        (
            "Robust action safety and energy "
            "values are simulated under the "
            "current twin, uncertainty set, "
            "and compressor constraints."
        ),
        (
            "The recommendation is advisory "
            "only and does not override the "
            "equipment controller."
        ),
    ]

    if not config.site_calibrated:
        boundaries.append(
            
                "The current twin is not "
                "site-calibrated to the real "
                "MetroPT asset; simulated "
                "control results must not be "
                "presented as measured site "
                "energy savings."
            
        )

    return boundaries


def _canonical_digest(
    payload: dict[str, Any],
) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


def build_proof_bundle(
    *,
    detector_evidence: IncidentEvidence,
    hypotheses: tuple[
        CandidateHypothesis,
        ...
    ],
    physics_results: tuple[
        PhysicsHypothesisResult,
        ...
    ],
    state: PhysicalStateUncertainty,
    scenarios: PhysicalScenarioSet,
    action: RobustFirstActionResult,
    config: ProofBundleConfig | None = None,
) -> dict[str, Any]:
    if config is None:
        config = ProofBundleConfig()

    _validate_noncausal(
        detector_evidence=(
            detector_evidence
        ),
        hypotheses=hypotheses,
        physics_results=physics_results,
        state=state,
        scenarios=scenarios,
        action=action,
    )

    _validate_action_proof(
        scenarios,
        action,
    )

    payload: dict[str, Any] = {
        "schema_version": "aeroxai.proof_bundle.v1",
        "causal_claim": False,
        "deployment": {
            "mode": config.deployment_mode,
            "override_equipment_ctrl": (
                config
                .override_equipment_ctrl
            ),
            "site_calibrated": config.site_calibrated,
        },
        "provenance": {
            "detector_evidence": "REAL",
            "candidate_hypotheses": "MODEL_INFERENCE_FROM_REAL",
            "physics_verification": "PHYSICS_MODEL_INFERENCE",
            "physical_state": "PHYSICS_MODEL_INFERENCE",
            "scenario_set": "PHYSICS_MODEL_INFERENCE",
            "recommendation": "SIMULATED",
        },
        "detector_evidence": {
            **asdict(detector_evidence),
            "evidence_class": "REAL",
        },
        "candidate_hypotheses": [
            asdict(item)
            for item in hypotheses
        ],
        "physics_verification": [
            asdict(item)
            for item in physics_results
        ],
        "physical_state": asdict(state),
        "scenario_set": asdict(scenarios),
        "recommendation": _recommendation_payload(action),
        "claim_boundaries": _claim_boundaries(config),
    }

    digest = _canonical_digest(payload)

    payload["bundle_id"] = (f"pb_{digest[:16]}")

    return payload


def write_proof_bundle(
    bundle: dict[str, Any],
    path: str | Path,
) -> Path:
    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            bundle,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_path
