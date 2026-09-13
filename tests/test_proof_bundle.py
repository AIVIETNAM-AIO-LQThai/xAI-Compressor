from __future__ import annotations

import pytest

from ml.optimization.robust_mpc import (
    RobustActionEvaluation,
    RobustFirstActionResult,
)
from ml.optimization.runtime_state import (
    CompressorRuntimeState,
)
from ml.reasoning.evidence import (
    GroupEvidence,
    IncidentEvidence,
)
from ml.reasoning.hypotheses import (
    CandidateHypothesis,
)
from ml.reasoning.physics_verifier import (
    PhysicsHypothesisResult,
)
from ml.reasoning.proof_bundle import (
    ProofBundleConfig,
    build_proof_bundle,
)
from ml.reasoning.scenarios import (
    PhysicalScenario,
    PhysicalScenarioSet,
)
from ml.reasoning.uncertainty import (
    IntervalEstimate,
    PhysicalStateUncertainty,
)


def _detector_evidence() -> IncidentEvidence:
    return IncidentEvidence(
        incident_id=1,
        incident_start=(
            "2020-04-18 00:00:00"
        ),
        explanation_timestamp=(
            "2020-04-18 00:40:00"
        ),
        timing="post_onset",
        groups=(
            GroupEvidence(
                group="pressure",
                share=0.60,
                calibration_percentile=0.99,
                counterfactual_alert_cleared=True,
            ),
        ),
        temporal_window_complete=True,
        temporal_bins_requested=12,
        temporal_bins_observed=12,
        causal_claim=False,
    )


def _hypotheses() -> tuple[
    CandidateHypothesis,
    ...
]:
    return (
        CandidateHypothesis(
            hypothesis_id=(
                "leak_like_persistent_outflow"
            ),
            label="Leak-like persistent outflow",
            matched_groups=("pressure",),
            strongest_share=0.60,
            strongest_percentile=0.99,
            counterfactual_support_groups=(
                "pressure",
            ),
            evidence_class=(
                "MODEL_INFERENCE_FROM_REAL"
            ),
            causal_claim=False,
        ),
    )


def _physics_results() -> tuple[
    PhysicsHypothesisResult,
    ...
]:
    return (
        PhysicsHypothesisResult(
            hypothesis_id=(
                "leak_like_persistent_outflow"
            ),
            status=(
                "observationally_equivalent"
            ),
            identifiable=False,
            estimated_level_kg_s=None,
            estimated_change_kg_s=0.01,
            required_observations=(
                "independent_process_demand",
            ),
            note=(
                "Leak and demand cannot be "
                "separated by receiver pressure "
                "alone."
            ),
            evidence_class=(
                "PHYSICS_MODEL_INFERENCE"
            ),
            causal_claim=False,
        ),
    )


def _state() -> PhysicalStateUncertainty:
    return PhysicalStateUncertainty(
        total_outflow_kg_s=IntervalEstimate(
            lower=0.106,
            center=0.110,
            upper=0.114,
        ),
        demand_kg_s=None,
        leak_kg_s=None,
        leak_identifiable=False,
        assumptions=(
            (
                "Receiver dynamics constrain "
                "total outflow only."
            ),
        ),
        evidence_class=(
            "PHYSICS_MODEL_INFERENCE"
        ),
        causal_claim=False,
    )


def _scenarios() -> PhysicalScenarioSet:
    scenarios = tuple(
        PhysicalScenario(
            id=label,
            total_outflow_kg_s=value,
            demand_kg_s=None,
            leak_kg_s=None,
            allocation_identifiable=False,
            source=(
                "bounded_total_outflow"
            ),
            evidence_class=(
                "PHYSICS_MODEL_INFERENCE"
            ),
            causal_claim=False,
        )
        for label, value in (
            ("outflow_low", 0.106),
            ("outflow_center", 0.110),
            ("outflow_high", 0.114),
        )
    )

    return PhysicalScenarioSet(
        scenarios=scenarios,
        allocation_identifiable=False,
        evidence_class=(
            "PHYSICS_MODEL_INFERENCE"
        ),
        causal_claim=False,
    )


def _action(
    *,
    robust_safe: bool = True,
    scenario_count: int = 3,
) -> RobustFirstActionResult:
    source_scenarios = (
        "outflow_low",
        "outflow_center",
        "outflow_high",
    )

    selected = RobustActionEvaluation(
        commands=(
            ("fixed_1", 1.0),
            ("fixed_2", 0.0),
            ("vsd_1", 0.75),
        ),
        source_scenarios=(
            source_scenarios
        ),
        interval_energy_kwh=0.44,
        startup_count=2,
        worst_case_min_pressure_bar_g=6.91,
        worst_case_max_pressure_bar_g=7.08,
        minimum_reserve_kg_s=0.026,
        objective_value=0.54,
        robust_safe=robust_safe,
    )

    runtime_state = (
        CompressorRuntimeState(
            compressor_id="fixed_1",
            is_on=True,
            seconds_in_state=60.0,
        ),
        CompressorRuntimeState(
            compressor_id="fixed_2",
            is_on=False,
            seconds_in_state=120.0,
        ),
        CompressorRuntimeState(
            compressor_id="vsd_1",
            is_on=True,
            seconds_in_state=60.0,
        ),
    )

    return RobustFirstActionResult(
        selected=selected,
        candidates=(selected,),
        scenario_count=scenario_count,
        horizon_intervals=10,
        valid_for_seconds=60.0,
        robust_horizon_objective_value=4.8,
        robust_horizon_energy_kwh=4.4,
        robust_horizon_startup_count=2,
        method=(
            "shared_action_robust_milp_"
            "with_one_step_safety_gate"
        ),
        evidence_class="SIMULATED",
        causal_claim=False,
        next_runtime_state=runtime_state,
    )


def test_proof_bundle_preserves_evidence_classes():
    bundle = build_proof_bundle(
        detector_evidence=(
            _detector_evidence()
        ),
        hypotheses=_hypotheses(),
        physics_results=(
            _physics_results()
        ),
        state=_state(),
        scenarios=_scenarios(),
        action=_action(),
    )

    assert (
        bundle[
            "provenance"
        ][
            "detector_evidence"
        ]
        == "REAL"
    )

    assert (
        bundle[
            "provenance"
        ][
            "candidate_hypotheses"
        ]
        == "MODEL_INFERENCE_FROM_REAL"
    )

    assert (
        bundle[
            "provenance"
        ][
            "physics_verification"
        ]
        == "PHYSICS_MODEL_INFERENCE"
    )

    assert (
        bundle[
            "provenance"
        ][
            "recommendation"
        ]
        == "SIMULATED"
    )

    assert (
        bundle[
            "causal_claim"
        ]
        is False
    )

    assert (
        bundle[
            "deployment"
        ][
            "override_equipment_ctrl"
        ]
        is False
    )

    assert (
        bundle[
            "recommendation"
        ][
            "robust_safe"
        ]
        is True
    )

    assert (
        bundle[
            "recommendation"
        ][
            "valid_for_seconds"
        ]
        == 60.0
    )


def test_proof_bundle_is_deterministic():
    kwargs = {
        "detector_evidence": (
            _detector_evidence()
        ),
        "hypotheses": (
            _hypotheses()
        ),
        "physics_results": (
            _physics_results()
        ),
        "state": _state(),
        "scenarios": (
            _scenarios()
        ),
        "action": _action(),
    }

    first = build_proof_bundle(
        **kwargs
    )

    second = build_proof_bundle(
        **kwargs
    )

    assert (
        first["bundle_id"]
        == second["bundle_id"]
    )


def test_proof_bundle_rejects_unsafe_action():
    with pytest.raises(
        ValueError,
        match="failed the robust safety gate",
    ):
        build_proof_bundle(
            detector_evidence=(
                _detector_evidence()
            ),
            hypotheses=_hypotheses(),
            physics_results=(
                _physics_results()
            ),
            state=_state(),
            scenarios=_scenarios(),
            action=_action(
                robust_safe=False
            ),
        )


def test_proof_bundle_rejects_scenario_mismatch():
    with pytest.raises(
        ValueError,
        match="scenario count",
    ):
        build_proof_bundle(
            detector_evidence=(
                _detector_evidence()
            ),
            hypotheses=_hypotheses(),
            physics_results=(
                _physics_results()
            ),
            state=_state(),
            scenarios=_scenarios(),
            action=_action(
                scenario_count=2
            ),
        )


def test_proof_bundle_rejects_control_override():
    with pytest.raises(
        ValueError,
        match="cannot authorize",
    ):
        ProofBundleConfig(
            deployment_mode="advisory",
            override_equipment_ctrl=True,
        )


def test_unidentified_leak_remains_unidentified():
    bundle = build_proof_bundle(
        detector_evidence=(
            _detector_evidence()
        ),
        hypotheses=_hypotheses(),
        physics_results=(
            _physics_results()
        ),
        state=_state(),
        scenarios=_scenarios(),
        action=_action(),
    )

    physical_state = bundle[
        "physical_state"
    ]

    assert (
        physical_state[
            "leak_identifiable"
        ]
        is False
    )

    assert (
        physical_state[
            "leak_kg_s"
        ]
        is None
    )

    assert any(
        "not site-calibrated"
        in boundary
        for boundary in bundle[
            "claim_boundaries"
        ]
    )
