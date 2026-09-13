from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.reasoning import (
    generate_candidate_hypotheses,
    incident_evidence_from_report,
    load_hypothesis_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _verification_report() -> dict:
    path = (
        ROOT
        / "docs"
        / "xai_verification_report.json"
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def test_hypothesis_config_is_valid():
    generation, specs = (
        load_hypothesis_config(
            ROOT
            / "configs"
            / "waste_hypotheses.yaml"
        )
    )

    assert generation.top_groups == 5
    assert len(specs) >= 6

    ids = {
        spec.id
        for spec in specs
    }

    assert (
        "leak_like_persistent_outflow"
        in ids
    )
    assert "demand_surge" in ids
    assert "sensor_inconsistency" in ids


def test_incident_two_evidence_is_reproduced():
    evidence = (
        incident_evidence_from_report(
            _verification_report(),
            incident_id=2,
        )
    )

    dominant = evidence.dominant_group

    assert dominant.group == "dv_pressure"

    assert dominant.share == pytest.approx(
        0.7756286,
        rel=1.0e-5,
    )

    assert (
        dominant.calibration_percentile
        == pytest.approx(
            0.995545657,
            rel=1.0e-6,
        )
    )

    assert (
        dominant
        .counterfactual_alert_cleared
        is True
    )

    assert evidence.causal_claim is False


def test_incident_one_preserves_gap():
    evidence = (
        incident_evidence_from_report(
            _verification_report(),
            incident_id=1,
        )
    )

    assert (
        evidence.temporal_bins_requested
        == 12
    )
    assert (
        evidence.temporal_bins_observed
        == 10
    )
    assert (
        evidence.temporal_window_complete
        is False
    )


def test_candidates_are_not_probabilities_or_causes():
    generation, specs = (
        load_hypothesis_config(
            ROOT
            / "configs"
            / "waste_hypotheses.yaml"
        )
    )

    evidence = (
        incident_evidence_from_report(
            _verification_report(),
            incident_id=2,
        )
    )

    candidates = (
        generate_candidate_hypotheses(
            evidence,
            specs,
            generation,
        )
    )

    ids = {
        item.hypothesis_id
        for item in candidates
    }

    assert (
        "leak_like_persistent_outflow"
        in ids
    )

    assert "demand_surge" in ids

    for candidate in candidates:
        assert (
            candidate.evidence_class
            == "MODEL_INFERENCE_FROM_REAL"
        )
        assert (
            candidate.causal_claim
            is False
        )


def test_unknown_incident_is_rejected():
    with pytest.raises(
        ValueError,
        match="Unknown incident id",
    ):
        incident_evidence_from_report(
            _verification_report(),
            incident_id=999,
        )