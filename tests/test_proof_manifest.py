from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import (
    TestClient,
)

from backend.app.main import app
from backend.app.proof_manifest import (
    build_proof_manifest,
    build_proof_manifest_from_payloads,
)

ROOT = Path(__file__).resolve().parents[1]
client = TestClient(app)


def _payloads() -> tuple[
    dict,
    dict,
]:
    real = json.loads(
        (
            ROOT
            / "docs"
            / "evidence_action_replay.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    simulated = json.loads(
        (
            ROOT
            / "docs"
            / "simulated_action_proof.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    return (
        real,
        simulated,
    )


def test_manifest_keeps_real_and_simulated_paths_separate():
    manifest = (
        build_proof_manifest()
    )

    real = manifest[
        "real_evidence_path"
    ]

    simulated = manifest[
        "simulated_validation_path"
    ]

    assert (
        real[
            "physical_bridge_available"
        ]
        is False
    )

    assert (
        real[
            "downstream_physics_executed"
        ]
        is False
    )

    assert (
        real[
            "robust_control_executed"
        ]
        is False
    )

    assert (
        real[
            "recommendation"
        ]
        is None
    )

    assert (
        simulated[
            "scope"
        ]
        == "SIMULATION_ONLY"
    )

    assert (
        simulated[
            "connected_to_real_incident"
        ]
        is False
    )

    assert (
        simulated[
            "robust_action"
        ][
            "robust_safe"
        ]
        is True
    )

    assert (
        manifest[
            "causal_claim"
        ]
        is False
    )


def test_manifest_id_is_deterministic():
    real, simulated = (
        _payloads()
    )

    first = (
        build_proof_manifest_from_payloads(
            real=real,
            simulated=simulated,
        )
    )

    second = (
        build_proof_manifest_from_payloads(
            real=real,
            simulated=simulated,
        )
    )

    assert (
        first[
            "manifest_id"
        ]
        == second[
            "manifest_id"
        ]
    )


def test_manifest_rejects_fake_real_connection():
    real, simulated = (
        _payloads()
    )

    contaminated = copy.deepcopy(
        simulated
    )

    contaminated[
        "connected_to_real_incident"
    ] = True

    with pytest.raises(
        ValueError,
        match=(
            "must not be connected "
            "to a real incident"
        ),
    ):
        build_proof_manifest_from_payloads(
            real=real,
            simulated=contaminated,
        )


def test_proof_manifest_api():
    response = client.get(
        "/evidence/proof-manifest"
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload[
            "project"
        ]
        == "AeroXAI"
    )

    assert (
        payload[
            "real_evidence_path"
        ][
            "recommendation"
        ]
        is None
    )

    assert (
        payload[
            "simulated_validation_path"
        ][
            "connected_to_real_incident"
        ]
        is False
    )

    assert (
        payload[
            "deployment"
        ][
            "override_equipment_ctrl"
        ]
        is False
    )
