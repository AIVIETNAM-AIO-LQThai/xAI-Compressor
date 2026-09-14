from __future__ import annotations

import copy

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.proof_bundle import (
    build_proof_bundle,
    verify_proof_bundle,
)

client = TestClient(app)


def test_fresh_proof_bundle_verifies() -> None:
    bundle = build_proof_bundle()

    verification = verify_proof_bundle(
        bundle
    )

    assert verification["verified"] is True
    assert all(
        verification["checks"].values()
    )
    assert verification["errors"] == []


def test_bundle_contains_exact_two_source_artifacts() -> None:
    bundle = build_proof_bundle()

    artifacts = bundle["artifacts"]

    assert {
        item["role"]
        for item in artifacts
    } == {
        "real_evidence_replay",
        "simulated_action_proof",
    }

    assert {
        item["evidence_class"]
        for item in artifacts
    } == {
        "REAL",
        "SIMULATED",
    }

    assert all(
        len(item["canonical_sha256"]) == 64
        for item in artifacts
    )
    assert all(
        len(item["file_sha256"]) == 64
        for item in artifacts
    )


def test_tampered_embedded_payload_fails_verification() -> None:
    bundle = build_proof_bundle()
    tampered = copy.deepcopy(bundle)

    simulated = next(
        item
        for item in tampered["artifacts"]
        if item["role"]
        == "simulated_action_proof"
    )

    simulated["payload"][
        "connected_to_real_incident"
    ] = True

    verification = verify_proof_bundle(
        tampered,
        check_current_sources=False,
    )

    assert verification["verified"] is False
    assert (
        verification["checks"]["bundle_id"]
        is False
    )
    assert (
        verification["checks"][
            "embedded_payload_digests"
        ]
        is False
    )
    assert (
        verification["checks"][
            "semantic_rebuild"
        ]
        is False
    )


def test_tampered_manifest_fails_verification() -> None:
    bundle = build_proof_bundle()
    tampered = copy.deepcopy(bundle)

    tampered["manifest"][
        "causal_claim"
    ] = True

    verification = verify_proof_bundle(
        tampered,
        check_current_sources=False,
    )

    assert verification["verified"] is False
    assert (
        verification["checks"]["bundle_id"]
        is False
    )
    assert (
        verification["checks"][
            "manifest_digest"
        ]
        is False
    )
    assert (
        verification["checks"][
            "semantic_rebuild"
        ]
        is False
    )


def test_proof_bundle_api() -> None:
    response = client.get(
        "/evidence/proof-bundle"
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload["schema_version"]
        == "aeroxai.proof_bundle.v1"
    )
    assert payload["bundle_id"].startswith(
        "pbundle_"
    )
    assert len(payload["artifacts"]) == 2


def test_proof_bundle_verify_api() -> None:
    bundle_response = client.get(
        "/evidence/proof-bundle"
    )
    bundle = bundle_response.json()

    response = client.post(
        "/evidence/proof-bundle/verify",
        json=bundle,
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["verified"] is True
    assert payload["errors"] == []
