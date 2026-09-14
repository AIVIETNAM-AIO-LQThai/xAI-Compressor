from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from backend.app.canonical_json import canonical_sha256
from backend.app.proof_manifest import (
    build_proof_manifest_from_payloads,
)
from backend.app.runtime import REPO_ROOT

REAL_SOURCE = REPO_ROOT / "docs" / "evidence_action_replay.json"
SIMULATED_SOURCE = (
    REPO_ROOT / "docs" / "simulated_action_proof.json"
)

SOURCE_SPECS = (
    (
        "real_evidence_replay",
        "REAL",
        REAL_SOURCE,
    ),
    (
        "simulated_action_proof",
        "SIMULATED",
        SIMULATED_SOURCE,
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise TypeError(
            f"{path} must contain a JSON object."
        )

    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _relative_path(path: Path) -> str:
    return path.relative_to(
        REPO_ROOT
    ).as_posix()


def _artifact_entry(
    *,
    role: str,
    evidence_class: str,
    path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "role": role,
        "evidence_class": evidence_class,
        "path": _relative_path(path),
        "canonical_sha256": canonical_sha256(
            payload
        ),
        "file_sha256": _file_sha256(path),
        "payload": payload,
    }


def build_proof_bundle() -> dict[str, Any]:
    real = _load_json(REAL_SOURCE)
    simulated = _load_json(
        SIMULATED_SOURCE
    )

    manifest = (
        build_proof_manifest_from_payloads(
            real=real,
            simulated=simulated,
        )
    )

    artifacts = [
        _artifact_entry(
            role="real_evidence_replay",
            evidence_class="REAL",
            path=REAL_SOURCE,
            payload=real,
        ),
        _artifact_entry(
            role="simulated_action_proof",
            evidence_class="SIMULATED",
            path=SIMULATED_SOURCE,
            payload=simulated,
        ),
    ]

    core: dict[str, Any] = {
        "schema_version": (
            "aeroxai.proof_bundle.v1"
        ),
        "project": "AeroXAI",
        "manifest": manifest,
        "manifest_sha256": (
            canonical_sha256(manifest)
        ),
        "artifacts": artifacts,
        "integrity_policy": {
            "canonicalization": (
                "JSON semantic normalization "
                "(integral floats -> integers), "
                "sorted keys, compact separators, "
                "ensure_ascii=true"
            ),
            "digest_algorithm": "sha256",
            "source_file_digest_scope": (
                "exact file bytes"
            ),
            "embedded_payload_digest_scope": (
                "canonical JSON value"
            ),
            "semantic_rebuild_required": True,
        },
    }

    bundle_id = canonical_sha256(
        core
    )

    return {
        **core,
        "bundle_id": (
            f"pbundle_{bundle_id[:16]}"
        ),
    }


def _expected_bundle_id(
    bundle: dict[str, Any],
) -> str:
    core = {
        key: value
        for key, value in bundle.items()
        if key != "bundle_id"
    }
    digest = canonical_sha256(core)
    return f"pbundle_{digest[:16]}"


def _artifact_map(
    bundle: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    artifacts = bundle.get(
        "artifacts"
    )
    if not isinstance(artifacts, list):
        raise TypeError(
            "Proof bundle artifacts must be a list."
        )

    result: dict[
        str,
        dict[str, Any],
    ] = {}

    for item in artifacts:
        if not isinstance(item, dict):
            raise TypeError(
                "Every proof bundle artifact "
                "must be an object."
            )

        role = item.get("role")
        if not isinstance(role, str):
            raise TypeError(
                "Every proof bundle artifact "
                "must have a string role."
            )

        if role in result:
            raise ValueError(
                f"Duplicate proof artifact role: {role}"
            )

        result[role] = item

    return result


def verify_proof_bundle(
    bundle: dict[str, Any],
    *,
    check_current_sources: bool = True,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    errors: list[str] = []

    checks["schema_version"] = (
        bundle.get("schema_version")
        == "aeroxai.proof_bundle.v1"
    )
    if not checks["schema_version"]:
        errors.append(
            "Unsupported proof bundle schema."
        )

    expected_bundle_id = (
        _expected_bundle_id(bundle)
    )
    checks["bundle_id"] = (
        bundle.get("bundle_id")
        == expected_bundle_id
    )
    if not checks["bundle_id"]:
        errors.append(
            "Bundle ID does not match bundle content."
        )

    manifest = bundle.get("manifest")
    if not isinstance(manifest, dict):
        checks["manifest_digest"] = False
        checks["semantic_rebuild"] = False
        errors.append(
            "Proof bundle manifest is missing "
            "or is not an object."
        )
        manifest = {}
    else:
        checks["manifest_digest"] = (
            bundle.get("manifest_sha256")
            == canonical_sha256(manifest)
        )
        if not checks["manifest_digest"]:
            errors.append(
                "Manifest SHA-256 does not match "
                "the embedded manifest."
            )

    try:
        artifacts = _artifact_map(bundle)
    except (TypeError, ValueError) as exc:
        artifacts = {}
        errors.append(str(exc))

    expected_roles = {
        "real_evidence_replay",
        "simulated_action_proof",
    }
    checks["artifact_roles"] = (
        set(artifacts) == expected_roles
    )
    if not checks["artifact_roles"]:
        errors.append(
            "Proof bundle must contain exactly "
            "the real replay and simulated proof "
            "artifacts."
        )

    embedded_payloads_ok = True

    for role, item in artifacts.items():
        payload = item.get("payload")

        if not isinstance(payload, dict):
            embedded_payloads_ok = False
            errors.append(
                f"{role} payload is missing or "
                "is not an object."
            )
            continue

        observed = canonical_sha256(
            payload
        )
        if (
            item.get("canonical_sha256")
            != observed
        ):
            embedded_payloads_ok = False
            errors.append(
                f"{role} canonical SHA-256 "
                "does not match its payload."
            )

    checks[
        "embedded_payload_digests"
    ] = embedded_payloads_ok

    real_item = artifacts.get(
        "real_evidence_replay"
    )
    simulated_item = artifacts.get(
        "simulated_action_proof"
    )

    semantic_rebuild_ok = False

    if (
        isinstance(real_item, dict)
        and isinstance(simulated_item, dict)
        and isinstance(
            real_item.get("payload"),
            dict,
        )
        and isinstance(
            simulated_item.get("payload"),
            dict,
        )
        and isinstance(manifest, dict)
    ):
        try:
            rebuilt = (
                build_proof_manifest_from_payloads(
                    real=real_item["payload"],
                    simulated=(
                        simulated_item["payload"]
                    ),
                )
            )
            semantic_rebuild_ok = (
                rebuilt == manifest
            )
        except (
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            errors.append(
                "Manifest semantic rebuild failed: "
                f"{exc}"
            )

    checks[
        "semantic_rebuild"
    ] = semantic_rebuild_ok
    if not semantic_rebuild_ok:
        errors.append(
            "Embedded source artifacts do not "
            "rebuild the embedded manifest exactly."
        )

    current_sources_ok = True

    if check_current_sources:
        for (
            role,
            evidence_class,
            path,
        ) in SOURCE_SPECS:
            item = artifacts.get(role)

            if not isinstance(item, dict):
                current_sources_ok = False
                continue

            if (
                item.get("evidence_class")
                != evidence_class
            ):
                current_sources_ok = False
                errors.append(
                    f"{role} evidence class changed."
                )

            if item.get("path") != _relative_path(
                path
            ):
                current_sources_ok = False
                errors.append(
                    f"{role} source path changed."
                )

            if not path.exists():
                current_sources_ok = False
                errors.append(
                    f"Current source file is missing: "
                    f"{_relative_path(path)}"
                )
                continue

            current_payload = _load_json(
                path
            )

            if (
                item.get("file_sha256")
                != _file_sha256(path)
            ):
                current_sources_ok = False
                errors.append(
                    f"{role} exact source file "
                    "SHA-256 no longer matches."
                )

            if (
                item.get("canonical_sha256")
                != canonical_sha256(
                    current_payload
                )
            ):
                current_sources_ok = False
                errors.append(
                    f"{role} canonical source "
                    "payload no longer matches."
                )

    checks[
        "current_source_files"
    ] = (
        current_sources_ok
        if check_current_sources
        else True
    )

    verified = all(
        checks.values()
    )

    return {
        "schema_version": (
            "aeroxai.proof_bundle_verification.v1"
        ),
        "bundle_id": bundle.get(
            "bundle_id"
        ),
        "verified": verified,
        "checks": checks,
        "errors": errors,
        "current_sources_checked": (
            check_current_sources
        ),
    }
