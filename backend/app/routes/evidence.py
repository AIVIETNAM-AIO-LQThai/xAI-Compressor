from typing import Any

from fastapi import APIRouter

from backend.app.evidence import (
    build_evidence_summary,
)
from backend.app.models import (
    EvidenceSummaryResponse,
)
from backend.app.proof_bundle import (
    build_proof_bundle,
    verify_proof_bundle,
)
from backend.app.proof_manifest import (
    build_proof_manifest,
)

router = APIRouter(
    prefix="/evidence",
    tags=["evidence"],
)


@router.get(
    "/summary",
    response_model=EvidenceSummaryResponse,
)
def get_evidence_summary() -> dict:
    return build_evidence_summary()


@router.get(
    "/proof-manifest",
)
def get_proof_manifest() -> dict:
    return build_proof_manifest()


@router.get(
    "/proof-bundle",
)
def get_proof_bundle() -> dict:
    return build_proof_bundle()


@router.post(
    "/proof-bundle/verify",
)
def verify_submitted_proof_bundle(
    bundle: dict[str, Any],
) -> dict:
    return verify_proof_bundle(
        bundle,
        check_current_sources=True,
    )
