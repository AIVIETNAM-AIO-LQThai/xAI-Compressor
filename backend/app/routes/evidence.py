from fastapi import APIRouter

from backend.app.evidence import (
    build_evidence_summary,
)
from backend.app.models import (
    EvidenceSummaryResponse,
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
