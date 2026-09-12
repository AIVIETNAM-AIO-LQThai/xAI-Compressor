from fastapi import APIRouter

from backend.app.evidence import (
    build_detection_evidence,
)
from backend.app.models import (
    DetectionEvidenceResponse,
)

router = APIRouter(
    prefix="/evidence",
    tags=["evidence"],
)


@router.get(
    "/detection",
    response_model=DetectionEvidenceResponse,
)
def get_detection_evidence() -> dict:
    return build_detection_evidence()
