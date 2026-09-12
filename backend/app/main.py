from fastapi import FastAPI
from fastapi.middleware.cors import (
    CORSMiddleware,
)

from backend.app.routes.detection import (
    router as detection_router,
)
from backend.app.routes.evidence import (
    router as evidence_router,
)
from backend.app.routes.optimization import (
    router as optimization_router,
)
from backend.app.routes.twin import (
    router as twin_router,
)

app = FastAPI(
    title="AeroXAI API",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    detection_router
)

app.include_router(
    evidence_router
)

app.include_router(
    twin_router
)

app.include_router(
    optimization_router
)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "project": "AeroXAI",
        "mode": "advisory",
    }
