"""One origin for the whole product: the Syrus API, the evidence and simulation API, and the built frontend.

    uvicorn backend.web:app --host 127.0.0.1 --port 8000

`/api/*` is the Syrus service (one budget, one session store, one governor); `/evidence/*`, `/twin/*`, `/optimization/*` and `/health` are the
existing dashboard routes, reused as they are; every other path serves `frontend/dist` when it has been built. The browser and a phone on the
same network use one URL and no CORS. With AEROXAI_PUBLIC=1 the simulation-compute routes (`/twin`, `/optimization`) are not served at all.
"""
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.app.routes.detection import router as detection
from backend.app.routes.evidence import router as evidence
from backend.app.routes.optimization import router as optimization
from backend.app.routes.twin import router as twin
from backend.syrus_app.app import create_app as create_syrus
from backend.syrus_app.service import ApiError
from backend.syrus_app.sessions import SessionError
from backend.syrus_app.settings import Settings

DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

def create_web(settings: Settings | None = None, adapter=None, store=None, dist: Path = DIST, public: bool | None = None) -> FastAPI:
    public = os.environ.get("AEROXAI_PUBLIC") == "1" if public is None else public
    syrus = create_syrus(settings, adapter, store)
    app = FastAPI(title="AeroXAI", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.svc = syrus.state.svc
    for exc in (ApiError, SessionError):
        app.add_exception_handler(exc, syrus.exception_handlers[exc])
    for router in (detection, evidence) if public else (detection, evidence, twin, optimization):
        app.include_router(router)
    app.router.routes.extend(r for r in syrus.router.routes if getattr(r, "path", "").startswith("/api/"))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "project": "AeroXAI", "mode": "advisory"}

    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app


app = create_web()
