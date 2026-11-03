"""One origin for the whole product: the Syrus API, the evidence and simulation API, and the built frontend.

    uvicorn backend.web:app --host 127.0.0.1 --port 8000

`/api/*` is the Syrus service (one budget, one session store, one governor); `/evidence/*`, `/explorer/*`, `/twin/*`, `/optimization/*`, `/lab/*` and `/health` are
the dashboard routes, reused as they are; `/config` tells the interface what this deployment serves; every other path serves `frontend/dist` when it
has been built. The browser and a phone on the same network use one URL and no CORS.

What is served depends on AEROXAI_MODE (see backend/access.py): `local` serves everything; `preview` puts every path behind the invite
gate, keeps the Scenario Lab, drops `/twin` and `/optimization`, and gives Syrus no key unless guests are explicitly allowed; `public` serves
read-only routes only, without any Syrus session or turn route and without a key.
"""
import dataclasses
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.access import Access, install, resolve
from backend.app.routes.detection import router as detection
from backend.app.routes.evidence import router as evidence
from backend.app.routes.explorer import router as explorer
from backend.app.routes.lab import router as lab
from backend.app.routes.optimization import router as optimization
from backend.app.routes.twin import router as twin
from backend.syrus_app.app import create_app as create_syrus
from backend.syrus_app.service import ApiError
from backend.syrus_app.sessions import SessionError
from backend.syrus_app.settings import Settings, load_settings

DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
READ_ONLY_SYRUS = ("/api/health", "/api/cases")


def _syrus_settings(settings: Settings, access: Access) -> Settings:
    """Visitors outside local mode never reach the provider unless the owner allowed guests; then under the preview's lower daily cap."""
    if access.mode == "local":
        return settings
    if not access.syrus_guests:
        return dataclasses.replace(settings, key=None, guests=False)
    caps = settings.cfg["caps_usd"]
    return dataclasses.replace(settings, cfg={**settings.cfg, "caps_usd": {**caps, "daily": min(caps["daily"], access.daily_usd)}})


def create_web(settings: Settings | None = None, adapter=None, store=None, dist: Path = DIST, public: bool | None = None, access: Access | None = None) -> FastAPI:
    if access is None:
        access = resolve() if public is None else resolve({**os.environ, "AEROXAI_MODE": "public" if public else "local"})
    settings = _syrus_settings(settings or load_settings(), access)
    if access.mode != "local" and not access.syrus_guests:
        adapter = None  # no adapter object exists, so no code path can call the provider
    syrus = create_syrus(settings, adapter, store)
    app = FastAPI(title="AeroXAI", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.svc, app.state.access = syrus.state.svc, access
    app.state.gate = install(app, access)
    for exc in (ApiError, SessionError):
        app.add_exception_handler(exc, syrus.exception_handlers[exc])
    routers = {"local": (detection, evidence, explorer, twin, optimization, lab), "preview": (detection, evidence, explorer, lab), "public": (detection, evidence, explorer)}
    for router in routers[access.mode]:
        app.include_router(router)
    keep = (lambda p: p.startswith(READ_ONLY_SYRUS)) if access.mode == "public" else (lambda p: p.startswith("/api/"))
    app.router.routes.extend(r for r in syrus.router.routes if keep(getattr(r, "path", "")))

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "project": "AeroXAI", "mode": "advisory"}

    @app.get("/config")
    def config() -> dict:
        return {"mode": access.mode, "features": access.features}

    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    return app


app = create_web()
