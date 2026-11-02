"""Slim FastAPI app for Syrus: case files, guest sessions and the retrospective turn. No equipment, simulation or optimisation endpoint.

Run from the repository root:  uvicorn backend.syrus_app.app:app
"""
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.syrus_app.budget import Budget
from backend.syrus_app.context import CaseStore
from backend.syrus_app.llm import Adapter, GroqAdapter
from backend.syrus_app.policy import DEPTHS
from backend.syrus_app.service import ApiError, Service
from backend.syrus_app.sessions import MODES, SessionError, Sessions
from backend.syrus_app.settings import Settings, load_settings


class SessionRequest(BaseModel):
    case_id: str
    mode: str


class TurnRequest(BaseModel):
    message: str
    depth: str = "auto"
    case_id: str | None = None
    mode: str | None = None


def create_app(settings: Settings | None = None, adapter: Adapter | None = None, store: CaseStore | None = None) -> FastAPI:
    settings = settings or load_settings()
    store = store or CaseStore()
    if adapter is None and settings.key:
        adapter = GroqAdapter(settings.key, settings.cfg)
    cfg = settings.cfg
    budget = Budget(cfg["caps_usd"], cfg["price_per_million"], settings.budget_file)
    svc = Service(settings, store, budget, Sessions(settings.secret, settings.limits), adapter)
    app = FastAPI(title="Syrus", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.svc = svc

    @app.exception_handler(ApiError)
    async def api_error(_: Request, e: ApiError):
        return JSONResponse({"error": {"code": e.code}}, status_code=e.status)

    @app.exception_handler(SessionError)
    async def session_error(_: Request, e: SessionError):
        return JSONResponse({"error": {"code": e.code}}, status_code=e.status)

    def thread(authorization: str | None):
        if not authorization or not authorization.startswith("Bearer "):
            raise SessionError("token_missing")
        return svc.sessions.get(authorization.removeprefix("Bearer "))

    @app.get("/api/health")
    async def health():
        reason = svc.offline_reason()
        return {"status": "offline" if reason else "ok", "reason": reason, "model": settings.model, "mock": False, "cases": len(store.manifest["cases"]),
                "manifest_sha256": store.manifest["sha256"]}

    @app.get("/api/cases")
    async def cases():
        return store.manifest

    @app.get("/api/cases/{case_id}")
    async def case(case_id: str, mode: str = "asof"):
        found = store.get(case_id, mode)
        if not found:
            raise ApiError(404, "case_not_found")
        return found

    @app.post("/api/session")
    async def session(req: SessionRequest, request: Request):
        if req.mode not in MODES or not store.get(req.case_id, req.mode):
            raise ApiError(404, "case_not_found")
        token, t = svc.sessions.mint(req.case_id, req.mode, request.client.host if request.client else "unknown")
        return {"token": token, "case_id": t.case_id, "mode": t.mode, "expires_in": settings.limits["session_minutes"] * 60,
                "case_sha256": store.get(req.case_id, req.mode)["sha256"], "chat_available": t.mode == "retro" and svc.offline_reason() is None,
                "offline_reason": svc.offline_reason()}

    @app.post("/api/syrus/turn")
    async def turn(req: TurnRequest, request: Request, authorization: str | None = Header(default=None)):
        t = thread(authorization)
        if (req.case_id and req.case_id != t.case_id) or (req.mode and req.mode != t.mode):
            raise ApiError(409, "case_mismatch")
        if req.depth != "auto" and req.depth not in DEPTHS:
            raise ApiError(400, "depth_invalid")
        status, body = await svc.turn(t, req.message, req.depth, request.client.host if request.client else "unknown")
        return JSONResponse(body, status_code=status)

    return app


app = create_app()
