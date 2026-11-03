"""Who may use what, decided on the server before any provider call, compute or data read.

Three modes, from AEROXAI_MODE (AEROXAI_PUBLIC=1 is read as `public`):

- local    everything, for the owner on 127.0.0.1;
- preview  invited guests only. Every request needs a signed cookie obtained with the invite code (AEROXAI_PREVIEW_CODE_SHA256 holds
           the SHA-256 of the code; the server refuses to start without it). Live Syrus only with AEROXAI_SYRUS_GUESTS=live, under a lower
           daily cap (AEROXAI_PREVIEW_DAILY_USD, default 2) and a per-guest turn quota (AEROXAI_GUEST_TURNS per UTC day, default 10);
           otherwise Syrus has no key and no adapter. Per-guest request limits on everything;
- public   read-only: no Syrus session or turn route, no key, no simulation compute.

Unsafe combinations (an unknown mode, diagnostics outside local, preview without a code) stop the server at start.
"""
import hashlib
import hmac
import os
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import parse_qs

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

MODES = ("local", "preview", "public")
API = ("/api/", "/evidence", "/explorer", "/lab", "/twin", "/optimization", "/config")
COOKIE = "aeroxai_preview"
COOKIE_HOURS = 12


@dataclass(frozen=True)
class Access:
    mode: str
    syrus_guests: bool  # live Syrus for guests (preview only)
    code_sha256: str | None
    secret: bytes
    daily_usd: float
    guest_turns: int
    requests_per_minute: int = 240
    lab_runs_per_10_minutes: int = 30

    @property
    def features(self) -> dict:
        local = self.mode == "local"
        return {"syrus_chat": local or (self.mode == "preview" and self.syrus_guests), "lab": "live" if self.mode != "public" else "off",
                "twin": local, "optimization": local}


def resolve(env=None) -> Access:
    env = os.environ if env is None else env
    mode = env.get("AEROXAI_MODE") or ("public" if env.get("AEROXAI_PUBLIC") == "1" else "local")
    if mode not in MODES:
        raise RuntimeError(f"AEROXAI_MODE must be one of {', '.join(MODES)}")
    code = (env.get("AEROXAI_PREVIEW_CODE_SHA256") or "").strip().lower() or None
    if mode != "local" and env.get("SYRUS_DIAGNOSTICS") == "1":
        raise RuntimeError("SYRUS_DIAGNOSTICS is for local development only")
    if mode == "preview" and (not code or len(code) != 64 or any(c not in "0123456789abcdef" for c in code)):
        raise RuntimeError("preview mode needs AEROXAI_PREVIEW_CODE_SHA256, the hex SHA-256 of the invite code")
    secret = (env.get("AEROXAI_PREVIEW_SECRET") or secrets.token_hex(32)).encode()
    return Access(mode=mode, syrus_guests=mode == "preview" and env.get("AEROXAI_SYRUS_GUESTS") == "live", code_sha256=code, secret=secret,
                  daily_usd=float(env.get("AEROXAI_PREVIEW_DAILY_USD") or 2.0), guest_turns=int(env.get("AEROXAI_GUEST_TURNS") or 10))


class Gate:
    """The preview's invite check, its signed visitor cookie and its per-visitor limits. In memory: one process, as the launcher enforces."""

    def __init__(self, access: Access, now=time.time):
        self.a, self.now = access, now
        self.attempts: dict[str, deque] = defaultdict(deque)
        self.requests: dict[str, deque] = defaultdict(deque)
        self.lab: dict[str, deque] = defaultdict(deque)
        self.turns: dict[tuple[str, str], int] = defaultdict(int)

    def _sign(self, body: str) -> str:
        return hmac.new(self.a.secret, body.encode(), hashlib.sha256).hexdigest()

    def cookie(self) -> str:
        body = f"{secrets.token_hex(8)}.{int(self.now()) + COOKIE_HOURS * 3600}"
        return f"{body}.{self._sign(body)}"

    def visitor(self, value: str | None) -> str | None:
        """The visitor id in a valid, unexpired cookie, else None."""
        try:
            vid, expires, sig = (value or "").split(".")
            ok = hmac.compare_digest(sig, self._sign(f"{vid}.{expires}")) and int(expires) > self.now()
        except ValueError:
            return None
        return vid if ok else None

    def _window(self, log: deque, limit: int, seconds: float) -> bool:
        t = self.now()
        while log and log[0] <= t - seconds:
            log.popleft()
        if len(log) >= limit:
            return False
        log.append(t)
        return True

    def code_ok(self, ip: str, code: str) -> bool | None:
        """True or False for a guess; None when this address has guessed too often (10 per 10 minutes)."""
        if not self._window(self.attempts[ip], 10, 600):
            return None
        return hmac.compare_digest(hashlib.sha256(code.strip().encode()).hexdigest(), self.a.code_sha256 or "")

    def allow(self, vid: str, path: str, method: str) -> str | None:
        """An error code when this visitor is over a limit, else None."""
        if not self._window(self.requests[vid], self.a.requests_per_minute, 60):
            return "rate_limited"
        if path.startswith("/lab") and method == "POST" and not self._window(self.lab[vid], self.a.lab_runs_per_10_minutes, 600):
            return "lab_rate_limited"
        if path == "/api/syrus/turn" and method == "POST":
            key = (vid, datetime.now(UTC).date().isoformat())
            if self.turns[key] >= self.a.guest_turns:
                return "guest_quota"
            self.turns[key] += 1
        return None


GATE_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AeroXAI preview</title><style>
:root{color-scheme:dark}body{margin:0;min-height:100vh;display:grid;place-items:center;background:#080a0d;color:#f2f3f5;font:16px/1.5 system-ui,sans-serif}
form{width:min(360px,90vw);display:grid;gap:12px;padding:28px;border:1px solid rgba(255,255,255,.11);border-radius:14px;background:#11161b}
h1{margin:0;font-size:22px}p{margin:0;color:#aab2bb;font-size:14px}input{min-height:44px;padding:0 12px;border-radius:9px;border:1px solid rgba(255,255,255,.18);background:#0e1115;color:#f2f3f5;font:inherit}
button{min-height:44px;border:0;border-radius:9px;background:#ff7126;color:#120a05;font-weight:600;font:inherit;cursor:pointer}.e{color:#f0647a}
</style></head><body><form method="post" action="/preview/enter"><h1>AeroXAI preview</h1><p>Invited guests only. Enter the invite code you were given.</p>
<label for="c" style="font-size:14px;color:#aab2bb">Invite code</label><input id="c" name="code" type="password" autocomplete="off" required autofocus>
<button type="submit">Open the preview</button>{error}</form></body></html>"""


def install(app, access: Access) -> Gate | None:
    """Adds the preview gate and the per-visitor limits; nothing is added in local or public mode."""
    if access.mode != "preview":
        return None
    gate = Gate(access)

    def page(error: str = "", status: int = 200) -> HTMLResponse:
        return HTMLResponse(GATE_PAGE.replace("{error}", f'<p class="e" role="alert">{error}</p>' if error else ""), status_code=status,
                            headers={"Cache-Control": "no-store"})

    @app.middleware("http")
    async def preview_gate(request: Request, call_next) -> Response:
        path, ip = request.url.path, request.client.host if request.client else "unknown"
        if path == "/health":
            return await call_next(request)
        if path == "/preview/enter" and request.method == "POST":
            form = parse_qs((await request.body())[:2048].decode("utf-8", "replace"))
            ok = gate.code_ok(ip, (form.get("code") or [""])[0])
            if ok is None:
                return page("Too many attempts. Wait ten minutes.", 429)
            if not ok:
                return page("That code is not valid.", 401)
            r = RedirectResponse("/", status_code=303)
            r.set_cookie(COOKIE, gate.cookie(), max_age=COOKIE_HOURS * 3600, httponly=True, samesite="strict", secure=request.url.scheme == "https", path="/")
            return r
        vid = gate.visitor(request.cookies.get(COOKIE))
        if vid is None:
            if path.startswith(API):
                return JSONResponse({"error": {"code": "preview_locked"}}, status_code=401)
            return page()
        if err := gate.allow(vid, path, request.method):
            return JSONResponse({"error": {"code": err}}, status_code=429)
        return await call_next(request)

    return gate
