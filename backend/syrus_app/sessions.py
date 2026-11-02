"""Guest sessions: signed tokens bound to one case and mode, server-held threads, and a per-IP minting limit."""
import base64
import hashlib
import hmac
import json
import secrets
import time
from collections import deque
from dataclasses import dataclass, field

MODES = ("asof", "retro")


class SessionError(Exception):
    def __init__(self, code: str, status: int = 401):
        super().__init__(code)
        self.code, self.status = code, status


@dataclass
class Thread:
    sid: str
    case_id: str
    mode: str
    exp: float
    turns: int = 0
    busy: bool = False
    exchanges: deque = field(default_factory=deque)         # (user text, answer text) of the latest turns
    stated: dict = field(default_factory=dict)              # finding id -> turn number it was last stated
    quals: set = field(default_factory=set)                 # qualification ids already shown in this conversation
    asked: set = field(default_factory=set)                 # follow-up test ids already asked
    last_question: int = 0                                  # turn number of the last follow-up question
    operator: list = field(default_factory=list)            # unverified statements by the user, never evidence
    focus: list = field(default_factory=list)               # topics of the last answered question, for a bare follow-up


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class Sessions:
    def __init__(self, secret: bytes, limits: dict, now=time.time):
        self.secret, self.limits, self.now = secret, limits, now
        self.threads, self.minted = {}, {}

    def _sign(self, payload: bytes) -> str:
        return _b64(hmac.new(self.secret, payload, hashlib.sha256).digest())

    def mint(self, case_id: str, mode: str, ip: str) -> tuple[str, Thread]:
        now = self.now()
        recent = self.minted.setdefault(ip, deque())
        while recent and now - recent[0] > 3600:
            recent.popleft()
        if len(recent) >= self.limits["sessions_per_ip_per_hour"]:
            raise SessionError("session_rate_limit", 429)
        recent.append(now)
        self.threads = {s: t for s, t in self.threads.items() if t.exp > now}
        t = Thread(secrets.token_urlsafe(12), case_id, mode, now + 60 * self.limits["session_minutes"], exchanges=deque(maxlen=self.limits["history_exchanges"]))
        self.threads[t.sid] = t
        payload = json.dumps({"sid": t.sid, "case": case_id, "mode": mode, "access": "public", "exp": t.exp}, separators=(",", ":")).encode()
        return f"{_b64(payload)}.{self._sign(payload)}", t

    def get(self, token: str) -> Thread:
        try:
            body, sig = token.split(".")
            payload = _unb64(body)
            claims = json.loads(payload)
        except ValueError as e:
            raise SessionError("token_invalid") from e
        if not hmac.compare_digest(sig, self._sign(payload)):
            raise SessionError("token_invalid")
        t = self.threads.get(claims["sid"])
        if claims["exp"] <= self.now() or t is None:
            raise SessionError("session_expired")
        if (t.case_id, t.mode) != (claims["case"], claims["mode"]) or claims["access"] != "public":
            raise SessionError("token_invalid")
        return t
