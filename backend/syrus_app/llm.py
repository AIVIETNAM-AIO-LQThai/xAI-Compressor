"""The one model Syrus uses: Groq-hosted qwen/qwen3.8-27b (owner decision D2). No fallback, routing or provider selection.

The adapter interface exists so tests can inject a scripted double; the production build has no mock mode and fails closed
(the service reports itself offline) when no key is configured.

Provider diagnostics are sanitised at the source: only rate-limit headers, token counts, the error type and code, and the model name
the provider reports, each checked against a strict pattern. For a transport failure (no HTTP response) only exception class
names, the stage the class implies, elapsed time, a reviewed OS error name and the names of proxy settings are kept. Prompts,
answers, URLs, exception text, the provider's error text and its echoed output (`failed_generation`) are never read into them, so
they are safe to log and to return to a developer.
"""
import asyncio
import errno
import json
import logging
import os
import platform
import re
import ssl
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

import httpx

log = logging.getLogger("syrus.provider")
HEADERS = {"retry-after": "retry_after", "x-ratelimit-limit-tokens": "limit_tokens", "x-ratelimit-remaining-tokens": "remaining_tokens",
           "x-ratelimit-reset-tokens": "reset_tokens", "x-ratelimit-limit-requests": "limit_requests", "x-ratelimit-remaining-requests": "remaining_requests",
           "x-ratelimit-reset-requests": "reset_requests", "x-request-id": "request_id"}
SAFE = re.compile(r"[0-9A-Za-z._:/-]{1,64}")
RATE_MESSAGE = re.compile(r"Limit (\d+), Used (\d+), Requested (\d+)")
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,40}")
REASON = re.compile(r"[A-Z0-9_]{1,40}")
PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
# What a transport error class says about where the exchange broke (httpx's own taxonomy); nothing here is a conclusion about the cause.
STAGE = {"ConnectError": "connect", "ConnectTimeout": "connect", "WriteError": "send", "WriteTimeout": "send", "ReadError": "receive",
         "ReadTimeout": "receive", "RemoteProtocolError": "receive", "LocalProtocolError": "send", "PoolTimeout": "pool", "ProxyError": "proxy",
         "CloseError": "close", "DecodingError": "decode", "UnsupportedProtocol": "request"}
# OS error codes reviewed as safe to name; anything else is reported only as "other".
WINSOCK = {10053: "WSAECONNABORTED", 10054: "WSAECONNRESET", 10057: "WSAENOTCONN", 10060: "WSAETIMEDOUT", 10061: "WSAECONNREFUSED", 10065: "WSAEHOSTUNREACH"}
POSIX = {errno.ECONNRESET, errno.ECONNABORTED, errno.ECONNREFUSED, errno.EPIPE, errno.ETIMEDOUT, errno.EHOSTUNREACH, errno.ENETUNREACH, errno.ENOTCONN}


class LlmError(Exception):
    """A provider failure that puts the service in an offline state; `reason` is shown to the user, never the provider text.

    `indeterminate` says whether the provider may have processed (and billed) the request: a refusal it answered (rate limit,
    authentication, a rejected request) is not, a dropped connection, a timeout, a server error or an unreadable 200 may be."""

    def __init__(self, reason: str, retry_after: float | None = None, diag: dict | None = None, indeterminate: bool | None = None):
        super().__init__(reason)
        self.reason, self.retry_after, self.diag = reason, retry_after, diag or {}
        self.indeterminate = reason in {"timeout", "provider_error"} if indeterminate is None else indeterminate


@dataclass(frozen=True)
class Completion:
    text: str
    tokens_in: int
    tokens_out: int
    diag: dict = field(default_factory=dict)


class Adapter(Protocol):
    async def complete(self, messages: list[dict], schema: dict, max_tokens: int, timeout: float) -> Completion: ...


def diagnostics(r: httpx.Response, body: dict | None = None) -> dict:
    """What a developer may see of one provider response: status, rate-limit headers, and for an error its type, code and token counts."""
    out = {"http": r.status_code, "response_received": True}
    out.update({key: v for header, key in HEADERS.items() if (v := r.headers.get(header)) and SAFE.fullmatch(v)})
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        out.update({f"error_{k}": err[k] for k in ("type", "code") if isinstance(err.get(k), str) and SAFE.fullmatch(err[k])})
        if isinstance(err.get("message"), str) and (m := RATE_MESSAGE.search(err["message"])):
            out.update(zip(("limit", "used", "requested"), map(int, m.groups()), strict=True))
    return out


def chain(exc: BaseException, limit: int = 6) -> list[str]:
    """Class names along the cause chain, outermost first. Names only; never `str(exc)`."""
    out, seen, e = [], set(), exc
    while e is not None and id(e) not in seen and len(out) < limit:
        seen.add(id(e))
        out.append(type(e).__name__ if NAME.fullmatch(type(e).__name__) else "other")
        e = e.__cause__ or e.__context__
    return out


def os_error(exc: BaseException) -> dict:
    """Reviewed OS and TLS codes found anywhere in the cause chain."""
    found, seen, e = {}, set(), exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, OSError) and not isinstance(e, ssl.SSLError):
            code = e.errno if e.errno in POSIX or e.errno in WINSOCK else getattr(e, "winerror", None)
            if code in WINSOCK:
                found["os_error"] = WINSOCK[code]
            elif code in POSIX:
                found["os_error"] = errno.errorcode[code]
        if isinstance(e, ssl.SSLError):
            for key, attr in (("ssl_reason", "reason"), ("ssl_library", "library")):
                if isinstance(getattr(e, attr, None), str) and REASON.fullmatch(getattr(e, attr)):
                    found[key] = getattr(e, attr)
        e = e.__cause__ or e.__context__
    return found


def environment(client: httpx.AsyncClient) -> dict:
    """Local settings that can redirect or interfere with a connection. Proxy variable and scheme names only, never their values."""
    try:
        loop = type(asyncio.get_running_loop()).__name__
    except RuntimeError:
        loop = None
    return {"httpx": httpx.__version__, "python": platform.python_version(), "platform": platform.system(), "event_loop": loop,
            "trust_env": getattr(client, "trust_env", None), "proxy_env": [v for v in PROXY_VARS if os.environ.get(v) or os.environ.get(v.lower())],
            "system_proxy_schemes": sorted(k for k in urllib.request.getproxies() if NAME.fullmatch(k))}


def transport_diagnostics(exc: BaseException, started: float, client: httpx.AsyncClient) -> dict:
    """A failure with no HTTP response: the exception class and chain, the stage that class implies, elapsed time, and the local setup."""
    name = type(exc).__name__
    return {"error": name if NAME.fullmatch(name) else "other", "stage": STAGE.get(name, "other"), "chain": chain(exc), "response_received": False,
            "message_empty": not str(exc), "elapsed_ms": round((time.monotonic() - started) * 1000), **os_error(exc), "environment": environment(client)}


class GroqAdapter:
    def __init__(self, key: str, cfg: dict, client: httpx.AsyncClient | None = None):
        self.key, self.model, self.url = key, cfg["model"], cfg["endpoint"]
        self.client = client or httpx.AsyncClient()

    async def complete(self, messages, schema, max_tokens, timeout):
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_completion_tokens": max_tokens, "reasoning_effort": "none",
                "response_format": {"type": "json_schema", "json_schema": {"name": "syrus_turn", "strict": True, "schema": schema}}}
        started = time.monotonic()
        try:
            r = await self.client.post(self.url, json=body, headers={"Authorization": f"Bearer {self.key}"}, timeout=timeout)
        except httpx.HTTPError as e:
            diag = transport_diagnostics(e, started, self.client)
            log.warning(json.dumps(diag))
            raise LlmError("timeout" if isinstance(e, httpx.TimeoutException) else "provider_error", diag=diag) from e
        elapsed = round((time.monotonic() - started) * 1000)
        if r.status_code != 200:
            try:
                err = r.json()
            except ValueError:
                err = None
            diag = {**diagnostics(r, err), "elapsed_ms": elapsed}
            log.warning(json.dumps(diag))
            reason = "rate_limited" if r.status_code == 429 else "provider_auth" if r.status_code in (401, 403) else "provider_error"
            retry = float(diag["retry_after"]) if reason == "rate_limited" and re.fullmatch(r"\d+(?:\.\d+)?", diag.get("retry_after", "")) else None
            answered = reason != "provider_error" or 400 <= r.status_code < 500  # a refusal the provider stated is not a processed request
            raise LlmError(reason, retry or None, diag, indeterminate=not answered)
        diag = {**diagnostics(r), "elapsed_ms": elapsed}
        try:
            data = r.json()
        except ValueError as e:
            raise LlmError("provider_error", diag={**diag, "error": "body_not_json"}) from e
        try:
            if isinstance(data.get("model"), str) and SAFE.fullmatch(data["model"]):
                diag["model"] = data["model"]
            return Completion(data["choices"][0]["message"]["content"], data["usage"]["prompt_tokens"], data["usage"]["completion_tokens"], diag)
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise LlmError("provider_error", diag={**diag, "error": "body_missing_fields"}) from e
