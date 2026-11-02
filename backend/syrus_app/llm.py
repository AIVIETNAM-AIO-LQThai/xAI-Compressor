"""The one model Syrus uses: Groq-hosted qwen/qwen3.8-27b (owner decision D2). No fallback, routing or provider selection.

The adapter interface exists so tests can inject a scripted double; the production build has no mock mode and fails closed
(the service reports itself offline) when no key is configured.

Provider diagnostics are sanitised at the source: only rate-limit headers, token counts, the error type and code, and the model name
the provider reports, each checked against a strict pattern. Prompts, answers, the provider's error text and its echoed output
(`failed_generation`) are never read into them, so they are safe to log and to return to a developer.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

import httpx

log = logging.getLogger("syrus.provider")
HEADERS = {"retry-after": "retry_after", "x-ratelimit-limit-tokens": "limit_tokens", "x-ratelimit-remaining-tokens": "remaining_tokens",
           "x-ratelimit-reset-tokens": "reset_tokens", "x-ratelimit-limit-requests": "limit_requests", "x-ratelimit-remaining-requests": "remaining_requests",
           "x-ratelimit-reset-requests": "reset_requests", "x-request-id": "request_id"}
SAFE = re.compile(r"[0-9A-Za-z._:/-]{1,64}")
RATE_MESSAGE = re.compile(r"Limit (\d+), Used (\d+), Requested (\d+)")


class LlmError(Exception):
    """A provider failure that puts the service in an offline state; `reason` is shown to the user, never the provider text."""

    def __init__(self, reason: str, retry_after: float | None = None, diag: dict | None = None):
        super().__init__(reason)
        self.reason, self.retry_after, self.diag = reason, retry_after, diag or {}


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
    out = {"http": r.status_code}
    out.update({key: v for header, key in HEADERS.items() if (v := r.headers.get(header)) and SAFE.fullmatch(v)})
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        out.update({f"error_{k}": err[k] for k in ("type", "code") if isinstance(err.get(k), str) and SAFE.fullmatch(err[k])})
        if isinstance(err.get("message"), str) and (m := RATE_MESSAGE.search(err["message"])):
            out.update(zip(("limit", "used", "requested"), map(int, m.groups()), strict=True))
    return out


class GroqAdapter:
    def __init__(self, key: str, cfg: dict, client: httpx.AsyncClient | None = None):
        self.key, self.model, self.url = key, cfg["model"], cfg["endpoint"]
        self.client = client or httpx.AsyncClient()

    async def complete(self, messages, schema, max_tokens, timeout):
        body = {"model": self.model, "messages": messages, "temperature": 0, "max_completion_tokens": max_tokens, "reasoning_effort": "none",
                "response_format": {"type": "json_schema", "json_schema": {"name": "syrus_turn", "strict": True, "schema": schema}}}
        try:
            r = await self.client.post(self.url, json=body, headers={"Authorization": f"Bearer {self.key}"}, timeout=timeout)
        except httpx.TimeoutException as e:
            raise LlmError("timeout", diag={"error": "timeout"}) from e
        except httpx.HTTPError as e:
            raise LlmError("provider_error", diag={"error": type(e).__name__}) from e
        if r.status_code != 200:
            try:
                err = r.json()
            except ValueError:
                err = None
            diag = diagnostics(r, err)
            log.warning(json.dumps(diag))
            reason = "rate_limited" if r.status_code == 429 else "provider_auth" if r.status_code in (401, 403) else "provider_error"
            retry = float(diag["retry_after"]) if reason == "rate_limited" and re.fullmatch(r"\d+(?:\.\d+)?", diag.get("retry_after", "")) else None
            raise LlmError(reason, retry or None, diag)
        try:
            data = r.json()
            diag = diagnostics(r)
            if isinstance(data.get("model"), str) and SAFE.fullmatch(data["model"]):
                diag["model"] = data["model"]
            return Completion(data["choices"][0]["message"]["content"], data["usage"]["prompt_tokens"], data["usage"]["completion_tokens"], diag)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            raise LlmError("provider_error", diag=diagnostics(r)) from e
