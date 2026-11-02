"""The one model Syrus uses: Groq-hosted qwen/qwen3.8-27b (owner decision D2). No fallback, routing or provider selection.

The adapter interface exists so tests can inject a scripted double; the production build has no mock mode and fails closed
(the service reports itself offline) when no key is configured.
"""
from dataclasses import dataclass
from typing import Protocol

import httpx


class LlmError(Exception):
    """A provider failure that puts the service in an offline state; `reason` is shown to the user, never the provider text."""

    def __init__(self, reason: str, retry_after: float | None = None):
        super().__init__(reason)
        self.reason, self.retry_after = reason, retry_after


@dataclass(frozen=True)
class Completion:
    text: str
    tokens_in: int
    tokens_out: int


class Adapter(Protocol):
    async def complete(self, messages: list[dict], schema: dict, max_tokens: int, timeout: float) -> Completion: ...


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
            raise LlmError("timeout") from e
        except httpx.HTTPError as e:
            raise LlmError("provider_error") from e
        if r.status_code == 429:
            raise LlmError("rate_limited", float(r.headers.get("retry-after", 0)) or None)
        if r.status_code in (401, 403):
            raise LlmError("provider_auth")
        if r.status_code != 200:
            raise LlmError("provider_error")
        try:
            data = r.json()
            return Completion(data["choices"][0]["message"]["content"], data["usage"]["prompt_tokens"], data["usage"]["completion_tokens"])
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LlmError("provider_error") from e
