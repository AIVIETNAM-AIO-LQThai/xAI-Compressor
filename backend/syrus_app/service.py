"""One Syrus turn: limits, offline and budget checks, one model call, deterministic validation, at most one retry, assembly."""
import hashlib
import json
import logging
import time
import uuid

from backend.syrus_app.answer import assemble, pick_lang, remember
from backend.syrus_app.budget import Budget, BudgetExceeded
from backend.syrus_app.context import CaseStore
from backend.syrus_app.llm import Adapter, LlmError
from backend.syrus_app.policy import TEXT
from backend.syrus_app.prompt import build
from backend.syrus_app.selection import SelectionError
from backend.syrus_app.sessions import Sessions, Thread
from backend.syrus_app.settings import Settings
from backend.syrus_app.tokens import estimate_request
from backend.syrus_app.validator import Fail, check, conveyed, missing, parse

log = logging.getLogger("syrus.turns")


class ApiError(Exception):
    def __init__(self, status: int, code: str):
        super().__init__(code)
        self.status, self.code = status, code


class Service:
    def __init__(self, settings: Settings, store: CaseStore, budget: Budget, sessions: Sessions, adapter: Adapter | None):
        self.settings, self.store, self.budget, self.sessions, self.adapter = settings, store, budget, sessions, adapter

    def offline_reason(self) -> str | None:
        if reason := self.settings.offline_reason():
            return reason
        return None if self.adapter else "adapter_missing"

    def ip_hash(self, ip: str) -> str:
        return hashlib.sha256(self.settings.ip_salt + ip.encode()).hexdigest()[:12]

    def offline(self, reason: str, retry_after: float | None = None, diag: dict | None = None) -> tuple[int, dict]:
        body = {"status": "offline", "reason": reason, "retry_after": retry_after}
        if self.settings.diagnostics and diag:
            body["diagnostics"] = diag
        return 503, body

    async def turn(self, thread: Thread, message: str, depth_pref: str, ip: str) -> tuple[int, dict]:
        lim = self.settings.limits
        if thread.mode != "retro":
            raise ApiError(403, "asof_chat_disabled")
        if not message.strip():
            raise ApiError(400, "empty_message")
        if len(message) > lim["message_chars"]:
            raise ApiError(400, "message_too_long")
        if thread.turns >= lim["turns_per_session"]:
            raise ApiError(429, "turn_limit")
        if thread.busy:
            raise ApiError(409, "turn_in_progress")
        if reason := self.offline_reason():
            self.record(thread, "offline:" + reason, 0, 0, 0, None, ip)
            return self.offline(reason)
        thread.busy = True
        try:
            return await self.run(thread, message, depth_pref, ip)
        finally:
            thread.busy = False

    async def run(self, thread: Thread, message: str, depth_pref: str, ip: str) -> tuple[int, dict]:
        lim, pack, cap = self.settings.limits, self.store.pack(thread.case_id), self.settings.input_cap
        start = time.monotonic()
        deadline = start + lim["turn_timeout_seconds"]
        turn_no = thread.turns + 1
        stated = {f for f, t in thread.stated.items() if turn_no - t <= lim["history_exchanges"]}
        try:
            built = build(pack, thread, message, depth_pref, cap, lim["history_exchanges"])
        except SelectionError as e:  # the context for this question is incomplete or too large: nothing is sent, nothing is shortened silently
            self.record(thread, "offline:" + e.reason, 0, 0, 0, None, ip)
            return self.offline(e.reason, diag={"detail": e.detail, "cap": cap})
        msgs, view = built.msgs, built.view
        diag = {"focus": list(view.focus), "dropped": list(view.dropped), "depth": view.depth, "history": built.keep, "tests": len(view.tests), "cap": cap, "estimate": {**built.parts, "total": built.est},
                "calls": []}
        used = [0, 0]
        fallback = None  # a reply that passed every check but a required finding, kept in case the retry fails
        fails: list[Fail] = []
        for attempt in (0, 1):
            remaining = deadline - time.monotonic()
            if attempt and remaining < lim["retry_min_seconds"]:
                break
            est = estimate_request(msgs, view.schema)
            if attempt and est > cap:  # the correction would not fit the budget: keep what we have rather than send an oversized call
                break
            try:
                held = self.budget.reserve(est, lim["max_output_tokens"])
            except BudgetExceeded as e:
                if attempt:
                    break
                self.record(thread, "offline:" + e.reason, 0, 0, 0, None, ip)
                return self.offline(e.reason, diag=diag)
            try:
                done = await self.adapter.complete(msgs, view.schema, lim["max_output_tokens"], remaining)
            except LlmError as e:
                # An indeterminate failure (dropped connection, timeout, server error) may have been processed: its input tokens stay against
                # the caps as an estimate, recorded apart from provider-reported usage. A refusal the provider stated is not charged.
                spent = self.budget.settle(held, est if e.indeterminate else 0, 0, e.indeterminate, lim["max_output_tokens"])
                diag["calls"].append({"estimated_in": est, "error": e.reason, "billing": "estimated_exposure" if e.indeterminate else "none",
                                      "counted_usd": round(spent, 6), **e.diag})
                if attempt and fallback:
                    break
                self.record(thread, "offline:" + e.reason, 0, 0, time.monotonic() - start, None, ip, diag)
                return self.offline(e.reason, e.retry_after, diag)
            self.budget.settle(held, done.tokens_in, done.tokens_out)
            diag["calls"].append({"estimated_in": est, "tokens_in": done.tokens_in, "tokens_out": done.tokens_out, "billing": "provider_reported", **done.diag})
            used[0] += done.tokens_in
            used[1] += done.tokens_out
            reply, fails = parse(done.text)
            if reply:
                depth = view.depth  # the server owns depth and the required findings; the declared purposes only label the answer
                fails = check(reply, pack, message, depth, view)
                if not fails:
                    miss = missing(reply, pack, depth, stated, view)
                    if not miss or attempt:
                        return self.finish(thread, pack, view, reply, depth, message, miss, used, start, ip, diag)
                    fallback, fails = (reply, depth, miss), [Fail("required_finding", ", ".join(miss))]
            msgs = [*msgs, {"role": "assistant", "content": done.text},
                    {"role": "user", "content": "The reply failed these checks: " + "; ".join(map(str, fails)) + ". Return the complete corrected JSON object; change only what is needed."}]
        if fallback:
            reply, depth, miss = fallback
            return self.finish(thread, pack, view, reply, depth, message, miss, used, start, ip, diag)
        thread.turns += 1
        lang = pick_lang("en", message)
        codes = sorted({f.code for f in fails})
        self.record(thread, "failed_checks", *used, time.monotonic() - start, None, ip, diag)
        out = {"status": "failed_checks", "lang": lang, "failed": codes, "message": f"{TEXT['failed'][lang]} ({', '.join(codes)}).",
               "provenance": {"case_id": pack.case_id, "case_sha256": pack.sha256, "model": self.settings.model, "validator": "failed"}}
        if self.settings.diagnostics:
            out["diagnostics"] = diag
        return 200, out

    def finish(self, thread, pack, view, reply, depth, message, miss, used, start, ip, diag) -> tuple[int, dict]:
        out, shown = assemble(reply, pack, thread, depth, pick_lang(reply.lang, message), miss, self.settings.model, view.dropped)
        remember(thread, reply, out, shown, message, self.settings.limits["operator_statements"], view.focus, conveyed(reply, pack))
        self.record(thread, "ok", *used, time.monotonic() - start, reply.refusal, ip, diag)
        if self.settings.diagnostics:
            out["diagnostics"] = diag
        return 200, out

    def record(self, thread: Thread, outcome: str, tokens_in: int, tokens_out: int, seconds: float, refusal: str | None, ip: str, diag: dict | None = None) -> None:
        """Turn metadata only (owner decision D9): no message or answer text. `focus` names a fixed topic list; estimates are token counts."""
        pack = self.store.pack(thread.case_id)
        extra = {"focus": diag["focus"], "dropped": diag["dropped"], "estimated_in": diag["estimate"]["total"]} if diag and "focus" in diag else {}
        log.info(json.dumps({"turn": uuid.uuid4().hex[:12], "mode": thread.mode, "case_sha256": pack.sha256, "model": self.settings.model,
                             "latency_ms": round(seconds * 1000), "tokens_in": tokens_in, "tokens_out": tokens_out, "outcome": outcome,
                             "refusal": refusal, "ip": self.ip_hash(ip), **extra}))
