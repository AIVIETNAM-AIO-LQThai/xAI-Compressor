import assert from "node:assert/strict";
import { afterEach, describe, it } from "node:test";

import { createSession, SessionError, sendTurn } from "../src/syrus/client.ts";
import { notice } from "../src/syrus/labels.ts";

const real = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = real;
});

function reply(status: number, body: unknown, raw?: string) {
  globalThis.fetch = (async () => new Response(raw ?? JSON.stringify(body), { status })) as typeof fetch;
}

const req = { message: "Why?", depth: "quick" as const, caseId: "c-h3vz", mode: "retro" as const };
const answer = { status: "ok", blocks: [], also_relevant: [], provenance: { validator: "passed" } };

describe("turn outcomes", () => {
  it("maps a validated answer", async () => {
    reply(200, answer);
    const o = await sendTurn("t", req);
    assert.equal(o.kind, "answer");
  });

  it("maps failed checks to their codes only", async () => {
    reply(200, { status: "failed_checks", failed: ["forbidden_phrase"], message: "text" });
    assert.deepEqual(await sendTurn("t", req), { kind: "failed_checks", codes: ["forbidden_phrase"] });
  });

  it("maps offline states with their retry time", async () => {
    reply(503, { status: "offline", reason: "rate_limited", retry_after: 37 });
    assert.deepEqual(await sendTurn("t", req), { kind: "offline", reason: "rate_limited", retryAfter: 37 });
    reply(503, { status: "offline", reason: "busy", retry_after: null });
    assert.deepEqual(await sendTurn("t", req), { kind: "offline", reason: "busy", retryAfter: null });
  });

  it("maps documented HTTP errors by code and status", async () => {
    for (const [status, code] of [[401, "session_expired"], [403, "asof_chat_disabled"], [409, "case_mismatch"], [429, "turn_limit"], [400, "message_too_long"]] as const) {
      reply(status, { error: { code } });
      assert.deepEqual(await sendTurn("t", req), { kind: "error", code, status });
    }
  });

  it("treats an unreadable body as an HTTP error, not a crash", async () => {
    reply(502, null, "<html>bad gateway</html>");
    assert.deepEqual(await sendTurn("t", req), { kind: "error", code: "http_502", status: 502 });
  });

  it("reports a lost connection and re-throws a cancellation", async () => {
    globalThis.fetch = (async () => {
      throw new TypeError("fetch failed");
    }) as typeof fetch;
    assert.deepEqual(await sendTurn("t", req), { kind: "network" });
    globalThis.fetch = (async () => {
      throw new DOMException("aborted", "AbortError");
    }) as typeof fetch;
    await assert.rejects(sendTurn("t", req), { name: "AbortError" });
  });

  it("sends the bearer token, the depth, and the case and mode", async () => {
    let seen: { headers: Record<string, string>; body: Record<string, string> } | undefined;
    globalThis.fetch = (async (_url: string, init: RequestInit) => {
      seen = { headers: init.headers as Record<string, string>, body: JSON.parse(init.body as string) };
      return new Response(JSON.stringify(answer), { status: 200 });
    }) as typeof fetch;
    await sendTurn("tok", { ...req, depth: "technical" });
    assert.equal(seen?.headers.Authorization, "Bearer tok");
    assert.deepEqual(seen?.body, { message: "Why?", depth: "technical", case_id: "c-h3vz", mode: "retro" });
  });
});

describe("sessions", () => {
  it("raises the server's error code when a session cannot be created", async () => {
    reply(404, { error: { code: "case_not_found" } });
    await assert.rejects(createSession("c-x", "retro"), (e: unknown) => e instanceof SessionError && e.code === "case_not_found" && e.status === 404);
  });
});

describe("user-facing messages", () => {
  const offline = ["guests_off", "key_missing", "caps_unconfirmed", "kill_switch", "adapter_missing", "budget_daily", "budget_total", "rate_limited", "busy", "paced", "timeout", "provider_error", "provider_auth", "context_over_budget", "context_incomplete"];
  const errors = ["session_expired", "token_invalid", "token_missing", "turn_limit", "session_rate_limit", "asof_chat_disabled", "message_too_long", "empty_message", "turn_in_progress", "case_mismatch", "case_not_found", "guest_quota", "preview_locked", "rate_limited"];

  it("has a specific message for every offline reason and error code the backend can return", () => {
    for (const reason of offline) assert.notEqual(notice({ kind: "offline", reason, retryAfter: null }).text, "Syrus is unavailable.", reason);
    for (const code of errors) assert.notEqual(notice({ kind: "error", code, status: 400 }).text, "Something went wrong.", code);
  });

  it("never offers a retry that cannot succeed, and never shows provider text", () => {
    assert.equal(notice({ kind: "error", code: "turn_limit", status: 429 }).canRetry, false);
    assert.equal(notice({ kind: "error", code: "asof_chat_disabled", status: 403 }).canRetry, false);
    assert.equal(notice({ kind: "error", code: "guest_quota", status: 429 }).canRetry, false);
    assert.equal(notice({ kind: "offline", reason: "guests_off", retryAfter: null }).canRetry, false);
    assert.equal(notice({ kind: "offline", reason: "rate_limited", retryAfter: 12 }).retryAfter, 12);
    assert.ok(!notice({ kind: "offline", reason: "provider_error", retryAfter: null }).text.includes("ReadError"));
  });
});
