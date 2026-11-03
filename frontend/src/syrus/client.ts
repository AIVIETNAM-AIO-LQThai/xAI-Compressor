import type {
  Answer,
  CaseFile,
  Depth,
  Health,
  Manifest,
  Mode,
  SessionInfo,
  TurnOutcome,
} from "./types";

// The extension lets the offline tests load this module in Node without a bundler.
import { STATIC, staticUrl } from "../staticData.ts";

// Same origin by default: `uvicorn backend.web:app` serves the API and this app, and the Vite dev server proxies /api.
const BASE: string = import.meta.env?.VITE_API_BASE_URL ?? "";

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(STATIC ? staticUrl(path) : `${BASE}${path}`, { signal });
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json() as Promise<T>;
}

export const getHealth = (signal?: AbortSignal) => getJson<Health>("/api/health", signal);
export const getManifest = (signal?: AbortSignal) => getJson<Manifest>("/api/cases", signal);

export function getCase(caseId: string, mode: Mode, signal?: AbortSignal) {
  return getJson<CaseFile>(`/api/cases/${encodeURIComponent(caseId)}?mode=${mode}`, signal);
}

export async function createSession(caseId: string, mode: Mode, signal?: AbortSignal): Promise<SessionInfo> {
  if (STATIC) throw new SessionError("guests_off", 503); // the showcase has no server and no chat
  const r = await fetch(`${BASE}/api/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ case_id: caseId, mode }),
    signal,
  });
  if (!r.ok) throw new SessionError(await errorCode(r), r.status);
  return r.json() as Promise<SessionInfo>;
}

export class SessionError extends Error {
  code: string;
  status: number;

  constructor(code: string, status: number) {
    super(code);
    this.code = code;
    this.status = status;
  }
}

async function errorCode(r: Response): Promise<string> {
  try {
    const body = await r.json();
    return body?.error?.code ?? body?.reason ?? `http_${r.status}`;
  } catch {
    return `http_${r.status}`;
  }
}

/** One turn. Never throws for an HTTP outcome: every documented response maps to a TurnOutcome. */
export async function sendTurn(
  token: string,
  req: { message: string; depth: Depth; caseId: string; mode: Mode },
  signal?: AbortSignal,
): Promise<TurnOutcome> {
  const started = performance.now();
  let r: Response;
  try {
    r = await fetch(`${BASE}/api/syrus/turn`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
      body: JSON.stringify({ message: req.message, depth: req.depth, case_id: req.caseId, mode: req.mode }),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    return { kind: "network" };
  }
  let body: Record<string, unknown> = {};
  try {
    body = await r.json();
  } catch {
    return { kind: "error", code: `http_${r.status}`, status: r.status };
  }
  if (r.status === 503 && body.status === "offline") {
    return {
      kind: "offline",
      reason: String(body.reason ?? "unknown"),
      retryAfter: typeof body.retry_after === "number" ? body.retry_after : null,
    };
  }
  if (r.ok && body.status === "ok") {
    return { kind: "answer", answer: body as unknown as Answer, ms: performance.now() - started };
  }
  if (r.ok && body.status === "failed_checks") {
    return { kind: "failed_checks", codes: (body.failed as string[]) ?? [] };
  }
  const err = body.error as { code?: string } | undefined;
  return { kind: "error", code: err?.code ?? `http_${r.status}`, status: r.status };
}
