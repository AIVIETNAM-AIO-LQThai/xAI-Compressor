import type { Mode } from "./syrus/types";

/**
 * What the user chose to inspect. The URL is the source of truth; the last explicit choice is also kept for this browser tab so
 * Syrus and the navigation can return to it. Nothing is ever selected by default.
 *
 * Only an incident has a reviewed Syrus case file. An alert episode is identified by its start; an interval is any recorded window.
 * Times are dataset local time, written `YYYY-MM-DDTHH:MM` so they survive a URL.
 */
export type Selection =
  | { kind: "incident"; caseId: string; mode: Mode }
  | { kind: "episode"; start: string }
  | { kind: "interval"; start: string; end: string };

export type Route = "overview" | "investigate" | "explore" | "incident" | "proof" | "twin" | "recommendation";

export interface Where {
  route: Route;
  selection: Selection | null;
}

const ROUTES: Route[] = ["overview", "investigate", "explore", "incident", "proof", "twin", "recommendation"];
const CASE = /^c-[a-z0-9]+$/;
const STAMP = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/;
const KEY = "aeroxai.selection";

/** `#investigate/incident/<case>/<asof|retro>`, `#explore/episode/<start>`, `#explore/interval/<start>/<end>`; the earlier `#investigation/<case>/<mode>` form is still understood. */
export function parse(hash: string): Where {
  const parts = hash.replace(/^#/, "").split("/");
  if (parts[0] === "investigation") {
    const [, caseId, mode] = parts;
    return { route: "investigate", selection: CASE.test(caseId ?? "") ? { kind: "incident", caseId, mode: mode === "asof" ? "asof" : "retro" } : null };
  }
  const route = ROUTES.find((r) => r === parts[0]) ?? "overview";
  if (route === "investigate" && parts[1] === "incident" && CASE.test(parts[2] ?? "")) {
    return { route, selection: { kind: "incident", caseId: parts[2], mode: parts[3] === "asof" ? "asof" : "retro" } };
  }
  if (route === "explore" && parts[1] === "episode" && STAMP.test(parts[2] ?? "")) return { route, selection: { kind: "episode", start: parts[2] } };
  if (route === "explore" && parts[1] === "interval" && STAMP.test(parts[2] ?? "") && STAMP.test(parts[3] ?? "") && parts[2] < parts[3]) {
    return { route, selection: { kind: "interval", start: parts[2], end: parts[3] } };
  }
  return { route, selection: null };
}

export function hashFor(s: Selection): string {
  if (s.kind === "incident") return `investigate/incident/${s.caseId}/${s.mode}`;
  return s.kind === "episode" ? `explore/episode/${s.start}` : `explore/interval/${s.start}/${s.end}`;
}

/** The latest choice is kept, plus the latest incident and the latest explorer view separately, so looking at an interval does not forget the incident. */
export function remember(s: Selection) {
  try {
    const json = JSON.stringify(s);
    sessionStorage.setItem(KEY, json);
    sessionStorage.setItem(`${KEY}.${s.kind === "incident" ? "incident" : "view"}`, json);
  } catch {
    // Private mode or blocked storage: the URL still carries the selection.
  }
}

function read(key: string): Selection | null {
  try {
    const s = JSON.parse(sessionStorage.getItem(key) ?? "null");
    if (s?.kind === "incident") return CASE.test(s.caseId) && (s.mode === "asof" || s.mode === "retro") ? s : null;
    if (s?.kind === "episode") return STAMP.test(s.start) ? s : null;
    if (s?.kind === "interval") return STAMP.test(s.start) && STAMP.test(s.end) && s.start < s.end ? s : null;
    return null;
  } catch {
    return null;
  }
}

export const recall = () => read(KEY);

/** The last incident the user chose, for the Investigation link and for Syrus, which only knows incidents. */
export function recallIncident(): Extract<Selection, { kind: "incident" }> | null {
  const s = read(`${KEY}.incident`);
  return s?.kind === "incident" ? s : null;
}

/** The last episode or interval the user chose in the explorer. */
export function recallView(): Exclude<Selection, { kind: "incident" }> | null {
  const s = read(`${KEY}.view`);
  return s && s.kind !== "incident" ? s : null;
}
