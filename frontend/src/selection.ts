import type { Mode } from "./syrus/types";

/**
 * What the user chose to inspect. The URL is the source of truth; the last explicit choice is also kept for this browser tab so
 * Syrus and the navigation can return to it. Nothing is ever selected by default.
 */
export type Selection = { kind: "incident"; caseId: string; mode: Mode };

export type Route = "overview" | "investigate" | "incident" | "proof" | "twin" | "recommendation";

export interface Where {
  route: Route;
  selection: Selection | null;
}

const ROUTES: Route[] = ["overview", "investigate", "incident", "proof", "twin", "recommendation"];
const CASE = /^c-[a-z0-9]+$/;
const KEY = "aeroxai.selection";

/** `#investigate/incident/<case>/<asof|retro>`; the earlier `#investigation/<case>/<mode>` form is still understood. */
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
  return { route, selection: null };
}

export function hashFor(s: Selection): string {
  return `investigate/incident/${s.caseId}/${s.mode}`;
}

export function remember(s: Selection) {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    // Private mode or blocked storage: the URL still carries the selection.
  }
}

export function recall(): Selection | null {
  try {
    const s = JSON.parse(sessionStorage.getItem(KEY) ?? "null");
    return s?.kind === "incident" && CASE.test(s.caseId) && (s.mode === "asof" || s.mode === "retro") ? s : null;
  } catch {
    return null;
  }
}
