import type { CaseRecord, Depth, TurnOutcome } from "./types";

export const FINDING_TITLES: Record<string, string> = {
  "F-MARGIN": "Threshold margin",
  "F-EWMA-MEMORY": "Smoothing memory",
  "F-EVIDENCE-TIMING": "Evidence timing",
  "F-CF-NONUNIQUE": "Repairs do not single out a group",
  "F-CF-PERSIST": "Persistence path",
  "F-TRAIN-RANGE": "Training range",
  "F-GAP": "Window completeness",
  "F-CANDIDATES": "Open candidates",
  "F-IDENT": "Leakage vs demand",
  "F-BRIDGE": "Physical action gate",
  "F-LABEL": "Documented label",
  "F-TIMING": "Alert vs documented onset",
  "F-SELECTION": "How cases were chosen",
  "F-ALERT-BURDEN": "Alert burden",
  "F-DISCRIMINATION": "Discrimination",
};

const RECORD_TITLES: Record<string, string> = {
  "alert.time": "Alert time",
  "alert.threshold": "Threshold",
  "alert.smoothed": "Smoothed score",
  "alert.raw": "Raw score",
  "alert.margin": "Margin",
  "alert.prev_smoothed": "Previous smoothed state",
  "alert.memory": "Memory share",
  "alert.hits": "Hit pattern",
  "detector.ewma": "Smoothing rule",
  "detector.persistence": "Persistence rule",
  "detector.calibration_bins": "Calibration bins",
  "detector.repair": "Repair method",
  "bridge.status": "Physical bridge",
  "label.incident": "Documented label",
  "label.timing": "Alert vs onset",
  "case.status": "Case status",
  "temporal.window": "Evidence window",
  "unavailable:definitions": "Signal definitions",
  "unavailable:documents": "Manuals and maintenance records",
  "unavailable:raw_series": "Raw telemetry",
  "unavailable:demand_measurement": "Downstream demand measurement",
};

export function groupName(id: string) {
  return id.replace(/_/g, " ");
}

export function recordTitle(id: string): string {
  if (RECORD_TITLES[id]) return RECORD_TITLES[id];
  const [prefix, rest = ""] = id.split(/[.:]/);
  if (prefix === "contrib") return `${groupName(rest)} contribution`;
  if (prefix === "calrank") return `${groupName(rest)} calibration rank`;
  if (prefix === "cf") return rest === "prewindow" ? "Pre-window bound" : `${groupName(rest)} repair`;
  if (prefix === "temporal") return `Evidence at ${rest.slice(0, 2)}:${rest.slice(2)}`;
  if (prefix === "hyp") return groupName(rest);
  if (prefix === "measured") return groupName(rest.replace("__", " · "));
  if (prefix === "bench") return `Benchmark · ${groupName(rest)}`;
  return id;
}

export function title(id: string): string {
  return FINDING_TITLES[id] ?? recordTitle(id);
}

export const DEPTHS: Array<{ id: Depth; label: string }> = [
  { id: "quick", label: "Quick Note" },
  { id: "investigation", label: "Investigation" },
  { id: "technical", label: "Technical" },
];

export const SUGGESTIONS = [
  "Why did this alert fire?",
  "Was this definitely a leak?",
  "What did the motor_current repair test?",
  "What evidence is missing?",
];

/** The display of a record as short text, for places that need one line. */
export function brief(r: CaseRecord): string {
  const d = r.display;
  if (typeof d === "string") return d;
  const entries = Object.entries(d).filter(([, v]) => typeof v === "string" || typeof v === "number");
  return entries.map(([k, v]) => `${groupName(k)} ${v}`).join(" · ");
}

const OFFLINE: Record<string, string> = {
  key_missing: "Syrus is not configured on this server.",
  caps_unconfirmed: "Syrus is waiting for the spending caps to be confirmed.",
  kill_switch: "Syrus has been switched off.",
  adapter_missing: "Syrus is not configured on this server.",
  budget_daily: "The daily budget is used up. Try again tomorrow.",
  budget_total: "The development budget is used up.",
  rate_limited: "The model provider is limiting requests.",
  busy: "Syrus is answering someone else.",
  timeout: "The model provider took too long.",
  provider_error: "The model provider could not be reached.",
  provider_auth: "The model provider refused the server's credentials.",
  context_over_budget: "This question needs more context than one answer can hold. Ask about one topic at a time.",
  context_incomplete: "The evidence for this question is incomplete, so Syrus will not answer it.",
};

const ERRORS: Record<string, string> = {
  session_expired: "The session ended. Send the question again to start a new one.",
  token_invalid: "The session ended. Send the question again to start a new one.",
  token_missing: "The session ended. Send the question again to start a new one.",
  turn_limit: "This conversation reached its turn limit. Reload to start a new one.",
  session_rate_limit: "Too many sessions from this network. Try again later.",
  asof_chat_disabled: "Chat is available in Retrospective mode only.",
  message_too_long: "That message is too long.",
  empty_message: "Write a question first.",
  turn_in_progress: "Syrus is still working on the previous question.",
  case_mismatch: "That conversation belongs to a different case.",
  case_not_found: "That case does not exist.",
};

export interface Notice {
  tone: "warn" | "error";
  text: string;
  retryAfter?: number;
  canRetry: boolean;
}

export function notice(o: Exclude<TurnOutcome, { kind: "answer" }>): Notice {
  switch (o.kind) {
    case "offline":
      return { tone: "warn", text: OFFLINE[o.reason] ?? "Syrus is unavailable.", retryAfter: o.retryAfter ?? undefined, canRetry: true };
    case "failed_checks":
      return { tone: "warn", text: "Syrus could not produce an answer that passes the evidence checks.", canRetry: true };
    case "network":
      return { tone: "error", text: "Cannot reach the server.", canRetry: true };
    case "error":
      return { tone: "error", text: ERRORS[o.code] ?? "Something went wrong.", canRetry: !["turn_limit", "asof_chat_disabled", "case_not_found"].includes(o.code) };
  }
}
