import { DATA } from "./staticData";

// Scenario lab client. Everything it returns is simulated; the sensor-fault runs score recorded data with a declared change applied.
const BASE: string = import.meta.env?.VITE_API_BASE_URL ?? "";

export type FaultKind = "offset" | "drift" | "flatline" | "noise" | "dropout" | "stuck";

export interface FaultRequest {
  start: string;
  hours: number;
  fault: FaultKind;
  channel: string | null;
  onset_hours: number;
  duration_hours: number | null;
  severity: number;
  stuck_value: 0 | 1;
  seed: number;
}

interface Series {
  raw_score: Array<number | null>;
  smoothed_score: Array<number | null>;
  threshold_hit: Array<boolean | null>;
  alert: Array<boolean | null>;
}

interface Outside {
  features_outside: number;
  top: Array<{ feature: string; bins: number }>;
}

export interface FaultRun {
  evidence_class: "SIMULATED";
  window: { start: string; end: string; warmup_hours: number };
  injected: { fault: FaultKind; channel: string | null; onset: string; end: string; severity: number; severity_unit: string | null; stuck_value: 0 | 1 | null; seed: number; training_std: number | null };
  threshold: number;
  t: string[];
  baseline: Series;
  perturbed: Series;
  channel: { name: string; feature: string; baseline: Array<number | null>; perturbed: Array<number | null> } | null;
  first_alert: { baseline: string | null; perturbed: string | null };
  bins: { expected: number; baseline_valid: number; perturbed_valid: number };
  support: { baseline: Outside; perturbed: Outside; training_features: number };
  detector_identifies_fault: false;
  notes: string[];
}

export interface TwinRequest {
  hours: number;
  base_demand_kg_s: number;
  noise_percent: number;
  scenario: "none" | "demand_step" | "leak_onset";
  onset_minutes: number;
  duration_minutes: number | null;
  magnitude_kg_s: number;
  seed: number;
}

/** A generic-plant run. Inputs are the declared ground truth, kept apart; the response depends only on their total in this model. */
export interface TwinRunResult {
  evidence_class: "SIMULATED";
  source: string;
  detector: "not_applicable";
  detector_reason: string;
  scenario: TwinRequest["scenario"];
  seed: number;
  magnitude_kg_s: number;
  onset_seconds: number | null;
  end_seconds: number | null;
  units: Record<string, string>;
  band: { lower: number; upper: number; safety_min: number; safety_max: number };
  time_seconds: number[];
  inputs: { base_demand_kg_s: number[]; added_demand_kg_s: number[]; leak_kg_s: number[]; total_outflow_kg_s: number[] };
  response: { pressure_bar_g: number[]; compressor_flow_kg_s: number[]; power_kw: number[]; safety_violation: boolean[]; on: Record<string, boolean[]> };
  summary: {
    pressure_min: number;
    pressure_max: number;
    energy_kwh: number;
    safety_violations: number;
    starts: number;
    base_demand_kg: number;
    added_demand_kg: number;
    leak_kg: number;
    total_outflow_kg: number;
  };
  notes: string[];
}

export class LabError extends Error {
  code: string;

  constructor(code: string) {
    super(code);
    this.code = code;
  }
}

async function call<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const r = await fetch(`${BASE}${path}`, body === undefined ? { signal } : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal });
  if (!r.ok) {
    let code = r.status === 404 || r.status === 405 ? "not_served" : `http_${r.status}`;
    try {
      code = (await r.json())?.detail?.code ?? code;
    } catch {
      // Keep the status code.
    }
    throw new LabError(code);
  }
  return r.json() as Promise<T>;
}

export const getBackgrounds = (hours: number, signal?: AbortSignal) => call<{ backgrounds: Array<{ start: string; hours: number }> }>(`/lab/backgrounds?hours=${hours}`, undefined, signal).then((r) => r.backgrounds);
export const runFault = (req: FaultRequest, signal?: AbortSignal) => call<FaultRun>("/lab/sensor-fault", req, signal);
export const runTwin = (req: TwinRequest, signal?: AbortSignal) => call<TwinRunResult>("/lab/twin", req, signal);

/** A fixed run computed when the showcase was exported, shown as a replay; nothing is simulated in the visitor's browser. */
export interface Replay {
  id: string;
  kind: "fault" | "twin";
  title: string;
  note: string;
  runs: Array<FaultRun | TwinRunResult>;
}

export const getReplays = (signal?: AbortSignal) =>
  fetch(`${DATA}/lab/replays.json`, { signal }).then((r) => (r.ok ? (r.json() as Promise<Array<Omit<Replay, "runs">>>) : Promise.reject(new LabError("not_served"))));
export const getReplay = (id: string, signal?: AbortSignal) =>
  fetch(`${DATA}/lab/replays/${id}.json`, { signal }).then((r) => (r.ok ? (r.json() as Promise<Replay>) : Promise.reject(new LabError("not_served"))));
