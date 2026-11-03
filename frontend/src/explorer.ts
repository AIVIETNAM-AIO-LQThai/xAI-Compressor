import { DATA, STATIC, days, staticUrl, stitch } from "./staticData";

// Read-only recorded-data explorer. Timestamps are dataset local time as "YYYY-MM-DD HH:MM:SS"; a null value is a recording gap.
const BASE: string = import.meta.env?.VITE_API_BASE_URL ?? "";

type Span = [string, string];

export interface ExplorerOverview {
  source: { dataset: string; doi: string; license: string; raw_rows: number; sampling_seconds: number; channels: { analog: string[]; digital: string[] } };
  span: Span;
  splits: { train: Span; calibration: Span; test: Span };
  bins: { valid: number; expected: number; share: number; train: number; calibration: number; test: number };
  threshold: number;
  rule: { ewma_alpha: number; hits: number; window: number; reset_gap_minutes: number };
  episodes: { total: number; linked: number; not_linked: number };
  incidents: Array<{ id: number; start: string; end: string; retrospective: true }>;
  activity: { bucket_hours: number; t: string[]; max_smoothed: Array<number | null>; bins: number[]; alert: boolean[] };
}

export class ExplorerError extends Error {
  code: string;

  constructor(code: string) {
    super(code);
    this.code = code;
  }
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(STATIC ? staticUrl(path) : `${BASE}${path}`, { signal });
  if (STATIC && r.status === 404) throw new ExplorerError(path.startsWith("/explorer/episode?") ? "episode_not_found" : "not_found");
  if (!r.ok) {
    let code = `http_${r.status}`;
    try {
      code = (await r.json())?.detail?.code ?? code;
    } catch {
      // Keep the status code.
    }
    throw new ExplorerError(code);
  }
  return r.json() as Promise<T>;
}

export interface Episode {
  start: string;
  end: string;
  alert_bins: number;
  peak_smoothed: number;
  peak_time: string;
  linked_incident: number | null;
}

/** Five-minute detail; a null is a gap or a value the period does not have. */
export interface RangeData {
  start: string;
  end: string;
  threshold: number;
  step_seconds: number;
  t: string[];
  split: Array<"train" | "calibration" | "test" | null>;
  raw_score: Array<number | null>;
  smoothed_score: Array<number | null>;
  threshold_hit: Array<boolean | null>;
  alert: Array<boolean | null>;
  restart: Array<boolean | null>;
  coverage: Array<number | null>;
  channels: Record<string, Record<string, Array<number | null>>>;
}

export interface RawData {
  start: string;
  end: string;
  step_seconds: number;
  t: string[];
  channels: Record<string, Array<number | null>>;
}

export interface BinData {
  t: string;
  split: "train" | "calibration" | "test";
  raw_score: number;
  smoothed_score: number;
  threshold: number;
  threshold_hit: boolean;
  alert: boolean | null;
  restart: boolean;
  coverage: number;
  samples: number;
}

/** Computed on request with the frozen detector's explanation code; not a reviewed case file. */
export interface EpisodeEvidence {
  episode: Episode;
  status: "computed_not_reviewed";
  causal_claim: false;
  not_computed: string[];
  linked_incident: number | null;
  evidence: {
    time: string;
    threshold: number;
    rule: { hits: number; window: number; alpha: number };
    smoothed_score: number;
    raw_score: number;
    restart: boolean;
    carried_share: number | null;
    recent: Array<{ time: string; smoothed_score: number; hit: boolean; restart: boolean }>;
    groups: Array<{ group: string; smoothed_contribution: number; share: number; calibration_percentile: number }>;
    window: { complete: boolean; bins: Array<{ time: string; percentiles: Record<string, number> }> };
  };
}

export const MAX_RANGE_DAYS = 14;
export const MAX_RAW_HOURS = 24;

const q = (s: string) => encodeURIComponent(s.replace("T", " "));
export const getEpisodes = (signal?: AbortSignal) => get<{ episodes: Episode[] }>("/explorer/episodes", signal).then((r) => r.episodes);
// The showcase joins one file per recorded day; days without a file have no recording.
async function dayFiles(kind: "range" | "raw" | "bins", start: string, end: string, signal?: AbortSignal) {
  const got = await Promise.all(days(start, end).map((d) => fetch(`${DATA}/explorer/${kind}/${d}.json`, { signal }).then((r) => (r.ok ? r.json() : null))));
  return got.filter(Boolean) as Array<Record<string, unknown>>;
}

const full = (s: string) => `${s.replace("T", " ").slice(0, 16)}:00`;

export async function getRange(start: string, end: string, signal?: AbortSignal): Promise<RangeData> {
  if (!STATIC) return get<RangeData>(`/explorer/range?start=${q(start)}&end=${q(end)}`, signal);
  if (instant(end) - instant(start) > MAX_RANGE_DAYS * 86_400_000 || instant(end) <= instant(start)) throw new ExplorerError("range_invalid");
  const parts = await dayFiles("range", start, end, signal);
  const first = parts[0] as { threshold?: number } | undefined;
  return { ...(stitch(parts, full(start), full(end), 300_000) as object), start: full(start), end: full(end), threshold: first?.threshold ?? 0, step_seconds: 300 } as RangeData;
}

export async function getRaw(start: string, end: string, signal?: AbortSignal): Promise<RawData> {
  if (!STATIC) return get<RawData>(`/explorer/raw?start=${q(start)}&end=${q(end)}`, signal);
  if (instant(end) - instant(start) > MAX_RAW_HOURS * 3_600_000 || instant(end) <= instant(start)) throw new ExplorerError("range_invalid");
  const parts = await dayFiles("raw", start, end, signal);
  return { ...(stitch(parts, full(start), full(end), 60_000, true) as object), start: full(start), end: full(end), step_seconds: 10 } as RawData;
}
export const getEpisodeEvidence = (start: string, signal?: AbortSignal) => get<EpisodeEvidence>(`/explorer/episode?start=${q(start)}`, signal);
export async function getBin(t: string, signal?: AbortSignal): Promise<BinData> {
  if (!STATIC) return get<BinData>(`/explorer/bin?t=${q(t)}`, signal);
  const [day] = await dayFiles("bins", t, t, signal);
  const found = (day as Record<string, BinData> | undefined)?.[full(t)];
  if (!found) throw new ExplorerError("bin_not_found");
  return found;
}

export const getOverview = (signal?: AbortSignal) => get<ExplorerOverview>("/explorer/overview", signal);

/** Dataset local time as a UTC instant, so charts drawn with `useUTC` show the recorded clock in every browser time zone. */
export const instant = (t: string) => Date.parse(`${t.replace(" ", "T")}Z`);
