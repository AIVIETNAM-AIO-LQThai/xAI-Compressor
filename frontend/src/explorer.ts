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
  const r = await fetch(`${BASE}${path}`, { signal });
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

export const MAX_RANGE_DAYS = 14;
export const MAX_RAW_HOURS = 24;

const q = (s: string) => encodeURIComponent(s.replace("T", " "));
export const getEpisodes = (signal?: AbortSignal) => get<{ episodes: Episode[] }>("/explorer/episodes", signal).then((r) => r.episodes);
export const getRange = (start: string, end: string, signal?: AbortSignal) => get<RangeData>(`/explorer/range?start=${q(start)}&end=${q(end)}`, signal);
export const getRaw = (start: string, end: string, signal?: AbortSignal) => get<RawData>(`/explorer/raw?start=${q(start)}&end=${q(end)}`, signal);
export const getBin = (t: string, signal?: AbortSignal) => get<BinData>(`/explorer/bin?t=${q(t)}`, signal);

export const getOverview = (signal?: AbortSignal) => get<ExplorerOverview>("/explorer/overview", signal);

/** Dataset local time as a UTC instant, so charts drawn with `useUTC` show the recorded clock in every browser time zone. */
export const instant = (t: string) => Date.parse(`${t.replace(" ", "T")}Z`);
