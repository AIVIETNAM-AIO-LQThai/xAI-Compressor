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

export const getOverview = (signal?: AbortSignal) => get<ExplorerOverview>("/explorer/overview", signal);

/** Dataset local time as a UTC instant, so charts drawn with `useUTC` show the recorded clock in every browser time zone. */
export const instant = (t: string) => Date.parse(`${t.replace(" ", "T")}Z`);
