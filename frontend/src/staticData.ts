/**
 * The showcase build (`vite build --mode showcase`) has no server: every read-only answer was precomputed by `scripts/export_showcase.py`
 * through the real application and stored as files under `data/`. Recorded days are stored one file per day without gap rows; the client
 * joins the days it needs and inserts gap rows with the server's own rule, so a line never runs across a gap at a day boundary either.
 */
export const STATIC = import.meta.env?.MODE === "showcase";
export const DATA = `${import.meta.env?.BASE_URL ?? "/"}data`;

const safe = (s: string) => s.replace(" ", "T").replace(/:/g, "-").slice(0, 16);

/** The file holding the answer the server would give for this read-only path. */
export function staticUrl(path: string): string {
  const [p, q = ""] = path.split("?");
  const qs = new URLSearchParams(q);
  if (p.startsWith("/api/cases/")) return `${DATA}/api/cases/${p.slice("/api/cases/".length)}.${qs.get("mode") ?? "asof"}.json`;
  if (p === "/explorer/episode") return `${DATA}/explorer/episode/${safe(qs.get("start") ?? "")}.json`;
  return `${DATA}${p}.json`;
}

type Columns = Record<string, Array<unknown>>;

function flatten(o: Record<string, unknown>, prefix = "", out: Columns = {}): Columns {
  for (const [k, v] of Object.entries(o)) {
    if (Array.isArray(v)) out[prefix + k] = v;
    else if (v && typeof v === "object") flatten(v as Record<string, unknown>, `${prefix}${k}.`, out);
  }
  return out;
}

function unflatten(cols: Columns): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [path, v] of Object.entries(cols)) {
    const keys = path.split(".");
    let at = out;
    for (const k of keys.slice(0, -1)) at = (at[k] ??= {}) as Record<string, unknown>;
    at[keys[keys.length - 1]] = v;
  }
  return out;
}

const ms = (t: string) => Date.parse(`${t.replace(" ", "T")}Z`);
const stamp = (n: number) => new Date(n).toISOString().slice(0, 19).replace("T", " ");

/** Joins day files, keeps rows inside [start, end] (or [start, end) for raw samples, as the server does), and inserts an empty row after every pause longer than `gapMs`, as the server does. */
export function stitch(parts: Array<Record<string, unknown>>, start: string, end: string, gapMs: number, endExclusive = false): Record<string, unknown> {
  const cols = parts.map((p) => flatten(p));
  const keys = Object.keys(cols[0] ?? { t: [] }).filter((k) => Array.isArray(cols[0]?.[k]) && cols[0][k].length === (cols[0].t ?? []).length);
  const all: Columns = Object.fromEntries(keys.map((k) => [k, cols.flatMap((c) => c[k] ?? [])]));
  const lo = ms(start);
  const hi = ms(end);
  const out: Columns = Object.fromEntries(keys.map((k) => [k, []]));
  let prev: number | null = null;
  (all.t as string[]).forEach((t, i) => {
    const x = ms(t);
    if (x < lo || x > hi || (endExclusive && x === hi)) return;
    if (prev !== null && x - prev > gapMs) for (const k of keys) out[k].push(k === "t" ? stamp(x - gapMs) : null);
    for (const k of keys) out[k].push(all[k][i]);
    prev = x;
  });
  return unflatten(out);
}

/** Every calendar day a window touches, as YYYY-MM-DD. */
export function days(start: string, end: string): string[] {
  const out: string[] = [];
  for (let d = ms(`${start.slice(0, 10)} 00:00:00`); d <= ms(end); d += 86_400_000) out.push(new Date(d).toISOString().slice(0, 10));
  return out;
}
