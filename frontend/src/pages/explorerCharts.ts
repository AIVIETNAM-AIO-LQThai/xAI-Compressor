import { instant, type ExplorerOverview, type RangeData, type RawData } from "../explorer";
import { runs } from "./runs";

// Chart colours, checked for colour-vision separation and contrast on the panel surface. Text stays in text tokens.
export const SCORE = "#c3cad2";
export const ALERT = "#e8601c";
export const INCIDENT = "#4f8fe8";
export const MUTED = "#aab2bb";
export const SURFACE = "#11161b";
const FAINT = "rgba(195,202,210,.38)";
const BIN = 300_000;

export const num = (v: number) => v.toLocaleString("en-US", { maximumFractionDigits: 0 });
export const stamp = (ms: number) => new Date(ms).toISOString().slice(0, 16).replace("T", " ");

type Series = Array<number | null>;

const band = (a: number, b: number, color: string, name = "") => [{ name, xAxis: a, itemStyle: { color } }, { xAxis: b }];

export function detectorOption(r: RangeData, o: ExplorerOverview, narrow: boolean) {
  const t = r.t.map(instant);
  const pos = (v: number | null) => (v !== null && v > 0 ? v : null); // a log axis has no zero
  const smooth = r.smoothed_score.map(pos);
  const lo = t[0];
  const hi = t[t.length - 1] + BIN;
  const clip = (a: number, b: number): [number, number] | null => (b > lo && a < hi ? [Math.max(a, lo), Math.min(b, hi)] : null);
  const areas = [
    ...runs(t, r.alert).map(([a, b]) => band(a, b, "rgba(232,96,28,.22)")),
    ...(["train", "calibration"] as const).flatMap((k, i) => {
      const c = clip(instant(o.splits[k][0]), instant(o.splits[k][1]));
      return c ? [band(c[0], c[1], i ? "rgba(255,255,255,.10)" : "rgba(255,255,255,.06)", k === "train" ? "Train" : "Calibration")] : [];
    }),
  ];
  const incidents = o.incidents.flatMap((i) => {
    const c = clip(instant(i.start), instant(i.end));
    return c ? [band(c[0], c[1], "rgba(79,143,232,.20)", `Incident ${i.id}`)] : [];
  });
  const restarts = t.flatMap((x, i) => (r.restart[i] && smooth[i] !== null ? [[x, smooth[i]]] : []));

  return {
    useUTC: true,
    animation: false,
    grid: { left: narrow ? 46 : 58, right: narrow ? 12 : 24, top: 20, bottom: 28 },
    xAxis: {
      type: "time",
      min: lo,
      max: hi,
      axisLine: { lineStyle: { color: "rgba(255,255,255,.18)" } },
      axisTick: { show: false },
      axisLabel: { color: MUTED, fontSize: 12, hideOverlap: true },
      splitLine: { show: false },
    },
    yAxis: {
      type: "log",
      logBase: 10,
      axisLabel: { color: MUTED, fontSize: 12, formatter: (v: number) => (v >= 1e6 ? `${num(v / 1e6)}M` : v >= 1000 ? `${num(v / 1000)}k` : num(v)) },
      splitLine: { lineStyle: { color: "rgba(255,255,255,.07)" } },
    },
    axisPointer: { link: [{ xAxisIndex: "all" }] },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "line", lineStyle: { color: "rgba(255,255,255,.35)" } },
      backgroundColor: "#181e24",
      borderColor: "rgba(255,255,255,.14)",
      textStyle: { color: "#f2f3f5", fontSize: 13 },
      formatter: (items: Array<{ dataIndex: number; seriesIndex: number }>) => {
        const i = items.find((p) => p.seriesIndex === 0)?.dataIndex;
        if (i === undefined) return "";
        const head = `<b>${stamp(t[i])}</b>`;
        if (r.smoothed_score[i] === null) return `${head}<br/>No valid data (recording gap)`;
        const state = r.split[i] === "test" ? (r.alert[i] ? "Alert" : r.threshold_hit[i] ? "At or above the threshold, no alert yet" : "Below the threshold") : "Alerts not evaluated in this period";
        return [head, `Smoothed score ${num(r.smoothed_score[i]!)}`, `Raw score ${num(r.raw_score[i]!)}`, state, r.restart[i] ? "Smoothing restarted after a gap" : ""].filter(Boolean).join("<br/>");
      },
    },
    series: [
      {
        type: "line",
        name: "Smoothed score",
        data: t.map((x, i) => [x, smooth[i]]),
        showSymbol: false,
        connectNulls: false,
        lineStyle: { width: 1.6, color: SCORE },
        itemStyle: { color: SCORE },
        markLine: { silent: true, symbol: "none", lineStyle: { color: MUTED, type: "dashed", width: 1 }, label: { show: false }, data: [{ yAxis: r.threshold }] },
        markArea: { silent: true, label: { color: MUTED, fontSize: 11, position: "insideTopLeft" }, data: areas },
      },
      {
        type: "line",
        name: "Raw score",
        data: t.map((x, i) => [x, pos(r.raw_score[i])]),
        showSymbol: false,
        connectNulls: false,
        lineStyle: { width: 1, color: FAINT },
        itemStyle: { color: FAINT },
        tooltip: { show: false },
      },
      {
        type: "scatter",
        name: "Smoothing restart",
        data: restarts,
        symbol: "triangle",
        symbolSize: narrow ? 7 : 9,
        itemStyle: { color: "transparent", borderColor: "#f2f3f5", borderWidth: 1.5 },
        z: 3,
        tooltip: { show: false },
      },
      {
        type: "line",
        name: "Documented incident",
        data: [],
        itemStyle: { color: INCIDENT },
        markArea: { silent: true, label: { color: "#f2f3f5", fontSize: 12, fontWeight: 600, position: "insideTop" }, data: incidents },
      },
    ],
  };
}

export interface Row {
  id: string;
  values: Series;
  min?: Series;
  max?: Series;
  step?: boolean;
  fixed?: [number, number];
}

/** One grid per channel, aligned on a shared time axis; each row names its channel, and lines break at gaps. */
export function stackOption(t: string[], rows: Row[], rowHeight: number, narrow: boolean) {
  const x = t.map(instant);
  const left = narrow ? 46 : 58;
  const top = (i: number) => 8 + i * (rowHeight + 10);
  const last = rows.length - 1;
  return {
    useUTC: true,
    animation: false,
    axisPointer: { link: [{ xAxisIndex: "all" }] },
    title: rows.map((r, i) => ({ text: r.id, left, top: top(i), textStyle: { color: MUTED, fontSize: 12, fontWeight: 600 } })),
    grid: rows.map((_, i) => ({ left, right: narrow ? 12 : 24, top: top(i) + 18, height: rowHeight - 18 })),
    xAxis: rows.map((_, i) => ({
      type: "time",
      gridIndex: i,
      min: x[0],
      max: x[x.length - 1],
      axisLine: { lineStyle: { color: "rgba(255,255,255,.18)" } },
      axisTick: { show: false },
      axisLabel: { show: i === last, color: MUTED, fontSize: 12, hideOverlap: true },
      splitLine: { show: false },
    })),
    yAxis: rows.map((r, i) => ({
      type: "value",
      gridIndex: i,
      scale: !r.fixed,
      min: r.fixed?.[0],
      max: r.fixed?.[1],
      splitNumber: 2,
      axisLabel: { color: MUTED, fontSize: 11 },
      splitLine: { lineStyle: { color: "rgba(255,255,255,.06)" } },
    })),
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "line", lineStyle: { color: "rgba(255,255,255,.35)" } },
      backgroundColor: "#181e24",
      borderColor: "rgba(255,255,255,.14)",
      textStyle: { color: "#f2f3f5", fontSize: 13 },
      formatter: (items: Array<{ dataIndex: number }>) => {
        const i = items[0]?.dataIndex;
        if (i === undefined) return "";
        return [`<b>${stamp(x[i])}</b>`, ...rows.map((r) => `${r.id}: ${r.values[i] === null ? "no data" : Number(r.values[i]!.toFixed(3))}`)].join("<br/>");
      },
    },
    series: rows.flatMap((r, i) => {
      const line = (name: string, values: Series, color: string, width: number) => ({
        type: "line",
        name,
        xAxisIndex: i,
        yAxisIndex: i,
        data: x.map((v, k) => [v, values[k]]),
        showSymbol: false,
        connectNulls: false,
        step: r.step ? ("end" as const) : false,
        lineStyle: { width, color },
        itemStyle: { color },
        ...(r.step ? { areaStyle: { color: "rgba(195,202,210,.14)" } } : {}),
      });
      return [...(r.min ? [line(`${r.id} min`, r.min, FAINT, 1)] : []), ...(r.max ? [line(`${r.id} max`, r.max, FAINT, 1)] : []), line(r.id, r.values, SCORE, 1.5)];
    }),
  };
}

/** Rows for the five-minute summary (mean with min and max) or the raw ten-second values. */
export function channelRows(names: string[], digital: boolean, src: RangeData | RawData): Row[] {
  if ("channels" in src && "split" in src) {
    const c = (src as RangeData).channels;
    return names.map((id) => (digital ? { id, values: c[id].active_ratio, step: true, fixed: [0, 1] as [number, number] } : { id, values: c[id].mean, min: c[id].min, max: c[id].max }));
  }
  const c = (src as RawData).channels;
  return names.map((id) => (digital ? { id, values: c[id], step: true, fixed: [0, 1] as [number, number] } : { id, values: c[id] }));
}
