import { instant } from "../explorer";
import type { FaultRun, TwinRunResult } from "../lab";
import { ALERT, MUTED, SCORE, num, stamp } from "./explorerCharts";
import { runs } from "./runs";

const FAINT = "rgba(195,202,210,.38)";
const AMBER = "rgba(245,185,66,.16)"; // simulated, as everywhere else in the interface
const AXIS = { axisLine: { lineStyle: { color: "rgba(255,255,255,.18)" } }, axisTick: { show: false }, splitLine: { show: false } };
const TIP = { backgroundColor: "#181e24", borderColor: "rgba(255,255,255,.14)", textStyle: { color: "#f2f3f5", fontSize: 13 } };
const Y = { splitLine: { lineStyle: { color: "rgba(255,255,255,.07)" } } };

const pos = (v: number | null) => (v !== null && v > 0 ? v : null);

/** The injected span, shaded and labelled as simulated, with the onset marked. */
function injected(r: FaultRun) {
  return { silent: true, label: { color: "#f5b942", fontSize: 11, position: "insideTopLeft" }, data: [[{ name: "Injected change (simulated)", xAxis: instant(r.injected.onset), itemStyle: { color: AMBER } }, { xAxis: instant(r.injected.end) }]] };
}

export function responseOption(r: FaultRun, narrow: boolean) {
  const t = r.t.map(instant);
  const alerts = runs(t, r.perturbed.alert).map(([a, b]) => [{ xAxis: a, itemStyle: { color: "rgba(232,96,28,.24)" } }, { xAxis: b }]);
  return {
    useUTC: true,
    animation: false,
    grid: { left: narrow ? 46 : 58, right: narrow ? 12 : 24, top: 22, bottom: 28 },
    xAxis: { type: "time", min: t[0], max: t[t.length - 1], ...AXIS, axisLabel: { color: MUTED, fontSize: 12, hideOverlap: true } },
    yAxis: { type: "log", logBase: 10, ...Y, axisLabel: { color: MUTED, fontSize: 12, formatter: (v: number) => (v >= 1e6 ? `${num(v / 1e6)}M` : v >= 1000 ? `${num(v / 1000)}k` : num(v)) } },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "line", lineStyle: { color: "rgba(255,255,255,.35)" } },
      ...TIP,
      formatter: (items: Array<{ dataIndex: number }>) => {
        const i = items[0]?.dataIndex;
        if (i === undefined) return "";
        const p = r.perturbed.smoothed_score[i];
        if (p === null) return `<b>${stamp(t[i])}</b><br/>No valid data in this bin (gap)`;
        return [`<b>${stamp(t[i])}</b>`, `With the change: ${num(p)}${r.perturbed.alert[i] ? " · alert" : ""}`, `Unchanged recording: ${r.baseline.smoothed_score[i] === null ? "no data" : num(r.baseline.smoothed_score[i]!)}`].join("<br/>");
      },
    },
    series: [
      {
        type: "line",
        name: "With the change",
        data: t.map((x, i) => [x, pos(r.perturbed.smoothed_score[i])]),
        showSymbol: false,
        connectNulls: false,
        lineStyle: { width: 1.8, color: SCORE },
        itemStyle: { color: SCORE },
        markLine: { silent: true, symbol: "none", lineStyle: { color: MUTED, type: "dashed", width: 1 }, label: { show: false }, data: [{ yAxis: r.threshold }] },
        markArea: { silent: true, data: alerts },
        z: 3,
      },
      {
        type: "line",
        name: "Unchanged recording",
        data: t.map((x, i) => [x, pos(r.baseline.smoothed_score[i])]),
        showSymbol: false,
        connectNulls: false,
        lineStyle: { width: 1.2, color: FAINT },
        itemStyle: { color: FAINT },
        markArea: injected(r),
      },
    ],
  };
}

export function channelOption(r: FaultRun, narrow: boolean) {
  const c = r.channel;
  if (!c) return null;
  const t = r.t.map(instant);
  const line = (name: string, v: Array<number | null>, color: string, width: number, extra = {}) => ({ type: "line", name, data: t.map((x, i) => [x, v[i]]), showSymbol: false, connectNulls: false, lineStyle: { width, color }, itemStyle: { color }, ...extra });
  return {
    useUTC: true,
    animation: false,
    grid: { left: narrow ? 46 : 58, right: narrow ? 12 : 24, top: 22, bottom: 28 },
    xAxis: { type: "time", min: t[0], max: t[t.length - 1], ...AXIS, axisLabel: { color: MUTED, fontSize: 12, hideOverlap: true } },
    yAxis: { type: "value", scale: true, splitNumber: 3, ...Y, axisLabel: { color: MUTED, fontSize: 12 } },
    tooltip: { trigger: "axis", ...TIP, valueFormatter: (v: number | null) => (v === null ? "no data" : String(Number(v.toFixed(3)))) },
    series: [line("With the change", c.perturbed, SCORE, 1.6, { z: 3 }), line("Unchanged recording", c.baseline, FAINT, 1.2, { markArea: injected(r) })],
  };
}

const sec = (s: number) => `${Math.floor(s / 3600)}:${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}`;

function twinBase(r: TwinRunResult, narrow: boolean) {
  return {
    animation: false,
    grid: { left: narrow ? 46 : 58, right: narrow ? 12 : 24, top: 22, bottom: 28 },
    xAxis: { type: "value", min: 0, max: r.time_seconds[r.time_seconds.length - 1], interval: r.time_seconds[r.time_seconds.length - 1] > 7200 ? 3600 : 900, ...AXIS, axisLabel: { color: MUTED, fontSize: 12, formatter: sec, hideOverlap: true } },
    tooltip: { trigger: "axis", ...TIP },
  };
}

const LEAK = "#e8601c";
const line = (r: TwinRunResult, name: string, v: number[], color: string, width: number, type: "solid" | "dashed" = "solid") => ({
  type: "line",
  name,
  data: r.time_seconds.map((x, i) => [x, v[i]]),
  showSymbol: false,
  lineStyle: { width, color, type },
  itemStyle: { color },
});
const onset = (r: TwinRunResult) => (r.onset_seconds === null ? [] : [{ xAxis: r.onset_seconds, lineStyle: { color: "#f5b942", width: 2, type: "solid" }, label: { show: false } }]);
const kg = (v: number) => `${v.toFixed(4)} kg/s`;

/** Known simulated inputs: what was declared, kept apart. Legitimate demand is base plus added demand. */
export function twinInputsOption(r: TwinRunResult, narrow: boolean) {
  const i = r.inputs;
  const legit = i.base_demand_kg_s.map((b, k) => b + i.added_demand_kg_s[k]);
  return {
    ...twinBase(r, narrow),
    yAxis: { type: "value", min: 0, splitNumber: 3, ...Y, axisLabel: { color: MUTED, fontSize: 12 } },
    tooltip: {
      trigger: "axis",
      ...TIP,
      formatter: (items: Array<{ dataIndex: number }>) => {
        const k = items[0]?.dataIndex;
        if (k === undefined) return "";
        return [`<b>${sec(r.time_seconds[k])}</b>`, `Base demand ${kg(i.base_demand_kg_s[k])}`, `Added demand ${kg(i.added_demand_kg_s[k])}`, `Leakage ${kg(i.leak_kg_s[k])}`, `Total outflow ${kg(i.total_outflow_kg_s[k])}`].join("<br/>");
      },
    },
    series: [
      { ...line(r, "Legitimate demand", legit, SCORE, 1.6), markLine: { silent: true, symbol: "none", data: onset(r) } },
      line(r, "Leakage", i.leak_kg_s, LEAK, 1.8),
      line(r, "Total outflow", i.total_outflow_kg_s, MUTED, 1.2, "dashed"),
    ],
  };
}

export function twinPressureOption(r: TwinRunResult, narrow: boolean) {
  const band = (v: number, color: string, dash: string) => ({ yAxis: v, lineStyle: { color, type: dash, width: 1 }, label: { show: false } });
  return {
    ...twinBase(r, narrow),
    yAxis: { type: "value", min: r.band.safety_min - 0.05, max: r.band.safety_max + 0.05, splitNumber: 4, ...Y, axisLabel: { color: MUTED, fontSize: 12 } },
    tooltip: { trigger: "axis", ...TIP, valueFormatter: (v: number) => `${v.toFixed(3)} bar g` },
    series: [
      {
        ...line(r, "Receiver pressure", r.response.pressure_bar_g, SCORE, 1.6),
        markLine: { silent: true, symbol: "none", data: [band(r.band.lower, MUTED, "dashed"), band(r.band.upper, MUTED, "dashed"), band(r.band.safety_min, ALERT, "dotted"), band(r.band.safety_max, ALERT, "dotted"), ...onset(r)] },
      },
    ],
  };
}

export function twinFlowOption(r: TwinRunResult, narrow: boolean) {
  return {
    ...twinBase(r, narrow),
    yAxis: { type: "value", scale: true, splitNumber: 3, ...Y, axisLabel: { color: MUTED, fontSize: 12 } },
    tooltip: { trigger: "axis", ...TIP, valueFormatter: (v: number) => kg(v) },
    series: [{ ...line(r, "Compressor flow", r.response.compressor_flow_kg_s, SCORE, 1.6), markLine: { silent: true, symbol: "none", data: onset(r) } }, line(r, "Total outflow", r.inputs.total_outflow_kg_s, MUTED, 1.2, "dashed")],
  };
}

export function twinPowerOption(r: TwinRunResult, narrow: boolean) {
  return {
    ...twinBase(r, narrow),
    yAxis: { type: "value", scale: true, splitNumber: 3, ...Y, axisLabel: { color: MUTED, fontSize: 12 } },
    tooltip: { trigger: "axis", ...TIP, valueFormatter: (v: number) => `${v.toFixed(1)} kW` },
    series: [{ ...line(r, "Compressor power", r.response.power_kw, SCORE, 1.6), markLine: { silent: true, symbol: "none", data: onset(r) } }],
  };
}
