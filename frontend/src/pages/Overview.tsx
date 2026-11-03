import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";

import { getDetectionEvidence, getEvidenceSummary } from "../api";
import { ErrorPanel } from "../components/ErrorPanel";
import { LoadingPanel } from "../components/LoadingPanel";
import { PageHeader } from "../components/PageHeader";
import { SourceTag } from "../components/SourceTag";
import { ExplorerError, getOverview, instant, type ExplorerOverview } from "../explorer";
import type { Selection } from "../selection";
import { dateLabel } from "../syrus/labels";
import { useCases } from "../syrus/useCase";
import { useNarrow } from "../useNarrow";
import type { DetectionEvidence, EvidenceSummary } from "../types";

// Chart colours, checked for colour-vision separation and contrast on the panel surface. Text stays in text tokens.
const SCORE = "#c3cad2";
const ALERT = "#e8601c";
const INCIDENT = "#4f8fe8";
const MUTED = "#aab2bb";
const SURFACE = "#11161b";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const num = (v: number) => v.toLocaleString("en-US", { maximumFractionDigits: 0 });
const compact = (v: number) => (v >= 1e6 ? `${num(v / 1e6)}M` : v >= 1000 ? `${num(v / 1000)}k` : num(v));
const stamp = (ms: number) => new Date(ms).toISOString().slice(0, 16).replace("T", " ");
const DAY = 86_400_000;

export function Overview({ open, choose }: { open: (hash: string) => void; choose: (s: Selection) => void }) {
  const [summary, setSummary] = useState<EvidenceSummary | null>(null);
  const [detection, setDetection] = useState<DetectionEvidence | null>(null);
  const [data, setData] = useState<ExplorerOverview | null>(null);
  const [indexError, setIndexError] = useState("");
  const [error, setError] = useState("");
  const { cases } = useCases();

  useEffect(() => {
    const ctl = new AbortController();
    Promise.all([getEvidenceSummary(), getDetectionEvidence()])
      .then(([s, d]) => {
        setSummary(s);
        setDetection(d);
      })
      .catch((e: Error) => setError(e.message));
    getOverview(ctl.signal).then(setData, (e: Error) => e.name !== "AbortError" && setIndexError(e instanceof ExplorerError ? e.code : "network"));
    return () => ctl.abort();
  }, []);

  if (error) return <ErrorPanel error={error} />;
  if (!summary || !detection || !cases || (!data && !indexError)) return <LoadingPanel />;

  const m = detection.metrics;
  const saving = summary.evidence_classes.SIMULATED.optimizer.comparison.energy_saving_percent as number;
  const timing = (alertTime: string) => detection.incidents.find((i) => i.explanation_timestamp.replace("T", " ").slice(0, 16) === alertTime.slice(0, 16))?.timing;

  return (
    <div className="page-shell overview">
      <PageHeader
        eyebrow="OVERVIEW"
        title="Recorded operation and detector activity"
        description="MetroPT-3 air-production unit · February to September 2020 · historical replay, advisory only."
        badge={<span className="badge-outline">No equipment control</span>}
      />

      {data ? <Activity data={data} /> : <IndexNotice code={indexError} />}

      <section className="kpis" aria-label="Recording and detector summary">
        {data ? (
          <>
            <article>
              <span>Recorded span</span>
              <strong>{Math.round((instant(data.span[1]) - instant(data.span[0])) / (30.44 * DAY))} months</strong>
              <small>
                {dateLabel(data.span[0])} – {dateLabel(data.span[1])} · {num(data.source.raw_rows)} rows every {data.source.sampling_seconds} s from{" "}
                {data.source.channels.analog.length + data.source.channels.digital.length} channels
              </small>
            </article>
            <article>
              <span>Valid 5-minute bins</span>
              <strong>{(data.bins.share * 100).toFixed(1)}%</strong>
              <small>
                {num(data.bins.valid)} of {num(data.bins.expected)}; the rest are recording gaps
              </small>
            </article>
            <article>
              <span>Alert episodes in the test period</span>
              <strong>{data.episodes.total}</strong>
              <small>
                {data.episodes.linked} near a documented incident · {data.episodes.not_linked} not linked to one
              </small>
            </article>
          </>
        ) : null}
        <article>
          <span>Documented incidents detected</span>
          <strong>{Math.round(m.timely_incident_recall * 4)} / 4</strong>
          <small>{Math.round(m.pre_onset_incident_recall * 4)} / 4 before the documented onset · evaluation context</small>
        </article>
      </section>

      <p className="foot credit-line">
        Source: MetroPT-3 dataset, Davari, Veloso, Ribeiro &amp; Gama (2021), UCI Machine Learning Repository,{" "}
        <a href="https://doi.org/10.24432/C5VW3R" target="_blank" rel="noreferrer">
          doi:10.24432/C5VW3R
        </a>
        , CC BY 4.0. Detector scores are model inference on that recording.
      </p>

      <div className="overview-split">
        <section className="panel">
          <div className="panel-head">
            <h2>Investigate a documented incident</h2>
          </div>
          <ul className="incident-list">
            {cases.map((c) => {
              const t = timing(c.alertTime);
              return (
                <li key={c.caseId}>
                  <button onClick={() => choose({ kind: "incident", caseId: c.caseId, mode: "retro" })}>
                    <span>
                      <strong>
                        Incident {c.number} · {dateLabel(c.alertTime)}
                      </strong>
                      <small>Alert {c.alertTime.slice(11, 16)} · dataset local time</small>
                    </span>
                    {t ? <em className="tag">{t === "pre_onset" ? "Alert before onset" : "Alert after onset"}</em> : null}
                  </button>
                </li>
              );
            })}
          </ul>
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>New here?</h2>
          </div>
          <p className="muted">One incident walked through end to end: the alert, the evidence behind it, and a conversation with Syrus.</p>
          <button className="secondary-button" onClick={() => open("explore")}>
            Explore the recorded data
          </button>{" "}
          {cases[1] ? (
            <button className="primary-button" onClick={() => choose({ kind: "incident", caseId: cases[1].caseId, mode: "retro" })}>
              Guided example: Incident 2
            </button>
          ) : null}
        </section>
      </div>

      <section className="panel sim-panel">
        <div className="panel-head">
          <h2>Simulation only</h2>
          <span className="tag warn">SIMULATED</span>
        </div>
        <p>
          Nominal dispatch saving in a synthetic three-compressor scenario: <b>{saving.toFixed(3)}%</b>. Not measured on the MetroPT-3 compressor.
        </p>
        <div className="sim-links">
          <button className="secondary-button" onClick={() => open("twin")}>
            Simulation lab
          </button>
          <button className="secondary-button" onClick={() => open("proof")}>
            Evidence proof
          </button>
        </div>
      </section>
    </div>
  );
}

function IndexNotice({ code }: { code: string }) {
  return (
    <section className="panel notice warn" role="status">
      {code === "explorer_index_missing" ? (
        <p>
          The recorded-data index is not built on this server, so detector activity cannot be drawn. Build it with <code>python -m scripts.build_explorer_index</code>.
        </p>
      ) : (
        <p>Detector activity could not be loaded.</p>
      )}
    </section>
  );
}

function Activity({ data }: { data: ExplorerOverview }) {
  const narrow = useNarrow();
  const option = useMemo(() => activityOption(data, narrow), [data, narrow]);
  const a = data.activity;
  const alerts = a.alert.filter(Boolean).length;
  const gaps = a.max_smoothed.filter((v) => v === null).length;
  return (
    <section className="panel activity">
      <div className="panel-head">
        <h2>Detector activity</h2>
        <div className="tags">
          <SourceTag source="MODEL INFERENCE" />
          <span className="tag">{a.bucket_hours}-hour maximum · log scale</span>
        </div>
      </div>
      <div className="legend" aria-hidden>
        <span>
          <i style={{ borderTopColor: SCORE }} /> Smoothed score
        </span>
        <span>
          <i className="dot" style={{ background: ALERT }} /> Alert in the {a.bucket_hours} h
        </span>
        <span>
          <i style={{ borderTopColor: MUTED, borderTopStyle: "dashed" }} /> Alert threshold {num(data.threshold)}
        </span>
        <span>
          <i style={{ borderTopColor: INCIDENT }} /> Documented incident, retrospective
        </span>
      </div>
      <ReactECharts
        option={option}
        style={{ height: narrow ? 260 : 340 }}
        notMerge
        aria-label={`Detector activity from ${dateLabel(data.span[0])} to ${dateLabel(data.span[1])}: ${alerts} ${a.bucket_hours}-hour periods contain an alert, ${gaps} have no valid data. A monthly table follows.`}
      />
      <p className="foot">
        Shaded at the start: the training and calibration periods, where alerts are not evaluated. Blank stretches are recording gaps and are not joined. Incident markers come from the
        dataset documentation and were never given to the detector.
      </p>
      <MonthTable data={data} />
    </section>
  );
}

function activityOption(data: ExplorerOverview, narrow: boolean) {
  const a = data.activity;
  const step = a.bucket_hours * 3_600_000;
  const points = a.t.map((t, i) => [instant(t), a.max_smoothed[i]]);
  const alerts = a.t.flatMap((t, i) => (a.alert[i] && a.max_smoothed[i] !== null ? [[instant(t), a.max_smoothed[i]]] : []));
  const starts = data.incidents.map((i) => instant(i.start));
  const side = (k: number) => (starts[k + 1] - starts[k] < 10 * DAY ? "right" : k > 0 && starts[k] - starts[k - 1] < 10 * DAY ? "left" : "center");
  const split = (t: number) => (["train", "calibration", "test"] as const).find((k) => t >= instant(data.splits[k][0]) && t <= instant(data.splits[k][1]));

  return {
    useUTC: true,
    animation: false,
    grid: { left: narrow ? 44 : 56, right: narrow ? 12 : 24, top: 28, bottom: 28 },
    xAxis: {
      type: "time",
      min: instant(data.span[0]),
      max: instant(data.span[1]),
      axisLine: { lineStyle: { color: "rgba(255,255,255,.18)" } },
      axisTick: { show: false },
      axisLabel: { color: MUTED, fontSize: 12, formatter: (v: number) => MONTHS[new Date(v).getUTCMonth()], hideOverlap: true },
      splitLine: { show: false },
    },
    yAxis: {
      type: "log",
      logBase: 10,
      axisLabel: { color: MUTED, fontSize: 12, formatter: compact },
      splitLine: { lineStyle: { color: "rgba(255,255,255,.07)" } },
    },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "line", lineStyle: { color: "rgba(255,255,255,.35)" } },
      backgroundColor: "#181e24",
      borderColor: "rgba(255,255,255,.14)",
      textStyle: { color: "#f2f3f5", fontSize: 13 },
      formatter: (items: Array<{ dataIndex: number; seriesIndex: number }>) => {
        const i = items.find((p) => p.seriesIndex === 0)?.dataIndex;
        if (i === undefined) return "";
        const t = instant(a.t[i]);
        const s = split(t);
        const head = `<b>${stamp(t)} – ${stamp(t + step).slice(11)}</b>`;
        if (a.max_smoothed[i] === null) return `${head}<br/>No valid data (recording gap)`;
        return [
          head,
          `Highest smoothed score ${num(a.max_smoothed[i]!)}`,
          `${a.bins[i]} of ${a.bucket_hours * 12} bins valid`,
          s === "test" ? (a.alert[i] ? "Contains an alert" : "No alert") : `${s === "train" ? "Training" : "Calibration"} period · alerts not evaluated`,
        ].join("<br/>");
      },
    },
    series: [
      {
        type: "line",
        name: "Smoothed score",
        data: points,
        showSymbol: false,
        connectNulls: false,
        lineStyle: { width: narrow ? 1.25 : 1.5, color: SCORE },
        itemStyle: { color: SCORE },
        markLine: {
          silent: true,
          symbol: "none",
          lineStyle: { color: MUTED, type: "dashed", width: 1 },
          label: { show: false },
          data: [{ yAxis: data.threshold }],
        },
        markArea: {
          silent: true,
          itemStyle: { color: "rgba(255,255,255,.05)" },
          label: { color: MUTED, fontSize: 11, position: "insideTopLeft" },
          data: [
            [{ name: narrow ? "" : "Train", xAxis: instant(data.splits.train[0]) }, { xAxis: instant(data.splits.train[1]) }],
            [{ name: "", xAxis: instant(data.splits.calibration[0]), itemStyle: { color: "rgba(255,255,255,.09)" } }, { xAxis: instant(data.splits.calibration[1]) }],
          ],
        },
      },
      {
        type: "scatter",
        name: "Alert",
        data: alerts,
        symbolSize: narrow ? 6 : 8,
        itemStyle: { color: ALERT, borderColor: SURFACE, borderWidth: 1.5 },
        z: 3,
        tooltip: { show: false },
      },
      {
        type: "line",
        name: "Documented incident",
        data: [],
        itemStyle: { color: INCIDENT },
        markLine: {
          symbol: "none",
          lineStyle: { color: INCIDENT, type: "solid", width: 2 },
          tooltip: { show: false },
          data: data.incidents.map((inc, k) => ({
            xAxis: starts[k],
            label: { formatter: String(inc.id), color: "#f2f3f5", fontSize: 12, fontWeight: 600, position: "end", align: side(k), distance: 4 },
          })),
        },
      },
    ],
  };
}

function MonthTable({ data }: { data: ExplorerOverview }) {
  const a = data.activity;
  const rows = new Map<string, { bins: number; max: number | null; alerts: number }>();
  a.t.forEach((t, i) => {
    const r = rows.get(t.slice(0, 7)) ?? { bins: 0, max: null, alerts: 0 };
    const v = a.max_smoothed[i];
    r.bins += a.bins[i];
    if (v !== null) r.max = Math.max(r.max ?? 0, v);
    if (a.alert[i]) r.alerts += 1;
    rows.set(t.slice(0, 7), r);
  });
  return (
    <details className="chart-table">
      <summary>Monthly table</summary>
      <table>
        <thead>
          <tr>
            <th>Month</th>
            <th>Valid bins</th>
            <th>Highest smoothed score</th>
            <th>{a.bucket_hours}-hour periods with an alert</th>
          </tr>
        </thead>
        <tbody>
          {[...rows].map(([k, r]) => (
            <tr key={k}>
              <td>
                {MONTHS[Number(k.slice(5)) - 1]} {k.slice(0, 4)}
              </td>
              <td>{num(r.bins)}</td>
              <td>{r.max === null ? "—" : num(r.max)}</td>
              <td>{k < data.splits.test[0].slice(0, 7) ? "not evaluated" : r.alerts}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}
