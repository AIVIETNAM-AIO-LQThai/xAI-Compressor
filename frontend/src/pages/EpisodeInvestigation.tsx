import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";

import { ErrorPanel } from "../components/ErrorPanel";
import { LoadingPanel } from "../components/LoadingPanel";
import { SourceTag } from "../components/SourceTag";
import { ExplorerError, getEpisodeEvidence, instant, type EpisodeEvidence } from "../explorer";
import type { Selection } from "../selection";
import { dateLabel, groupName } from "../syrus/labels";
import { useCases } from "../syrus/useCase";
import { useNarrow } from "../useNarrow";
import { MUTED, SCORE, num, stamp } from "./explorerCharts";

const clock = (t: string) => t.slice(11, 16);
const pct = (v: number, d = 1) => `${(v * 100).toFixed(d)}%`;
const share = (v: number) => (v > 0 && v < 0.0005 ? "<0.1%" : pct(v));
const TONES = [SCORE, "#e8601c", "#4f8fe8"];
const DASH = ["solid", "dashed", "dotted"] as const;

type Ev = EpisodeEvidence["evidence"];

/** An alert episode with no reviewed case file: what the frozen detector itself shows, computed on request. Syrus is off. */
export function EpisodeInvestigation({ selection, go }: { selection: Extract<Selection, { kind: "episode" }>; go: (s: Selection) => void }) {
  const [data, setData] = useState<EpisodeEvidence | null>(null);
  const [error, setError] = useState("");
  const { cases } = useCases();

  useEffect(() => {
    const ctl = new AbortController();
    setData(null);
    setError("");
    getEpisodeEvidence(selection.start, ctl.signal).then(setData, (e: Error) => e.name !== "AbortError" && setError(e instanceof ExplorerError ? e.code : "network"));
    return () => ctl.abort();
  }, [selection.start]);

  if (error === "episode_not_found") return <Missing start={selection.start} />;
  if (error) return <ErrorPanel error="The episode evidence could not be computed." />;
  if (!data) return <LoadingPanel label="Computing the evidence with the frozen detector…" />;

  const { episode: e, evidence: v } = data;
  const linked = data.linked_incident === null ? null : cases?.find((c) => c.number === data.linked_incident);
  const above = v.smoothed_score / v.threshold - 1;
  return (
    <div className="page-shell investigation">
      <header className="inv-header">
        <div>
          <div className="eyebrow">INVESTIGATION / ALERT EPISODE</div>
          <h1>Why did the detector alert?</h1>
          <p className="muted">
            MetroPT-3 air-production unit · {dateLabel(e.start)} {clock(e.start)} · {e.alert_bins} alert {e.alert_bins === 1 ? "bin" : "bins"} · dataset local time
          </p>
        </div>
        <div className="tags">
          <SourceTag source="MODEL INFERENCE" />
          <SourceTag source="UNREVIEWED" />
          <span className="computed-badge">Syrus off</span>
        </div>
      </header>

      <div className="metrics">
        <div className="metric">
          <span>Smoothed score</span>
          <strong>{num(v.smoothed_score)}</strong>
          <small>at {clock(v.time)}, the first alert bin</small>
        </div>
        <div className="metric">
          <span>Threshold</span>
          <strong>{num(v.threshold)}</strong>
          <small>calibration q0.995</small>
        </div>
        <div className="metric accent">
          <span>Above threshold</span>
          <strong>
            {above >= 0 ? "+" : ""}
            {(above * 100).toFixed(1)}%
          </strong>
          <small>not a fault probability</small>
        </div>
      </div>

      <Contributions v={v} />
      <Mechanics v={v} />
      <Window v={v} />

      <section className="notice warn scope-note" role="note">
        <p>
          <b>Computed from the recording with the frozen detector, on request.</b> Nobody has reviewed these figures as a case file, and Syrus has none for this episode, so it stays off here rather than borrow another incident's
          findings.
          {data.linked_incident === null
            ? " No documented incident is linked to this episode. That does not make it a false alarm: the recording has no label for this time."
            : ` This episode counts toward documented Incident ${data.linked_incident}.`}
        </p>
        <p>
          Not computed here: counterfactual repairs, candidate explanations and the tests that would tell them apart, and any manual, procedure or maintenance record (none is loaded for this asset). The ranking says where the
          detector's evidence sits, not what physically changed. Episode {e.start} to {e.end}, peak {num(e.peak_smoothed)} at {e.peak_time}.
        </p>
        <div className="scope-links">
          {linked ? (
            <button className="secondary-button" onClick={() => go({ kind: "incident", caseId: linked.caseId, mode: "retro" })}>
              Open the reviewed Incident {linked.number}
            </button>
          ) : null}
          <a className="secondary-button" href={`#explore/episode/${e.start.replace(" ", "T").slice(0, 16)}`}>
            See the recorded data
          </a>
        </div>
      </section>
    </div>
  );
}

function Missing({ start }: { start: string }) {
  return (
    <div className="page-shell">
      <h1>No alert episode starts then</h1>
      <p className="muted">No alert episode starts at {start.replace("T", " ")}.</p>
      <a className="secondary-button" href="#explore">
        Choose an episode in the Data Explorer
      </a>
    </div>
  );
}

function Contributions({ v }: { v: Ev }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h3>Detector contributions</h3>
        <span className="tag">not probabilities</span>
      </div>
      <ul className="contribs">
        {v.groups.map((g) => (
          <li key={g.group}>
            <span>{groupName(g.group)}</span>
            <div className="bar">
              <i style={{ width: `${g.share * 100}%` }} />
            </div>
            <b>{share(g.share)}</b>
            <small>calibration percentile {pct(g.calibration_percentile)}</small>
          </li>
        ))}
      </ul>
      <p className="foot">Share of the smoothed score carried by each signal group. The percentile places the group's smoothed contribution among the calibration period's values, so a group can sit high there while carrying a small share of this score.</p>
    </section>
  );
}

function Mechanics({ v }: { v: Ev }) {
  const hits = v.recent.filter((r) => r.hit).length;
  return (
    <section className="panel">
      <div className="panel-head">
        <h3>Alert mechanics</h3>
        <span className="tag">
          rule: {v.rule.hits} of last {v.rule.window} smoothed bins
        </span>
      </div>
      <ol className="bins">
        {v.recent.map((r) => (
          <li key={r.time} className={r.hit ? "hit" : ""}>
            <time>{clock(r.time)}</time>
            <span>{r.hit ? "at or above" : "below"}</span>
          </li>
        ))}
      </ol>
      <p className="foot">
        {v.recent.length < v.rule.window ? `Smoothing restarted at ${clock(v.recent[0].time)} after a gap, so only ${v.recent.length} ${v.recent.length === 1 ? "bin counts" : "bins count"} toward the rule. ` : ""}
        {hits} of the last {v.recent.length} bins were at or above the threshold when the alert fired.
        {v.carried_share !== null
          ? ` Earlier bins carried through the smoothing account for ${pct(v.carried_share, 2)} of the current smoothed score; the bin's own raw score is ${num(v.raw_score)}.`
          : ` The bin's own raw score is ${num(v.raw_score)}.`}
      </p>
    </section>
  );
}

function Window({ v }: { v: Ev }) {
  const narrow = useNarrow();
  const top = v.groups.slice(0, 3).map((g) => g.group);
  const key = top.join();
  const option = useMemo(
    () => ({
      useUTC: true,
      animation: false,
      grid: { left: narrow ? 44 : 52, right: 16, top: 16, bottom: 28 },
      xAxis: {
        type: "time",
        axisLine: { lineStyle: { color: "rgba(255,255,255,.18)" } },
        axisTick: { show: false },
        axisLabel: { color: MUTED, fontSize: 12, hideOverlap: true, formatter: (t: number) => stamp(t).slice(11) },
        splitLine: { show: false },
      },
      yAxis: { type: "value", min: 0, max: 1, splitNumber: 2, axisLabel: { color: MUTED, fontSize: 12, formatter: (x: number) => `${x * 100}%` }, splitLine: { lineStyle: { color: "rgba(255,255,255,.07)" } } },
      tooltip: { trigger: "axis", backgroundColor: "#181e24", borderColor: "rgba(255,255,255,.14)", textStyle: { color: "#f2f3f5", fontSize: 13 }, valueFormatter: (x: number) => pct(x) },
      series: key.split(",").map((g, i) => ({
        type: "line",
        name: groupName(g),
        data: v.window.bins.map((b) => [instant(b.time), b.percentiles[g]]),
        showSymbol: false,
        lineStyle: { width: 2, color: TONES[i], type: DASH[i] },
        itemStyle: { color: TONES[i] },
      })),
    }),
    [v, narrow, key],
  );
  return (
    <section className="panel">
      <div className="panel-head">
        <h3>Evidence over the preceding hour</h3>
        <span className="tag">calibration percentile</span>
      </div>
      <div className="legend" aria-hidden>
        {top.map((g, i) => (
          <span key={g}>
            <i style={{ borderTopColor: TONES[i], borderTopStyle: DASH[i] }} />
            {groupName(g)}
          </span>
        ))}
      </div>
      <ReactECharts option={option} notMerge style={{ height: narrow ? 200 : 240 }} aria-label="Calibration percentile of the three leading signal groups over the hour before the alert" />
      {!v.window.complete ? <p className="foot">The recording has fewer than 12 bins before this alert, so the window is shorter.</p> : null}
      <p className="foot">The three groups with the largest share at the alert. These rank the detector's smoothed evidence; they do not show when a fault began.</p>
    </section>
  );
}
