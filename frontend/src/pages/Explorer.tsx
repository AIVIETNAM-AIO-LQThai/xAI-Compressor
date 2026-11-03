import { useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import * as echarts from "echarts";

import { ErrorPanel } from "../components/ErrorPanel";
import { LoadingPanel } from "../components/LoadingPanel";
import { SourceTag } from "../components/SourceTag";
import {
  ExplorerError,
  MAX_RANGE_DAYS,
  MAX_RAW_HOURS,
  getBin,
  getEpisodes,
  getOverview,
  getRange,
  getRaw,
  instant,
  type BinData,
  type Episode,
  type ExplorerOverview,
  type RangeData,
  type RawData,
} from "../explorer";
import type { Selection } from "../selection";
import { dateLabel } from "../syrus/labels";
import { useCases } from "../syrus/useCase";
import { useNarrow } from "../useNarrow";
import { channelRows, detectorOption, num, stackOption, stamp } from "./explorerCharts";

const HOUR = 3_600_000;
const DAY = 24 * HOUR;
const GROUP = "explorer";
const stampOf = (ms: number) => new Date(ms).toISOString().slice(0, 16);
const show = (t: string) => t.replace("T", " ").slice(0, 16);
const when = (t: string) => `${dateLabel(t)} ${show(t).slice(11)}`;

interface Win {
  start: string; // YYYY-MM-DDTHH:MM, dataset local time
  end: string;
}

type Interval = Extract<Selection, { kind: "interval" }>;
type Pick = (s: Selection) => void;

/** A window around an episode: a quarter of its length, at least three hours, on each side. */
export function around(e: Episode): Win {
  const a = instant(e.start);
  const b = instant(e.end) + 300_000;
  const pad = Math.max(3 * HOUR, (b - a) / 4);
  return { start: stampOf(a - pad), end: stampOf(Math.min(b + pad, a + MAX_RANGE_DAYS * DAY)) };
}

/** The recorded data for a chosen episode or interval. Nothing is shown until the user chooses one. */
export function Explorer({ selection, go }: { selection: Exclude<Selection, { kind: "incident" }> | null; go: Pick }) {
  const [overview, setOverview] = useState<ExplorerOverview | null>(null);
  const [episodes, setEpisodes] = useState<Episode[] | null>(null);
  const [failure, setFailure] = useState("");

  useEffect(() => {
    const ctl = new AbortController();
    Promise.all([getOverview(ctl.signal), getEpisodes(ctl.signal)]).then(
      ([o, e]) => {
        setOverview(o);
        setEpisodes(e);
      },
      (e: Error) => e.name !== "AbortError" && setFailure(e instanceof ExplorerError ? e.code : "network"),
    );
    return () => ctl.abort();
  }, []);

  if (failure === "explorer_index_missing") {
    return (
      <div className="page-shell">
        <ErrorPanel error="The recorded-data index is not built on this server. Build it with: python -m scripts.build_explorer_index" />
      </div>
    );
  }
  if (failure) return <ErrorPanel error="The recorded data could not be loaded." />;
  if (!overview || !episodes) return <LoadingPanel label="Loading the recording…" />;

  const episode = selection?.kind === "episode" ? episodes.find((e) => e.start.replace(" ", "T").slice(0, 16) === selection.start) : undefined;
  const win: Win | null = selection?.kind === "interval" ? selection : episode ? around(episode) : null;
  return (
    <div className="page-shell explorer">
      <header className="inv-header">
        <div>
          <div className="eyebrow">DATA EXPLORER</div>
          <h1>{win ? (episode ? "Alert episode" : "Recorded interval") : "Choose what to look at"}</h1>
          <p className="muted">
            {win
              ? `${when(win.start)} – ${when(win.end)} · MetroPT-3, dataset local time`
              : "The whole recording is available: pick an alert episode, a documented incident window, or any interval of up to 14 days."}
          </p>
        </div>
      </header>
      {selection?.kind === "episode" && !episode ? <p className="notice warn">No alert episode starts at {show(selection.start)}.</p> : null}
      <Chooser overview={overview} episodes={episodes} win={win} selected={episode} go={go} />
      {win ? <View key={`${win.start}|${win.end}`} win={win} overview={overview} episodes={episodes} episode={episode} go={go} /> : null}
    </div>
  );
}

function Chooser({ overview, episodes, win, selected, go }: { overview: ExplorerOverview; episodes: Episode[]; win: Win | null; selected?: Episode; go: Pick }) {
  const [open, setOpen] = useState(win === null);
  const [start, setStart] = useState(win?.start ?? "");
  const [end, setEnd] = useState(win?.end ?? "");
  const [filter, setFilter] = useState<"all" | "none" | "linked">("all");
  const [order, setOrder] = useState<"time" | "peak" | "length">("time");
  const [error, setError] = useState("");
  const lo = overview.span[0].replace(" ", "T").slice(0, 16);
  const hi = overview.span[1].replace(" ", "T").slice(0, 16);
  // A new choice closes the list and the fields follow the window on screen.
  useEffect(() => {
    if (!win) return;
    setOpen(false);
    setStart(win.start);
    setEnd(win.end);
  }, [win?.start, win?.end]); // eslint-disable-line react-hooks/exhaustive-deps

  const apply = (a: string, b: string) => {
    const span = instant(b) - instant(a);
    if (!a || !b || !(span > 0)) return setError("Choose a start before the end.");
    if (span > MAX_RANGE_DAYS * DAY) return setError(`The window can be at most ${MAX_RANGE_DAYS} days.`);
    if (a < lo || b > hi) return setError(`The recording runs from ${show(lo)} to ${show(hi)}.`);
    setError("");
    go({ kind: "interval", start: a, end: b });
  };
  const length = (ms: number) => apply(start, stampOf(Math.min(instant(start) + ms, instant(hi))));
  const shown = useMemo(() => {
    const list = episodes.filter((e) => (filter === "all" ? true : filter === "linked" ? e.linked_incident !== null : e.linked_incident === null));
    const key = { time: (e: Episode) => e.start, peak: (e: Episode) => -e.peak_smoothed, length: (e: Episode) => -e.alert_bins };
    return [...list].sort((a, b) => (key[order](a) < key[order](b) ? -1 : 1));
  }, [episodes, filter, order]);

  return (
    <section className="panel chooser">
      <div className="panel-head">
        <h2>{win ? "Window" : "Choose"}</h2>
        {win ? (
          <button className="link-button" aria-expanded={open} onClick={() => setOpen(!open)}>
            {open ? "Hide choices" : "Change"}
          </button>
        ) : null}
      </div>
      {open || !win ? (
        <>
          <div className="pick-row">
            <label>
              Start
              <input type="datetime-local" value={start} min={lo} max={hi} onChange={(e) => setStart(e.target.value)} />
            </label>
            <label>
              End
              <input type="datetime-local" value={end} min={lo} max={hi} onChange={(e) => setEnd(e.target.value)} />
            </label>
            <button className="primary-button" onClick={() => apply(start, end)}>
              Show interval
            </button>
          </div>
          <div className="presets" role="group" aria-label="Window length from the start">
            {[
              ["6 hours", 6 * HOUR],
              ["1 day", DAY],
              ["3 days", 3 * DAY],
              ["7 days", 7 * DAY],
              ["14 days", 14 * DAY],
            ].map(([label, ms]) => (
              <button key={label} className="secondary-button" onClick={() => length(ms as number)}>
                {label}
              </button>
            ))}
          </div>
          {error ? (
            <p className="notice error" role="alert">
              {error}
            </p>
          ) : null}

          <h3>Documented incident windows</h3>
          <div className="presets">
            {overview.incidents.map((i) => (
              <button key={i.id} className="secondary-button" onClick={() => apply(stampOf(instant(i.start) - 3 * HOUR), stampOf(Math.min(instant(i.end) + 3 * HOUR, instant(i.start) + MAX_RANGE_DAYS * DAY)))}>
                Incident {i.id} · {dateLabel(i.start)}
              </button>
            ))}
          </div>
          <p className="foot">Incident windows are retrospective labels from the dataset documentation. A time with no label is not known to be healthy.</p>

          <h3>Alert episodes in the test period ({shown.length} of {episodes.length})</h3>
          <div className="episode-tools">
            <div className="segmented compact" role="group" aria-label="Filter episodes">
              {(["all", "none", "linked"] as const).map((f) => (
                <button key={f} className={filter === f ? "active" : ""} aria-pressed={filter === f} onClick={() => setFilter(f)}>
                  {f === "all" ? "All" : f === "none" ? "Not linked" : "Linked"}
                </button>
              ))}
            </div>
            <label>
              Order
              <select value={order} onChange={(e) => setOrder(e.target.value as typeof order)}>
                <option value="time">Time</option>
                <option value="peak">Peak score</option>
                <option value="length">Alert bins</option>
              </select>
            </label>
          </div>
          <ul className="episode-list">
            {shown.map((e) => (
              <li key={e.start}>
                <button className={selected?.start === e.start ? "active" : ""} aria-pressed={selected?.start === e.start} onClick={() => go({ kind: "episode", start: e.start.replace(" ", "T").slice(0, 16) })}>
                  <span>
                    <strong>{when(e.start)}</strong>
                    <small>
                      {e.alert_bins} alert {e.alert_bins === 1 ? "bin" : "bins"} · peak {num(e.peak_smoothed)}
                    </small>
                  </span>
                  <em className="tag">{e.linked_incident === null ? "Not linked" : `Incident ${e.linked_incident}`}</em>
                </button>
              </li>
            ))}
          </ul>
        </>
      ) : null}
    </section>
  );
}

function View({ win, overview, episodes, episode, go }: { win: Win; overview: ExplorerOverview; episodes: Episode[]; episode?: Episode; go: Pick }) {
  const narrow = useNarrow();
  const { cases } = useCases();
  const [data, setData] = useState<RangeData | null>(null);
  const [error, setError] = useState("");
  const [bin, setBin] = useState<{ t: string; detail: BinData | null; gap: boolean } | null>(null);
  const span = instant(win.end) - instant(win.start);
  const rawAllowed = span <= MAX_RAW_HOURS * HOUR;

  useEffect(() => {
    const ctl = new AbortController();
    getRange(win.start, win.end, ctl.signal).then(setData, (e: Error) => e.name !== "AbortError" && setError(e instanceof ExplorerError ? e.code : "network"));
    return () => ctl.abort();
  }, [win.start, win.end]);

  const option = useMemo(() => (data ? detectorOption(data, overview, narrow) : null), [data, overview, narrow]);
  const inspect = (t: number) => {
    if (!data) return;
    const i = data.t.findIndex((x) => instant(x) === t);
    if (i < 0) return;
    const at = data.t[i].replace(" ", "T").slice(0, 16);
    if (data.smoothed_score[i] === null) return setBin({ t: at, detail: null, gap: true });
    setBin({ t: at, detail: null, gap: false });
    getBin(at).then(
      (detail) => setBin((b) => (b?.t === at ? { t: at, detail, gap: false } : b)),
      () => setBin((b) => (b?.t === at ? { t: at, detail: null, gap: true } : b)),
    );
  };
  const ready = (chart: echarts.ECharts) => {
    chart.group = GROUP;
    echarts.connect(GROUP);
    const zr = chart.getZr();
    zr.off("click");
    zr.on("click", (e: { offsetX: number; offsetY: number }) => {
      const p = [e.offsetX, e.offsetY];
      if (!chart.containPixel("grid", p)) return;
      const [ms] = chart.convertFromPixel({ seriesIndex: 0 }, p) as number[];
      inspect(Math.floor(ms / 300_000) * 300_000);
    });
  };

  if (error) return <p className="notice error">{error === "range_invalid" ? "That window is not valid." : "The recorded data for this window could not be loaded."}</p>;
  if (!data || !option) return <LoadingPanel label="Loading the window…" />;

  const valid = data.smoothed_score.filter((v) => v !== null).length;
  const expected = Math.floor(span / 300_000) + 1;
  const test = data.split.filter((s) => s === "test").length;
  const alertBins = data.alert.filter(Boolean).length;
  const training = data.split.some((s) => s === "train" || s === "calibration");
  const inside = episodes.filter((e) => e.start !== episode?.start && instant(e.end) >= instant(win.start) && instant(e.start) <= instant(win.end));
  const incidents = overview.incidents.filter((i) => instant(i.end) >= instant(win.start) && instant(i.start) <= instant(win.end));
  const caseOf = (n: number) => cases?.find((c) => c.number === n);
  const move = (by: number) => {
    const d = Math.round(span * by);
    const a = instant(win.start) + d;
    const b = instant(win.end) + d;
    if (a < instant(overview.span[0]) || b > instant(overview.span[1])) return;
    go({ kind: "interval", start: stampOf(a), end: stampOf(b) });
  };

  return (
    <>
      <section className="panel">
        <div className="panel-head">
          <h2>Detector score</h2>
          <SourceTag source="MODEL INFERENCE" />
          <div className="nav-buttons">
            <button className="secondary-button" onClick={() => move(-0.5)} aria-label="Earlier by half a window">
              ← Earlier
            </button>
            <button className="secondary-button" onClick={() => move(0.5)} aria-label="Later by half a window">
              Later →
            </button>
          </div>
        </div>
        <div className="legend" aria-hidden>
          <span>
            <i style={{ borderTopColor: "#c3cad2" }} /> Smoothed score
          </span>
          <span>
            <i style={{ borderTopColor: "rgba(195,202,210,.38)" }} /> Raw score
          </span>
          <span>
            <i style={{ borderTopColor: "#aab2bb", borderTopStyle: "dashed" }} /> Threshold {num(data.threshold)}
          </span>
          <span>
            <i className="dot" style={{ background: "rgba(232,96,28,.6)", borderRadius: 2 }} /> Alert
          </span>
          <span>
            <i className="dot" style={{ background: "rgba(79,143,232,.6)", borderRadius: 2 }} /> Documented incident, retrospective
          </span>
          <span>▵ Smoothing restart after a gap</span>
        </div>
        <ReactECharts
          option={option}
          style={{ height: narrow ? 260 : 320 }}
          notMerge
          onChartReady={ready}
          aria-label={`Detector score from ${show(win.start)} to ${show(win.end)}: ${valid} valid bins of ${expected}, ${alertBins} alert bins.`}
        />
        <p className="foot">
          {num(valid)} of {num(expected)} five-minute bins have valid data; blank stretches are recording gaps and are not joined. {test > 0 ? `${num(alertBins)} of ${num(test)} test-period bins are in an alert. ` : ""}
          Click the chart to inspect a bin.
        </p>
        {training ? <p className="foot">Training and calibration periods are shaded: scores there are in-sample or set the threshold, so no alerts are evaluated.</p> : null}
        {bin ? <BinPanel bin={bin} data={data} close={() => setBin(null)} center={(t) => go({ kind: "interval", start: stampOf(instant(t) - 3 * HOUR), end: stampOf(instant(t) + 3 * HOUR) })} /> : null}
      </section>

      {episode ? <EpisodeNote e={episode} caseId={episode.linked_incident ? caseOf(episode.linked_incident)?.caseId : undefined} go={go} /> : null}

      <Channels win={win} data={data} rawAllowed={rawAllowed} narrow={narrow} analog={overview.source.channels.analog} digital={overview.source.channels.digital} />

      {incidents.length || inside.length ? (
        <section className="panel">
          <div className="panel-head">
            <h2>In this window</h2>
          </div>
          {incidents.map((i) => (
            <p key={i.id} className="inside-line">
              <b>Incident {i.id}</b> (retrospective label, {when(i.start.replace(" ", "T"))} – {when(i.end.replace(" ", "T"))}).{" "}
              {caseOf(i.id) ? (
                <button className="link-button" onClick={() => go({ kind: "incident", caseId: caseOf(i.id)!.caseId, mode: "retro" })}>
                  Investigate Incident {i.id}
                </button>
              ) : null}
            </p>
          ))}
          {inside.map((e) => (
            <p key={e.start} className="inside-line">
              Alert episode {when(e.start)}: {e.alert_bins} alert {e.alert_bins === 1 ? "bin" : "bins"}, peak {num(e.peak_smoothed)} ·{" "}
              {e.linked_incident === null ? "not linked to a documented incident" : `linked to Incident ${e.linked_incident}`}.{" "}
              <button className="link-button" onClick={() => go({ kind: "episode", start: e.start.replace(" ", "T").slice(0, 16) })}>
                Open episode
              </button>
            </p>
          ))}
        </section>
      ) : null}
    </>
  );
}

function EpisodeNote({ e, caseId, go }: { e: Episode; caseId?: string; go: Pick }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Episode</h2>
        <span className="tag">{e.linked_incident === null ? "Not linked" : `Incident ${e.linked_incident}`}</span>
      </div>
      <dl className="insp-values">
        <div>
          <dt>Alert</dt>
          <dd>
            {when(e.start)} – {when(e.end)} · {e.alert_bins} alert {e.alert_bins === 1 ? "bin" : "bins"}
          </dd>
        </div>
        <div>
          <dt>Peak</dt>
          <dd>
            {num(e.peak_smoothed)} at {when(e.peak_time)}
          </dd>
        </div>
      </dl>
      <p className="foot">
        <a className="link-button" href={`#investigate/episode/${e.start.replace(" ", "T").slice(0, 16)}`}>
          Investigate this episode
        </a>
        : the detector's contributions and alert mechanics, computed from the recording and not reviewed.
      </p>
      {e.linked_incident === null ? (
        <p className="foot">
          No documented incident overlaps this episode under the evaluation's rule (from 24 hours before an incident's start to its end). That is not a confirmed false alarm: the recording carries no label for this
          time. Syrus has no reviewed case file for an episode like this and stays off.
        </p>
      ) : (
        <p className="foot">
          This episode counts toward documented Incident {e.linked_incident} under the evaluation's rule.{" "}
          {caseId ? (
            <button className="link-button" onClick={() => go({ kind: "incident", caseId, mode: "retro" })}>
              Investigate Incident {e.linked_incident}
            </button>
          ) : null}
        </p>
      )}
    </section>
  );
}

function BinPanel({ bin, data, close, center }: { bin: { t: string; detail: BinData | null; gap: boolean }; data: RangeData; close: () => void; center: (t: string) => void }) {
  const d = bin.detail;
  const i = data.t.findIndex((x) => x.replace(" ", "T").slice(0, 16) === bin.t);
  const state = !d ? "" : d.split !== "test" ? "Alerts are not evaluated in this period" : d.alert ? "In an alert" : d.threshold_hit ? "At or above the threshold, not yet an alert" : "Below the threshold";
  return (
    <div className="bin-panel" role="status">
      <div className="panel-head">
        <h3>Bin {show(bin.t)}</h3>
        <button className="link-button" onClick={close}>
          Close
        </button>
      </div>
      {bin.gap ? (
        <p className="muted">No valid data in this bin: the recording has a gap here, and nothing is filled in.</p>
      ) : !d ? (
        <p className="muted">Loading the bin…</p>
      ) : (
        <>
          <dl className="insp-values">
            <div>
              <dt>Smoothed score</dt>
              <dd>
                {num(d.smoothed_score)} against a threshold of {num(d.threshold)} · {state}
              </dd>
            </div>
            <div>
              <dt>Raw score</dt>
              <dd>{num(d.raw_score)}</dd>
            </div>
            <div>
              <dt>Data in the bin</dt>
              <dd>
                {d.samples} samples ({Math.round(d.coverage * 100)}% of the bin){d.restart ? " · smoothing restarted here after a gap" : ""}
              </dd>
            </div>
            {i >= 0
              ? Object.entries(data.channels)
                  .filter(([, v]) => "mean" in v)
                  .map(([c, v]) => (
                    <div key={c}>
                      <dt>{c}</dt>
                      <dd>
                        mean {fmt(v.mean[i])} · min {fmt(v.min[i])} · max {fmt(v.max[i])}
                      </dd>
                    </div>
                  ))
              : null}
          </dl>
          <button className="secondary-button" onClick={() => center(bin.t)}>
            Zoom to 6 hours around this bin
          </button>
        </>
      )}
    </div>
  );
}

const fmt = (v: number | null | undefined) => (v === null || v === undefined ? "—" : String(Number(v.toFixed(3))));

function Channels({ win, data, rawAllowed, narrow, analog, digital }: { win: Win; data: RangeData; rawAllowed: boolean; narrow: boolean; analog: string[]; digital: string[] }) {
  const [raw, setRaw] = useState<RawData | null>(null);
  const [mode, setMode] = useState<"summary" | "raw">("summary");
  const [state, setState] = useState<"idle" | "loading" | "failed">("idle");
  const asked = useRef("");

  useEffect(() => {
    if (mode !== "raw" || !rawAllowed || asked.current === `${win.start}|${win.end}`) return;
    asked.current = `${win.start}|${win.end}`;
    setState("loading");
    getRaw(win.start, win.end).then(
      (r) => {
        setRaw(r);
        setState("idle");
      },
      () => {
        asked.current = "";
        setState("failed");
      },
    );
  }, [mode, rawAllowed, win.start, win.end]);

  const useRaw = mode === "raw" && rawAllowed && raw !== null;
  const src: RangeData | RawData = useRaw ? raw! : data;
  const rowH = narrow ? 76 : 88;
  const a = useMemo(() => stackOption(src.t, channelRows(analog, false, src), rowH, narrow), [src, analog, narrow, rowH]);
  const d = useMemo(() => stackOption(src.t, channelRows(digital, true, src), 62, narrow), [src, digital, narrow]);
  const ready = (chart: echarts.ECharts) => {
    chart.group = GROUP;
    echarts.connect(GROUP);
  };

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Recorded channels</h2>
        <SourceTag source="RECORDED" />
        <div className="segmented compact" role="group" aria-label="Detail">
          <button className={!useRaw ? "active" : ""} aria-pressed={!useRaw} onClick={() => setMode("summary")}>
            5-minute summary
          </button>
          <button className={useRaw ? "active" : ""} aria-pressed={useRaw} disabled={!rawAllowed} onClick={() => setMode("raw")} title={rawAllowed ? undefined : `Raw 10-second data needs a window of ${MAX_RAW_HOURS} hours or less`}>
            Raw 10 s
          </button>
        </div>
      </div>
      <p className="foot top-foot">
        {useRaw ? "Every recorded 10-second sample." : "Mean of each five-minute bin, with its minimum and maximum as faint lines."} Values as recorded; units not loaded. Gaps stay blank.
        {!rawAllowed ? ` Raw 10-second data needs a window of ${MAX_RAW_HOURS} hours or less.` : ""}
        {state === "loading" ? " Loading raw data…" : state === "failed" ? " Raw data could not be loaded." : ""}
      </p>
      <h3>Analog channels</h3>
      <ReactECharts option={a} notMerge style={{ height: analog.length * (rowH + 10) + 30 }} onChartReady={ready} aria-label="Analog channels over the window" />
      <h3>Digital channels {useRaw ? "(0 or 1)" : "(share of the bin active)"}</h3>
      <ReactECharts option={d} notMerge style={{ height: digital.length * 72 + 30 }} onChartReady={ready} aria-label="Digital channels over the window" />
    </section>
  );
}
