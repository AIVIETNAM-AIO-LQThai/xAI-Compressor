import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";

import { ErrorPanel } from "../components/ErrorPanel";
import { LoadingPanel } from "../components/LoadingPanel";
import { LabError, getBackgrounds, runFault, runTwin, type FaultKind, type FaultRequest, type FaultRun, type TwinRequest, type TwinRunResult } from "../lab";
import { dateLabel } from "../syrus/labels";
import { useNarrow } from "../useNarrow";
import { num } from "./explorerCharts";
import { channelOption, flowOption, pressureOption, responseOption } from "./labCharts";
import { responseText, severityText } from "./labText";

const ANALOG = ["tp2", "tp3", "h1", "dv_pressure", "reservoirs", "oil_temperature", "motor_current"];
const DIGITAL = ["comp", "dv_electric", "towers", "mpg", "lps", "pressure_switch", "oil_level", "caudal_impulses"];
const FAULTS: Array<[FaultKind, string, string]> = [
  ["offset", "Offset (step)", "A constant is added to one analog channel from the onset."],
  ["drift", "Drift (ramp)", "One analog channel ramps linearly from the onset to the full size at the end of the change."],
  ["flatline", "Flatline", "One analog channel is held at its last value before the onset."],
  ["noise", "Added noise", "Seeded Gaussian noise is added to one analog channel."],
  ["dropout", "Dropout", "A share of all samples is removed, so bins lose coverage and leave the detector's view."],
  ["stuck", "Digital stuck-at", "One digital signal is held at 0 or 1."],
];

type Tab = "fault" | "twin";

/** Declared, seeded changes, always labelled simulated. The detector scores recorded data with a change applied; the generic twin is not scored. */
export function ScenarioLab() {
  const [tab, setTab] = useState<Tab>("fault");
  return (
    <div className="page-shell lab">
      <header className="inv-header">
        <div>
          <div className="eyebrow">SCENARIO LAB</div>
          <h1>What would the detector see?</h1>
          <p className="muted">Declared changes on a recorded background, or a generic compressor plant. Nothing here is a measurement of the MetroPT-3 machine.</p>
        </div>
        <span className="tag warn">SIMULATED</span>
      </header>
      <div className="segmented" role="group" aria-label="Model">
        <button className={tab === "fault" ? "active" : ""} aria-pressed={tab === "fault"} onClick={() => setTab("fault")}>
          <span>Sensor fault on recorded data</span>
          <small>Scored by the frozen detector</small>
        </button>
        <button className={tab === "twin" ? "active" : ""} aria-pressed={tab === "twin"} onClick={() => setTab("twin")}>
          <span>Generic receiver twin</span>
          <small>Detector not applicable</small>
        </button>
      </div>
      {tab === "fault" ? <FaultLab /> : <TwinLab />}
    </div>
  );
}

function Unavailable({ code }: { code: string }) {
  return code === "not_served" ? (
    <p className="notice warn" role="status">
      The scenario lab is a simulation, so it is not served by the public build.
    </p>
  ) : code === "explorer_index_missing" ? (
    <p className="notice warn" role="status">
      The recorded-data index is not built on this server. Build it with <code>python -m scripts.build_explorer_index</code>.
    </p>
  ) : (
    <p className="notice error" role="alert">
      {code === "window_invalid" ? "That window is outside the test period, or has no recorded hours before it." : code === "background_incomplete" ? "The recording has too little data in that window." : "The scenario could not be run."}
    </p>
  );
}

function Field({ label, children, hint }: { label: string; children: React.ReactNode; hint?: string }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint ? <small>{hint}</small> : null}
    </label>
  );
}

function FaultLab() {
  const narrow = useNarrow();
  const [backgrounds, setBackgrounds] = useState<Array<{ start: string; hours: number }> | null>(null);
  const [form, setForm] = useState<FaultRequest>({ start: "", hours: 12, fault: "offset", channel: "tp3", onset_hours: 4, duration_hours: null, severity: 6, stuck_value: 0, seed: 0 });
  const [run, setRun] = useState<FaultRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState("");

  useEffect(() => {
    const ctl = new AbortController();
    getBackgrounds(form.hours, ctl.signal).then(
      (b) => {
        setBackgrounds(b);
        setForm((f) => (b.some((x) => x.start === f.start) ? f : { ...f, start: b[0]?.start ?? "" }));
      },
      (e: Error) => e.name !== "AbortError" && setProblem(e instanceof LabError ? e.code : "network"),
    );
    return () => ctl.abort();
  }, [form.hours]);

  const set = <K extends keyof FaultRequest>(k: K, v: FaultRequest[K]) => setForm((f) => ({ ...f, [k]: v }));
  const kind = form.fault;
  const digital = kind === "stuck";
  const channels = digital ? DIGITAL : ANALOG;
  const pick = (f: FaultKind) => setForm((x) => ({ ...x, fault: f, channel: f === "dropout" ? null : f === "stuck" ? DIGITAL[0] : ANALOG.includes(x.channel ?? "") ? x.channel : "tp3", severity: f === "dropout" ? 0.5 : x.fault === "dropout" ? 6 : x.severity }));

  const go = () => {
    setBusy(true);
    setProblem("");
    runFault(form).then(setRun, (e: Error) => setProblem(e instanceof LabError ? e.code : "network")).finally(() => setBusy(false));
  };

  if (problem === "not_served" || problem === "explorer_index_missing") return <Unavailable code={problem} />;
  if (!backgrounds) return problem ? <ErrorPanel error="The scenario lab could not be reached." /> : <LoadingPanel label="Finding recorded backgrounds…" />;

  return (
    <>
      <section className="panel">
        <div className="panel-head">
          <h2>Declare a change</h2>
        </div>
        <p className="foot top-foot">{FAULTS.find((f) => f[0] === kind)?.[2]} The same recorded window is also scored without the change.</p>
        <div className="lab-form">
          <Field label="Recorded background" hint="Quiet stretches of the test period, with no threshold crossing.">
            <select value={form.start} onChange={(e) => set("start", e.target.value)}>
              {backgrounds.map((b) => (
                <option key={b.start} value={b.start}>
                  {dateLabel(b.start)} {b.start.slice(11)} · {b.hours} h
                </option>
              ))}
            </select>
          </Field>
          <Field label="Window length">
            <select value={form.hours} onChange={(e) => setForm((f) => ({ ...f, hours: Number(e.target.value), onset_hours: Math.min(f.onset_hours, Number(e.target.value) - 1) }))}>
              {[6, 12, 24].map((h) => (
                <option key={h} value={h}>
                  {h} hours
                </option>
              ))}
            </select>
          </Field>
          <Field label="Change">
            <select value={kind} onChange={(e) => pick(e.target.value as FaultKind)}>
              {FAULTS.map(([id, label]) => (
                <option key={id} value={id}>
                  {label}
                </option>
              ))}
            </select>
          </Field>
          {kind !== "dropout" ? (
            <Field label="Channel">
              <select value={form.channel ?? ""} onChange={(e) => set("channel", e.target.value)}>
                {channels.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </Field>
          ) : null}
          <Field label="Begins after (hours)">
            <input type="number" min={1} max={form.hours - 1} step={0.5} value={form.onset_hours} onChange={(e) => set("onset_hours", Number(e.target.value))} />
          </Field>
          <Field label="Lasts (hours)" hint="Empty: to the end of the window.">
            <input type="number" min={0.5} max={24} step={0.5} value={form.duration_hours ?? ""} onChange={(e) => set("duration_hours", e.target.value === "" ? null : Number(e.target.value))} />
          </Field>
          {kind === "offset" || kind === "drift" || kind === "noise" ? (
            <Field label="Size (× training standard deviation)" hint="Channel units are not loaded, so size is relative to the channel's spread in the training period.">
              <input type="number" min={0} max={20} step={0.5} value={form.severity} onChange={(e) => set("severity", Number(e.target.value))} />
            </Field>
          ) : null}
          {kind === "dropout" ? (
            <Field label="Share of samples removed">
              <input type="number" min={0} max={1} step={0.1} value={form.severity} onChange={(e) => set("severity", Number(e.target.value))} />
            </Field>
          ) : null}
          {kind === "stuck" ? (
            <Field label="Held at">
              <select value={form.stuck_value} onChange={(e) => set("stuck_value", Number(e.target.value) as 0 | 1)}>
                <option value={0}>0</option>
                <option value={1}>1</option>
              </select>
            </Field>
          ) : null}
          {kind === "noise" || kind === "dropout" ? (
            <Field label="Seed" hint="The same seed gives the same run.">
              <input type="number" min={0} step={1} value={form.seed} onChange={(e) => set("seed", Math.max(0, Math.floor(Number(e.target.value))))} />
            </Field>
          ) : null}
        </div>
        <button className="primary-button run-button" onClick={go} disabled={busy || !form.start}>
          {busy ? "Running…" : "Run"}
        </button>
        {problem ? <Unavailable code={problem} /> : null}
      </section>
      {run ? <FaultResult run={run} narrow={narrow} /> : null}
    </>
  );
}

function FaultResult({ run, narrow }: { run: FaultRun; narrow: boolean }) {
  const resp = useMemo(() => responseOption(run, narrow), [run, narrow]);
  const chan = useMemo(() => channelOption(run, narrow), [run, narrow]);
  const i = run.injected;
  const s = run.support;
  return (
    <>
      <section className="panel">
        <div className="panel-head">
          <h2>Detector response</h2>
          <span className="tag">smoothed score · log scale</span>
        </div>
        <p className={run.first_alert.perturbed ? "result alerted" : "result"} role="status">
          {responseText(run.first_alert, i.onset)}
        </p>
        <div className="legend" aria-hidden>
          <span>
            <i style={{ borderTopColor: "#c3cad2" }} /> With the change
          </span>
          <span>
            <i style={{ borderTopColor: "rgba(195,202,210,.38)" }} /> Unchanged recording
          </span>
          <span>
            <i style={{ borderTopColor: "#aab2bb", borderTopStyle: "dashed" }} /> Threshold {num(run.threshold)}
          </span>
          <span>
            <i className="dot" style={{ background: "rgba(232,96,28,.6)", borderRadius: 2 }} /> Alert
          </span>
          <span>
            <i className="dot" style={{ background: "rgba(245,185,66,.5)", borderRadius: 2 }} /> Injected change (simulated)
          </span>
        </div>
        <ReactECharts option={resp} notMerge style={{ height: narrow ? 240 : 300 }} aria-label="Smoothed detector score with and without the injected change" />
        <p className="foot">The detector sees only the changed samples. It is not told what was injected and does not name a fault. Smoothing and the 3-of-4 alert rule start from the recording's own history, so the unchanged line matches the recorded series.</p>
      </section>

      {chan && run.channel ? (
        <section className="panel">
          <div className="panel-head">
            <h2>The changed channel</h2>
            <span className="tag">{run.channel.name} · five-minute {run.channel.feature.endsWith("mean") ? "mean" : "active share"}</span>
          </div>
          <ReactECharts option={chan} notMerge style={{ height: narrow ? 200 : 240 }} aria-label="The changed channel with and without the injected change" />
          <p className="foot">Values as recorded; units not loaded.</p>
        </section>
      ) : null}

      <section className="panel">
        <div className="panel-head">
          <h2>What was injected</h2>
          <span className="tag warn">ground truth · simulated</span>
        </div>
        <dl className="insp-values">
          <div>
            <dt>Change</dt>
            <dd>
              {FAULTS.find((f) => f[0] === i.fault)?.[1]}
              {i.channel ? ` on ${i.channel}` : ""}
              {i.stuck_value !== null ? `, held at ${i.stuck_value}` : ""}
            </dd>
          </div>
          <div>
            <dt>Size</dt>
            <dd>{severityText(i.fault, i.severity, i.training_std)}</dd>
          </div>
          <div>
            <dt>From</dt>
            <dd>
              {i.onset.slice(0, 16)} to {i.end.slice(0, 16)}
            </dd>
          </div>
          <div>
            <dt>Seed</dt>
            <dd>{i.seed}</dd>
          </div>
          <div>
            <dt>Valid bins</dt>
            <dd>
              {run.bins.perturbed_valid} of {run.bins.expected} with the change; {run.bins.baseline_valid} of {run.bins.expected} without
            </dd>
          </div>
          <div>
            <dt>Training range</dt>
            <dd>
              {s.perturbed.features_outside} of {s.training_features} features left the training range at some point with the change; {s.baseline.features_outside} without it{s.baseline.features_outside > 0 ? " (the recorded test period already differs from the training period)" : ""}.
              {s.perturbed.features_outside > s.baseline.features_outside ? " The change pushed more features outside what the detector was trained on, which makes its score less reliable." : ""}
            </dd>
          </div>
        </dl>
        {run.notes.map((n) => (
          <p key={n} className="foot">
            {n}
          </p>
        ))}
      </section>
    </>
  );
}

function TwinLab() {
  const narrow = useNarrow();
  const [form, setForm] = useState<TwinRequest>({ hours: 1, base_demand_kg_s: 0.09, noise_percent: 2, scenario: "leak_onset", onset_minutes: 20, magnitude_kg_s: 0.02, seed: 0 });
  const [run, setRun] = useState<TwinRunResult | null>(null);
  const [twin, setTwin] = useState<TwinRunResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState("");
  const set = <K extends keyof TwinRequest>(k: K, v: TwinRequest[K]) => setForm((f) => ({ ...f, [k]: v }));

  const go = () => {
    setBusy(true);
    setProblem("");
    setTwin(null);
    runTwin(form).then(setRun, (e: Error) => setProblem(e instanceof LabError ? e.code : "network")).finally(() => setBusy(false));
  };
  // The same size as the other kind of outflow: the plant cannot tell them apart.
  const compare = () => {
    if (!run || run.scenario === "none") return;
    runTwin({ ...form, scenario: run.scenario === "leak_onset" ? "demand_step" : "leak_onset" }).then(setTwin, () => setProblem("failed"));
  };

  const p = useMemo(() => (run ? pressureOption(run, narrow) : null), [run, narrow]);
  const f = useMemo(() => (run ? flowOption(run, narrow) : null), [run, narrow]);
  if (problem === "not_served") return <Unavailable code={problem} />;
  const same = run && twin && JSON.stringify(run.pressure_bar_g) === JSON.stringify(twin.pressure_bar_g) && JSON.stringify(run.compressor_flow_kg_s) === JSON.stringify(twin.compressor_flow_kg_s);

  return (
    <>
      <section className="panel">
        <div className="panel-head">
          <h2>Declare a scenario</h2>
        </div>
        <p className="foot top-foot">A generic 10 m³ receiver with two fixed compressors and one variable-speed compressor under the baseline controller. It is not a model of the MetroPT-3 compressor.</p>
        <div className="lab-form">
          <Field label="Scenario">
            <select value={form.scenario} onChange={(e) => set("scenario", e.target.value as TwinRequest["scenario"])}>
              <option value="none">Baseline cycles</option>
              <option value="demand_step">Demand step</option>
              <option value="leak_onset">Leak begins</option>
            </select>
          </Field>
          <Field label="Run length (hours)">
            <input type="number" min={0.25} max={6} step={0.25} value={form.hours} onChange={(e) => set("hours", Number(e.target.value))} />
          </Field>
          <Field label="Base demand (kg/s)">
            <input type="number" min={0.02} max={0.15} step={0.01} value={form.base_demand_kg_s} onChange={(e) => set("base_demand_kg_s", Number(e.target.value))} />
          </Field>
          <Field label="Demand noise (%)">
            <input type="number" min={0} max={10} step={0.5} value={form.noise_percent} onChange={(e) => set("noise_percent", Number(e.target.value))} />
          </Field>
          {form.scenario !== "none" ? (
            <>
              <Field label="Begins after (minutes)">
                <input type="number" min={1} max={350} step={1} value={form.onset_minutes} onChange={(e) => set("onset_minutes", Number(e.target.value))} />
              </Field>
              <Field label="Extra outflow (kg/s)">
                <input type="number" min={0} max={0.05} step={0.005} value={form.magnitude_kg_s} onChange={(e) => set("magnitude_kg_s", Number(e.target.value))} />
              </Field>
            </>
          ) : null}
          <Field label="Seed" hint="The same seed gives the same run.">
            <input type="number" min={0} step={1} value={form.seed} onChange={(e) => set("seed", Math.max(0, Math.floor(Number(e.target.value))))} />
          </Field>
        </div>
        <button className="primary-button run-button" onClick={go} disabled={busy}>
          {busy ? "Running…" : "Run"}
        </button>
        {problem ? <Unavailable code={problem} /> : null}
      </section>

      {run && p && f ? (
        <>
          <section className="panel">
            <div className="panel-head">
              <h2>Receiver pressure</h2>
              <span className="tag warn">simulated</span>
            </div>
            <ReactECharts option={p} notMerge style={{ height: narrow ? 220 : 280 }} aria-label="Simulated receiver pressure with the control band" />
            <p className="foot">
              Dashed: control band {run.band.lower} to {run.band.upper} bar g. Dotted: safety limits. {run.onset_seconds !== null ? "Amber line: the declared change begins. " : ""}Pressure ranged from {run.summary.pressure_min.toFixed(2)} to{" "}
              {run.summary.pressure_max.toFixed(2)} bar g with {run.summary.safety_violations} safety violations.
            </p>
          </section>
          <section className="panel">
            <div className="panel-head">
              <h2>Compressor flow and demand</h2>
            </div>
            <div className="legend" aria-hidden>
              <span>
                <i style={{ borderTopColor: "#c3cad2" }} /> Compressor flow
              </span>
              <span>
                <i style={{ borderTopColor: "rgba(195,202,210,.38)" }} /> Demand, with any injected change
              </span>
            </div>
            <ReactECharts option={f} notMerge style={{ height: narrow ? 200 : 240 }} aria-label="Simulated compressor flow and demand" />
            <p className="foot">
              Energy {run.summary.energy_kwh.toFixed(1)} kWh, {run.summary.starts} compressor starts. {run.notes[0]}
            </p>
            {run.scenario !== "none" ? (
              <>
                <button className="secondary-button" onClick={compare}>
                  Run the same size as {run.scenario === "leak_onset" ? "a demand step" : "a leak"}
                </button>
                {twin ? <p className="result" role="status">{same ? "The pressure and compressor flow are identical: the plant cannot tell a leak from extra demand." : "The two runs differ."}</p> : null}
              </>
            ) : null}
          </section>
          <section className="panel">
            <div className="panel-head">
              <h2>Detector</h2>
              <span className="tag">not applicable</span>
            </div>
            <p>{run.detector_reason}</p>
            <p className="foot">Scoring it would mean inventing the other 14 channels, the digital state sequences, oil temperature and motor current from a pressure curve. For leaks on the real machine, see the recorded air-leak incidents in the Investigation view.</p>
          </section>
        </>
      ) : null}
    </>
  );
}
