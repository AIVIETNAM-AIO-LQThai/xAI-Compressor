import { refFor, type Ref } from "./Inspector";
import { groupName, recordTitle, title } from "./labels";
import type { CaseFile, CaseRecord } from "./types";

type Rec = (id: string) => CaseRecord | undefined;
const str = (r: CaseRecord | undefined) => (r && typeof r.display === "string" ? r.display : "n/a");
const obj = (r: CaseRecord | undefined) => (r && typeof r.display === "object" ? (r.display as Record<string, string>) : {});
const clock = (t: string) => t.slice(11, 16);
const persistenceTag = (r: CaseRecord | undefined) => {
  const v = r?.value as { hits: number; window: number } | undefined;
  return v ? `${v.hits} of last ${v.window} smoothed bins` : "persistence rule";
};
const firstSentence = (t: string) => t.match(/^.*?[.!?](?=\s|$)/)?.[0] ?? t;

function Metric({ label, value, note, accent, onOpen }: { label: string; value: string; note?: string; accent?: boolean; onOpen?: () => void }) {
  return (
    <button className={`metric ${accent ? "accent" : ""}`} onClick={onOpen} disabled={!onOpen}>
      <span>{label}</span>
      <strong>{value}</strong>
      {note ? <small>{note}</small> : null}
    </button>
  );
}

function Mechanics({ rec, open }: { rec: Rec; open: (id: string) => void }) {
  const hits = rec("alert.hits")?.value as { hits: string[]; misses: string[]; undetermined: string[] } | undefined;
  const prev = rec("alert.prev_smoothed")?.value as { score: number; time: string } | undefined;
  const raw = rec("alert.raw")?.value as number | undefined;
  const smoothed = rec("alert.smoothed")?.value as number | undefined;
  const threshold = rec("alert.threshold")?.value as number | undefined;
  if (!hits || !prev || raw === undefined || smoothed === undefined || threshold === undefined) return null;
  const bins = [
    ...hits.misses.map((t) => ({ t, s: "check" })),
    ...hits.undetermined.map((t) => ({ t, s: "either" })),
    ...hits.hits.map((t) => ({ t, s: "hit" })),
  ].sort((a, b) => a.t.localeCompare(b.t));
  const top = Math.max(prev.score, smoothed, threshold, raw);
  const bar = (v: number) => `${Math.max(1.5, (v / top) * 100)}%`;
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Alert mechanics</h2>
        <span className="tag">{persistenceTag(rec("detector.persistence"))}</span>
      </div>
      <ol className="bins">
        {bins.map((b) => (
          <li key={b.t} className={b.s}>
            <time>{clock(b.t)}</time>
            <span>{b.s === "hit" ? "at or above" : b.s === "either" ? "one of two" : "below"}</span>
          </li>
        ))}
      </ol>
      <button className="link-button note-link" onClick={() => open("alert.hits")}>
        {str(rec("alert.hits"))}
      </button>
      <div className="bars">
        <div>
          <label>
            Previous smoothed state · {prev.time.slice(11, 16)}
            <b>{obj(rec("alert.prev_smoothed")).score}</b>
          </label>
          <div className="bar">
            <i style={{ width: bar(prev.score) }} />
          </div>
        </div>
        <div>
          <label>
            Current raw score
            <b>{str(rec("alert.raw"))}</b>
          </label>
          <div className="bar muted-bar">
            <i style={{ width: bar(raw) }} />
          </div>
        </div>
        <p className="foot">
          The previous smoothed state accounts for {str(rec("alert.memory"))} of the current smoothed score.{" "}
          <button className="link-button" onClick={() => open("alert.memory")}>
            Source
          </button>
        </p>
      </div>
    </section>
  );
}

function Contributions({ rec, c, open }: { rec: Rec; c: CaseFile; open: (id: string) => void }) {
  const rows = c.records.filter((r) => r.id.startsWith("contrib.") && r.id !== "contrib.others");
  if (!rows.length) return null;
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Detector contributions</h2>
        <span className="tag">not probabilities</span>
      </div>
      <ul className="contribs">
        {rows.map((r) => {
          const v = r.value as { group: string; share: number };
          const d = obj(r);
          const rank = rec(`calrank.${v.group}`);
          return (
            <li key={r.id}>
              <button className="link-button" onClick={() => open(r.id)}>
                {groupName(v.group)}
              </button>
              <div className="bar">
                <i style={{ width: `${v.share * 100}%` }} />
              </div>
              <b>{d.share}</b>
              <small>
                percentile {d.percentile}
                {rank ? ` · rank ${str(rank)}` : ""}
              </small>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function Timeline({ c, open }: { c: CaseFile; open: (id: string) => void }) {
  const win = c.records.find((r) => r.id === "temporal.window")?.value as { groups: string[]; start: string; end: string } | undefined;
  const rows = c.records.filter((r) => /^temporal\.\d{4}$/.test(r.id));
  if (!win || rows.length < 2) return null;
  const W = 520;
  const H = 150;
  const x = (i: number) => 28 + (i / (rows.length - 1)) * (W - 40);
  const y = (p: number) => 10 + (1 - p) * (H - 36);
  const dash = ["", "6 3", "2 3"];
  const tone = ["var(--cyan)", "var(--text)", "var(--muted)"];
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Evidence over the window</h2>
        <span className="tag">calibration percentile</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Calibration percentile of each group's smoothed contribution across the explained window" className="timeline">
        {[0, 0.5, 1].map((p) => (
          <g key={p}>
            <line x1="28" x2={W - 12} y1={y(p)} y2={y(p)} className="grid" />
            <text x="2" y={y(p) + 4}>
              {p * 100}
            </text>
          </g>
        ))}
        {win.groups.map((g, gi) => (
          <polyline
            key={g}
            fill="none"
            stroke={tone[gi]}
            strokeWidth="2"
            strokeDasharray={dash[gi]}
            points={rows.map((r, i) => `${x(i)},${y((r.value as { percentiles: Record<string, number> }).percentiles[g])}`).join(" ")}
          />
        ))}
        {rows.map((r, i) =>
          i % 3 === 0 || i === rows.length - 1 ? (
            <text key={r.id} x={x(i)} y={H - 6} textAnchor="middle">
              {clock((r.value as { time: string }).time)}
            </text>
          ) : null,
        )}
      </svg>
      <div className="legend">
        {win.groups.map((g, gi) => (
          <span key={g}>
            <i style={{ borderTopColor: tone[gi], borderTopStyle: gi === 0 ? "solid" : gi === 1 ? "dashed" : "dotted" }} />
            {groupName(g)}
          </span>
        ))}
        <button className="link-button" onClick={() => open("temporal.window")}>
          Scope
        </button>
      </div>
    </section>
  );
}

function Repairs({ c, open }: { c: CaseFile; open: (id: string) => void }) {
  const rows = c.records.filter((r) => r.id.startsWith("cf.") && r.id !== "cf.prewindow");
  if (!rows.length) return null;
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Counterfactual repairs</h2>
        <span className="tag">tests the detector, not the machine</span>
      </div>
      <table className="plain">
        <thead>
          <tr>
            <th>Group repaired</th>
            <th>Smoothed after</th>
            <th>Result</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const d = obj(r);
            return (
              <tr key={r.id}>
                <td>
                  <button className="link-button" onClick={() => open(r.id)}>
                    {groupName(r.id.slice(3))}
                  </button>
                </td>
                <td>{d.repaired_smoothed}</td>
                <td>{d.result}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="foot">
        Top five groups, one at a time, checked at the alert bin only.{" "}
        <button className="link-button" onClick={() => open("detector.repair")}>
          Method
        </button>
      </p>
    </section>
  );
}

function Findings({ c, open }: { c: CaseFile; open: (r: Ref) => void }) {
  const fired = c.findings.filter((f) => f.fires);
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Verified findings</h2>
        <span className="tag">{fired.length} apply</span>
      </div>
      <ul className="findings">
        {fired.map((f) => (
          <li key={f.id}>
            <button onClick={() => open({ kind: "finding", id: f.id })}>
              <strong>{title(f.id)}</strong>
              <span>{firstSentence(f.text)}</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

function Candidates({ c, open }: { c: CaseFile; open: (r: Ref) => void }) {
  const hyps = c.records.filter((r) => r.kind === "hypothesis");
  if (!hyps.length) return null;
  const layers = ["Process", "Evidence"].map((layer) => ({ layer, items: hyps.filter((h) => obj(h).layer === layer) }));
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Candidate explanations</h2>
        <span className="tag">unranked · all open</span>
      </div>
      {layers.map(({ layer, items }) => (
        <div key={layer} className="layer">
          <h3>{layer} layer</h3>
          <div className="chips">
            {items.map((h) => (
              <button key={h.id} className="chip" onClick={() => open(refFor(h.id, c))}>
                {obj(h).label}
              </button>
            ))}
          </div>
        </div>
      ))}
      {c.pair_limits.map((p) => (
        <p key={p.pair.join()} className="limit">
          {p.pair.map(groupName).join(" and ")} {p.text.en}.
        </p>
      ))}
      <h3>Discriminating observations</h3>
      <ul className="tests">
        {c.tests.map((t) => (
          <li key={t.id}>
            <button className="link-button" onClick={() => open({ kind: "test", id: t.id })}>
              {t.observations[0].text.en.split(/[,.]/)[0]}
            </button>
            <small>
              {t.observations[0].availability === "external" ? "needs outside data" : "from the dataset"} · {t.maturity_text.en.split(":")[0]} · {t.result_text.en}
            </small>
          </li>
        ))}
      </ul>
    </section>
  );
}

function Boundaries({ c, rec, open }: { c: CaseFile; rec: Rec; open: (r: Ref) => void }) {
  const gaps = c.records.filter((r) => r.kind === "unavailable");
  const bridge = rec("bridge.status");
  const label = rec("label.incident");
  const timing = rec("label.timing");
  const items = (bridge && typeof bridge.display === "object" ? ((bridge.display as { missing?: string[] }).missing ?? []) : []) as string[];
  return (
    <>
      {label ? (
        <section className="panel">
          <div className="panel-head">
            <h2>Documented incident</h2>
            <span className="tag">retrospective label</span>
          </div>
          <p>
            {obj(label).condition} · {obj(label).interval}
            {timing ? ` · alert ${str(timing)} onset` : ""}
          </p>
          <button className="link-button" onClick={() => open({ kind: "finding", id: "F-LABEL" })}>
            What this label is
          </button>
        </section>
      ) : null}
      <section className="panel">
        <div className="panel-head">
          <h2>Missing evidence</h2>
          <span className="tag">not loaded</span>
        </div>
        <ul className="missing">
          {gaps.map((g) => (
            <li key={g.id}>
              <button className="link-button" onClick={() => open({ kind: "record", id: g.id })}>
                {recordTitle(g.id)}
              </button>
            </li>
          ))}
        </ul>
      </section>
      {bridge ? (
        <section className="panel">
          <div className="panel-head">
            <h2>Physical action</h2>
            <span className="tag warn">{obj(bridge).status}</span>
          </div>
          <p className="muted">No real-machine recommendation. Missing: {items.join("; ")}.</p>
        </section>
      ) : null}
    </>
  );
}

export function EvidencePane({ c, onInspect }: { c: CaseFile; onInspect: (r: Ref) => void }) {
  const byId = new Map(c.records.map((r) => [r.id, r]));
  const rec: Rec = (id) => byId.get(id);
  const open = (id: string) => onInspect(refFor(id, c));
  return (
    <div className="evidence-pane">
      <div className="metrics">
        <Metric label="Smoothed score" value={str(rec("alert.smoothed"))} note={`at ${clock(c.alert_time)}`} onOpen={() => open("alert.smoothed")} />
        <Metric label="Threshold" value={str(rec("alert.threshold"))} note="calibration q0.995" onOpen={() => open("alert.threshold")} />
        <Metric label="Above threshold" value={str(rec("alert.margin"))} note="not a fault probability" accent onOpen={() => open("alert.margin")} />
      </div>
      <Mechanics rec={rec} open={open} />
      <Findings c={c} open={onInspect} />
      <Contributions rec={rec} c={c} open={open} />
      <Timeline c={c} open={open} />
      <Repairs c={c} open={open} />
      <Candidates c={c} open={onInspect} />
      <Boundaries c={c} rec={rec} open={onInspect} />
    </div>
  );
}
