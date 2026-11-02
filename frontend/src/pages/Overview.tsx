import { useEffect, useState } from "react";
import { ArrowRight } from "lucide-react";

import { getDetectionEvidence, getEvidenceSummary } from "../api";
import { ErrorPanel } from "../components/ErrorPanel";
import { LoadingPanel } from "../components/LoadingPanel";
import { PageHeader } from "../components/PageHeader";
import { getManifest } from "../syrus/client";
import type { Manifest } from "../syrus/types";
import type { DetectionEvidence, EvidenceSummary } from "../types";
import { dateLabel } from "./Investigation";

const STEPS = [
  ["01 · DETECT", "Review a real alert", "Frozen detector, scores, incident timing."],
  ["02 · EXPLAIN", "Inspect verified findings", "Contributions, smoothing, counterfactual tests."],
  ["03 · INVESTIGATE", "Ask Syrus", "Cited answers with their limits."],
  ["04 · GUIDE", "Choose the next check", "Unranked candidates; real actions withheld."],
];

export function Overview({ open }: { open: (hash: string) => void }) {
  const [summary, setSummary] = useState<EvidenceSummary | null>(null);
  const [detection, setDetection] = useState<DetectionEvidence | null>(null);
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([getEvidenceSummary(), getDetectionEvidence(), getManifest()])
      .then(([s, d, m]) => {
        setSummary(s);
        setDetection(d);
        setManifest(m);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  if (error) return <ErrorPanel error={error} />;
  if (!summary || !detection || !manifest) return <LoadingPanel />;

  const m = detection.metrics;
  const saving = summary.evidence_classes.SIMULATED.optimizer.comparison.energy_saving_percent as number;
  const cases = [...manifest.cases].sort((a, b) => a.alert_time.localeCompare(b.alert_time));
  const timing = (alertTime: string) => {
    const hit = detection.incidents.find((i) => i.explanation_timestamp.replace("T", " ").slice(0, 16) === alertTime.slice(0, 16));
    if (!hit) return null;
    return hit.timing === "pre_onset" ? "Pre-onset alert" : "Post-onset alert";
  };

  return (
    <div className="page-shell">
      <PageHeader
        eyebrow="SYSTEM OVERVIEW"
        title="From alerts to evidence-backed decisions."
        description="Historical MetroPT-3 replay · advisory only."
        badge={<span className="badge-outline">Historical replay · Advisory</span>}
      />

      <section className="kpis" aria-label="Detector evaluation">
        <article>
          <span>Documented incidents detected</span>
          <strong>{Math.round(m.timely_incident_recall * 4)} / 4</strong>
          <small>within the predefined timely window</small>
        </article>
        <article>
          <span>Pre-onset alerts</span>
          <strong>{Math.round(m.pre_onset_incident_recall * 4)} / 4</strong>
          <small>not a verified physical precursor</small>
        </article>
        <article>
          <span>Benchmark-unmatched episodes</span>
          <strong>{m.false_alerts_per_24h.toFixed(3)}</strong>
          <small>per 24 h of valid exposure; not confirmed false alarms</small>
        </article>
        <article>
          <span>Product mode</span>
          <strong className="accent-text">Advisory</strong>
          <small>No equipment control</small>
        </article>
      </section>

      <ol className="steps">
        {STEPS.map(([k, t, d]) => (
          <li key={k}>
            <span>{k}</span>
            <strong>{t}</strong>
            <small>{d}</small>
          </li>
        ))}
      </ol>

      <div className="overview-split">
        <section className="panel">
          <div className="panel-head">
            <h2>Historical incidents</h2>
            <button className="link-button" onClick={() => open("investigation")}>
              Open investigation <ArrowRight size={14} />
            </button>
          </div>
          <ul className="incident-list">
            {cases.map((c, i) => (
              <li key={c.case_id}>
                <button onClick={() => open(`investigation/${c.case_id}`)}>
                  <span>
                    <strong>
                      Incident {i + 1} · {dateLabel(c.alert_time)}
                    </strong>
                    <small>Alert {c.alert_time.slice(11, 16)} · dataset local time</small>
                  </span>
                  {timing(c.alert_time) ? <em className="tag">{timing(c.alert_time)}</em> : null}
                </button>
              </li>
            ))}
          </ul>
        </section>

        <section className="panel">
          <div className="panel-head">
            <h2>Syrus · Engineering companion</h2>
          </div>
          <p>Ask about a selected incident. Answers cite verified records and state what the data cannot show.</p>
          <ul className="checks">
            <li>Relevant evidence only</li>
            <li>Reviewed scientific qualifications</li>
            <li>Real and simulated evidence kept apart</li>
          </ul>
          <button className="primary-button" onClick={() => open("investigation/c-h3vz/retro")}>
            Ask Syrus about Incident 2
          </button>
        </section>
      </div>

      <section className="panel sim-panel">
        <div className="panel-head">
          <h2>Simulation only</h2>
          <span className="tag warn">SIMULATED</span>
        </div>
        <p>
          Nominal dispatch saving in the synthetic scenario: <b>{saving.toFixed(3)}%</b>. Not measured on the MetroPT-3 compressor.
        </p>
        <div className="sim-links">
          <button className="secondary-button" onClick={() => open("proof")}>
            Evidence proof
          </button>
          <button className="secondary-button" onClick={() => open("twin")}>
            Simulation lab
          </button>
        </div>
      </section>
    </div>
  );
}
