import { useEffect, useState } from "react";
import {
  Activity,
  Gauge,
  ShieldCheck,
  Zap,
} from "lucide-react";

import { getEvidenceSummary } from "../api";
import { ErrorPanel } from "../components/ErrorPanel";
import { EvidenceBadge } from "../components/EvidenceBadge";
import { LoadingPanel } from "../components/LoadingPanel";
import { MetricCard } from "../components/MetricCard";
import { PageHeader } from "../components/PageHeader";
import type { EvidenceSummary } from "../types";

function pct(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function Overview() {
  const [data, setData] =
    useState<EvidenceSummary | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getEvidenceSummary()
      .then(setData)
      .catch((err: Error) => setError(err.message));
  }, []);

  if (error) {
    return <ErrorPanel error={error} />;
  }

  if (!data) {
    return <LoadingPanel />;
  }

  const detection =
    data.evidence_classes.REAL.detection;
  const optimizer =
    data.evidence_classes.SIMULATED.optimizer;
  const baseline =
    data.evidence_classes.SIMULATED.baseline;
  const robustness =
    data.evidence_classes.SIMULATED.robustness;

  const saving =
    optimizer.comparison.energy_saving_percent as number;
  const highLeak =
    baseline.leak_comparison
      .additional_energy_percent as number;

  return (
    <div className="page-shell">
      <PageHeader
        eyebrow="SYSTEM OVERVIEW"
        title="Compressed-air intelligence, with evidence boundaries visible."
        description="AeroXAI combines real-telemetry anomaly detection with a separate physics-based simulation and advisory optimizer. The UI never presents simulated savings as measured plant performance."
      />

      <div className="evidence-ribbon">
        <EvidenceBadge kind="REAL" />
        <span>
          MetroPT-3 detection + anomaly explanation
        </span>
        <EvidenceBadge kind="SIMULATED" />
        <span>
          twin + compressor dispatch + robustness
        </span>
      </div>

      <section className="metrics-grid">
        <MetricCard
          label="Documented incidents detected"
          value={`${Math.round(
            detection.metrics.timely_incident_recall * 4,
          )} / 4`}
          note="within the predefined timely window"
          accent="lime"
        />
        <MetricCard
          label="Pre-onset detection"
          value={pct(
            detection.metrics.pre_onset_incident_recall,
          )}
          note="2 of 4 documented incidents"
        />
        <MetricCard
          label="False alert episodes"
          value={detection.metrics.false_alerts_per_24h.toFixed(
            2,
          )}
          note="per 24 h of valid test exposure"
          accent="amber"
        />
        <MetricCard
          label="Nominal simulated dispatch saving"
          value={`${saving.toFixed(3)}%`}
          note="electrical energy only; synthetic scenario"
        />
      </section>

      <section className="overview-grid">
        <article className="panel hero-panel">
          <div className="panel-title-row">
            <div>
              <span className="panel-kicker">
                CONTROL POSTURE
              </span>
              <h2>Advisory by design</h2>
            </div>
            <ShieldCheck size={28} />
          </div>

          <div className="safety-state">
            <div className="safety-orb">
              <span>60 s</span>
              <small>validity</small>
            </div>
            <div>
              <h3>
                Recommendation must be re-optimized
                before the next action.
              </h3>
              <p>
                The frozen one-hour open-loop schedule
                failed several perturbation tests, so
                AeroXAI explicitly rejects unattended
                open-loop execution.
              </p>
            </div>
          </div>

          <div className="status-strip">
            <span>
              <i className="status-dot good" />
              PLC override disabled
            </span>
            <span>
              <i className="status-dot warn" />
              Open-loop schedule not approved
            </span>
            <span>
              <i className="status-dot good" />
              Re-optimization required
            </span>
          </div>
        </article>

        <article className="panel">
          <span className="panel-kicker">
            ENERGY SIGNAL
          </span>
          <h2>Leak scenario impact</h2>
          <div className="big-number">
            +{highLeak.toFixed(2)}%
          </div>
          <p className="muted">
            Simulated baseline energy increase when
            leak input rises from 0.005 to 0.020 kg/s.
          </p>
          <div className="inline-icon-copy">
            <Zap size={18} />
            <span>
              Separate from the optimizer's 0.369%
              dispatch saving.
            </span>
          </div>
        </article>

        <article className="panel">
          <span className="panel-kicker">
            MODEL VALIDITY
          </span>
          <h2>Robustness stress test</h2>
          <div className="robustness-card">
            <Gauge size={22} />
            <div>
              <strong>
                {robustness.all_tested_perturbations_safe
                  ? "Passed tested perturbations"
                  : "Not robust open-loop"}
              </strong>
              <span>
                Worst tested pressure:{" "}
                {Number(
                  robustness.worst_tested_minimum_pressure_bar_g,
                ).toFixed(2)}{" "}
                bar(g)
              </span>
            </div>
          </div>
          <p className="muted">
            This is a limitation that changes product
            behavior, not a result hidden by retuning.
          </p>
        </article>

        <article className="panel">
          <span className="panel-kicker">
            EXPLAINABILITY
          </span>
          <h2>Two explanation layers</h2>
          <div className="explain-stack">
            <div>
              <Activity size={18} />
              <p>
                <strong>Detection:</strong> exact PCA
                reconstruction-error contributions.
              </p>
            </div>
            <div>
              <ShieldCheck size={18} />
              <p>
                <strong>Recommendation:</strong>{" "}
                schedule, constraint, pressure and
                reserve rationale.
              </p>
            </div>
          </div>
        </article>
      </section>
    </div>
  );
}
