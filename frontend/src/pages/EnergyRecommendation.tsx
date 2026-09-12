import { useMemo, useState } from "react";
import {
  AlertTriangle,
  RefreshCw,
  Shield,
} from "lucide-react";

import { recommendOptimization } from "../api";
import { EvidenceBadge } from "../components/EvidenceBadge";
import { MetricCard } from "../components/MetricCard";
import { PageHeader } from "../components/PageHeader";
import type { OptimizationResponse } from "../types";

const DEFAULT_PROFILE = [
  ...Array(15).fill(0.07),
  ...Array(15).fill(0.11),
  ...Array(15).fill(0.145),
  ...Array(15).fill(0.09),
] as number[];

export function EnergyRecommendation() {
  const [pressure, setPressure] = useState(7.0);
  const [leak, setLeak] = useState(0.005);
  const [result, setResult] =
    useState<OptimizationResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");

  const demandSummary = useMemo(
    () => "0.070 → 0.110 → 0.145 → 0.090 kg/s",
    [],
  );

  async function run() {
    setRunning(true);
    setError("");

    try {
      setResult(
        await recommendOptimization({
          initial_pressure_bar_g: pressure,
          demand_profile_kg_s: DEFAULT_PROFILE,
          leak_mass_flow_kg_s: leak,
        }),
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Optimizer failed.",
      );
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="page-shell">
      <PageHeader
        eyebrow="ENERGY RECOMMENDATION"
        title="One advisory action. Sixty seconds. Then re-evaluate."
        description="AeroXAI solves the constrained forecast horizon but exposes only the current short-lived recommendation because the frozen schedule did not remain safe under all open-loop perturbations."
        badge={<EvidenceBadge kind="SIMULATED" />}
      />

      <div className="advisory-banner">
        <Shield size={21} />
        <div>
          <strong>
            Advisory only — no PLC or equipment writes
          </strong>
          <span>
            Recommendation validity: 60 s · continuous
            re-optimization required
          </span>
        </div>
      </div>

      <section className="recommendation-layout">
        <aside className="panel control-panel">
          <span className="panel-kicker">
            FORECAST INPUT
          </span>
          <h2>Current planning state</h2>

          <label className="field">
            <span>
              Receiver pressure
              <small>bar(g)</small>
            </span>
            <input
              type="number"
              value={pressure}
              step={0.1}
              onChange={(event) =>
                setPressure(
                  Number(event.target.value),
                )
              }
            />
          </label>

          <label className="field">
            <span>
              Assumed leak
              <small>kg/s</small>
            </span>
            <input
              type="number"
              value={leak}
              step={0.001}
              min={0}
              onChange={(event) =>
                setLeak(Number(event.target.value))
              }
            />
          </label>

          <div className="profile-summary">
            <small>1 h synthetic demand forecast</small>
            <strong>{demandSummary}</strong>
            <span>15 minutes per block</span>
          </div>

          <button
            className="primary-button"
            onClick={run}
            disabled={running}
          >
            <RefreshCw
              size={17}
              className={running ? "spin" : ""}
            />
            {running
              ? "Solving..."
              : "Generate advisory action"}
          </button>

          {error ? (
            <div className="inline-error">{error}</div>
          ) : null}
        </aside>

        <div className="recommendation-results">
          {result ? (
            <>
              <div className="metrics-grid metrics-grid-3">
                <MetricCard
                  label="Predicted next pressure"
                  value={`${result.explanation.predicted_pressure_end_bar_g.toFixed(
                    3,
                  )} bar(g)`}
                  accent="lime"
                />
                <MetricCard
                  label="Pressure safety margin"
                  value={`${result.explanation.safety_margin_bar.toFixed(
                    3,
                  )} bar`}
                  accent={
                    result.explanation.safety_margin_bar <
                    0.1
                      ? "amber"
                      : "cyan"
                  }
                />
                <MetricCard
                  label="Available reserve"
                  value={`${result.explanation.reserve_available_kg_s.toFixed(
                    3,
                  )} kg/s`}
                  note={`required ${result.explanation.required_reserve_kg_s.toFixed(
                    3,
                  )}`}
                />
              </div>

              <article className="panel action-panel">
                <div className="panel-title-row">
                  <div>
                    <span className="panel-kicker">
                      CURRENT ACTION
                    </span>
                    <h2>Dispatch recommendation</h2>
                  </div>
                  <span className="validity-pill">
                    valid {result.safety.valid_for_seconds}s
                  </span>
                </div>

                <div className="compressor-grid">
                  {result.current_actions.map(
                    (action) => (
                      <div
                        key={action.compressor_id}
                        className={`compressor-card ${
                          action.state === "on"
                            ? "compressor-on"
                            : "compressor-off"
                        }`}
                      >
                        <div className="compressor-ring">
                          <span>
                            {Math.round(
                              action.load_fraction * 100,
                            )}
                            %
                          </span>
                        </div>
                        <div>
                          <strong>
                            {action.compressor_id}
                          </strong>
                          <small>
                            {action.kind.toUpperCase()} ·{" "}
                            {action.state.toUpperCase()}
                          </small>
                        </div>
                      </div>
                    ),
                  )}
                </div>

                <div className="reason-box">
                  <strong>Why this action?</strong>
                  <p>{result.explanation.reason}</p>
                </div>
              </article>

              <article className="panel warning-panel">
                <AlertTriangle size={23} />
                <div>
                  <strong>
                    Open-loop horizon is not approved
                  </strong>
                  <p>
                    {result.safety.robustness_status.split("_").join(" ")}
                    . The solver's full horizon is a
                    planning forecast; only the current
                    action is surfaced for advisory use.
                  </p>
                </div>
              </article>
            </>
          ) : (
            <article className="panel empty-state">
              <div className="decision-radar">
                <span />
                <span />
                <span />
              </div>
              <h2>Awaiting optimization</h2>
              <p>
                Generate a recommendation to see the
                current compressor dispatch, pressure
                margin, reserve margin, and explanation.
              </p>
            </article>
          )}
        </div>
      </section>
    </div>
  );
}
