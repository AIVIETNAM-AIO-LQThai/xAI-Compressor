import { useEffect, useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  CircleStop,
  FlaskConical,
  GitBranch,
  LockKeyhole,
  ShieldCheck,
} from "lucide-react";

import {
  getProofBundle,
  verifyProofBundle,
} from "../api";
import { ErrorPanel } from "../components/ErrorPanel";
import { EvidenceBadge } from "../components/EvidenceBadge";
import { LoadingPanel } from "../components/LoadingPanel";
import { PageHeader } from "../components/PageHeader";
import type {
  ProofBundleResponse,
  ProofBundleVerification,
} from "../types";

import "./ProofManifest.css";
import "./ProofIntegrity.css";

function humanize(value: string) {
  return value
    .replace(/_/g, " ")
    .toLowerCase()
    .replace(
      /\b\w/g,
      (letter: string) => letter.toUpperCase(),
    );
}

function commandLabel(
  compressorId: string,
  value: number,
) {
  void compressorId;

  if (value === 0) {
    return "OFF";
  }

  if (value === 1) {
    return "ON";
  }

  return `${Math.round(value * 100)}%`;
}

function shortHash(value: string) {
  if (value.length <= 20) {
    return value;
  }

  return `${value.slice(0, 10)}…${value.slice(-10)}`;
}

export function ProofManifest() {
  const [bundle, setBundle] =
    useState<ProofBundleResponse | null>(null);
  const [verification, setVerification] =
    useState<ProofBundleVerification | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getProofBundle()
      .then(async (proofBundle) => {
        setBundle(proofBundle);
        const result = await verifyProofBundle(
          proofBundle,
        );
        setVerification(result);
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  if (error) {
    return <ErrorPanel error={error} />;
  }

  if (!bundle || !verification) {
    return <LoadingPanel />;
  }

  const data = bundle.manifest;
  const real = data.real_evidence_path;
  const simulated = data.simulated_validation_path;
  const action = simulated.robust_action;
  const state = simulated.physical_state;
  const recovery =
    simulated.inverse_physics_recovery;
  const checks = Object.entries(
    verification.checks,
  );

  return (
    <div className="page-shell proof-page">
      <PageHeader
        eyebrow="PROOF-CARRYING EVIDENCE"
        title="A recommendation is only as strong as the boundary it refuses to cross."
        description="This manifest shows the real MetroPT evidence path and the separate simulated validation path side by side. AeroXAI deliberately stops real-asset reasoning when calibration evidence is missing."
      />

      <div className="proof-identity">
        <div>
          <span className="proof-overline">
            BUNDLE ID
          </span>
          <code>{bundle.bundle_id}</code>
          <span className="proof-identity-secondary">
            Manifest {data.manifest_id}
          </span>
        </div>
        <div className="proof-identity-state">
          <ShieldCheck size={18} />
          <span>ADVISORY · NO EQUIPMENT WRITE</span>
        </div>
      </div>

      <section
        className={
          verification.verified
            ? "proof-verification proof-verification-pass"
            : "proof-verification proof-verification-fail"
        }
      >
        <div className="proof-verification-head">
          <div className="proof-verification-icon">
            {verification.verified ? (
              <ShieldCheck size={25} />
            ) : (
              <AlertTriangle size={25} />
            )}
          </div>
          <div>
            <span className="proof-overline">
              TAMPER-EVIDENT BUNDLE
            </span>
            <h2>
              {verification.verified
                ? "Cryptographic integrity verified"
                : "Integrity verification failed"}
            </h2>
            <p>
              SHA-256 checks, source-file hashes and a
              semantic manifest rebuild are evaluated
              against the exact bundle shown below.
            </p>
          </div>
          <div className="proof-verification-status">
            {verification.verified
              ? "VERIFIED"
              : "FAILED"}
          </div>
        </div>

        <div className="proof-verification-checks">
          {checks.map(([name, passed]) => (
            <div
              className={
                passed
                  ? "proof-check proof-check-pass"
                  : "proof-check proof-check-fail"
              }
              key={name}
            >
              {passed ? (
                <CheckCircle2 size={15} />
              ) : (
                <CircleStop size={15} />
              )}
              <span>{humanize(name)}</span>
              <strong>{passed ? "PASS" : "FAIL"}</strong>
            </div>
          ))}
        </div>

        <div className="proof-artifact-grid">
          {bundle.artifacts.map((artifact) => (
            <article
              className="proof-artifact"
              key={artifact.role}
            >
              <div className="proof-artifact-head">
                <EvidenceBadge
                  kind={artifact.evidence_class}
                />
                <span>{humanize(artifact.role)}</span>
              </div>
              <code>{artifact.path}</code>
              <dl>
                <div>
                  <dt>Canonical JSON</dt>
                  <dd title={artifact.canonical_sha256}>
                    {shortHash(
                      artifact.canonical_sha256,
                    )}
                  </dd>
                </div>
                <div>
                  <dt>Exact file</dt>
                  <dd title={artifact.file_sha256}>
                    {shortHash(
                      artifact.file_sha256,
                    )}
                  </dd>
                </div>
              </dl>
            </article>
          ))}
          <article className="proof-artifact">
            <div className="proof-artifact-head">
              <ShieldCheck size={17} />
              <span>Manifest digest</span>
            </div>
            <code>
              {shortHash(bundle.manifest_sha256)}
            </code>
            <p>
              The embedded source payloads must rebuild
              this exact manifest.
            </p>
          </article>
        </div>

        {verification.errors.length > 0 && (
          <div className="proof-verification-errors">
            {verification.errors.map((message) => (
              <p key={message}>{message}</p>
            ))}
          </div>
        )}
      </section>

      <section className="proof-path-grid">
        <article className="proof-track proof-track-real">
          <div className="proof-track-head">
            <EvidenceBadge kind="REAL" />
            <span>INCIDENT {real.incident_id}</span>
          </div>

          <div className="proof-track-title">
            <GitBranch size={24} />
            <div>
              <h2>Observed evidence path</h2>
              <p>
                Real telemetry can support detector
                evidence and candidate hypotheses.
              </p>
            </div>
          </div>

          <div className="proof-flow">
            <div className="proof-node proof-node-good">
              <CheckCircle2 size={18} />
              <div>
                <strong>Verified detector evidence</strong>
                <span>
                  {real.detector_evidence_class}
                </span>
              </div>
            </div>

            <ArrowRight
              size={17}
              className="proof-arrow"
            />

            <div className="proof-node">
              <GitBranch size={18} />
              <div>
                <strong>Candidate hypotheses</strong>
                <span>
                  {real.candidate_hypotheses.length}
                  {" "}plausible explanations retained
                </span>
              </div>
            </div>

            <ArrowRight
              size={17}
              className="proof-arrow"
            />

            <div className="proof-node proof-node-stop">
              <CircleStop size={18} />
              <div>
                <strong>Physical bridge withheld</strong>
                <span>
                  No real-asset action issued
                </span>
              </div>
            </div>
          </div>

          <div className="proof-hypotheses">
            <span className="proof-overline">
              CANDIDATE HYPOTHESES
            </span>
            <div>
              {real.candidate_hypotheses.map(
                (hypothesis) => (
                  <span
                    className="proof-chip"
                    key={hypothesis}
                  >
                    {humanize(hypothesis)}
                  </span>
                ),
              )}
            </div>
          </div>

          <div className="proof-boundary">
            <div className="proof-boundary-icon">
              <LockKeyhole size={22} />
            </div>
            <div>
              <span className="proof-overline">
                SCIENTIFIC STOP CONDITION
              </span>
              <h3>
                {humanize(real.bridge_status)}
              </h3>
              <p>
                The following requirements are still
                missing:
              </p>
              <ul>
                {real.missing_requirements.map(
                  (requirement) => (
                    <li key={requirement}>
                      {humanize(requirement)}
                    </li>
                  ),
                )}
              </ul>
            </div>
          </div>

          <div className="proof-state-row">
            <span>
              Physics executed
              <strong>
                {real.downstream_physics_executed
                  ? "YES"
                  : "NO"}
              </strong>
            </span>
            <span>
              Robust control executed
              <strong>
                {real.robust_control_executed
                  ? "YES"
                  : "NO"}
              </strong>
            </span>
            <span>
              Real recommendation
              <strong>
                {real.recommendation === null
                  ? "WITHHELD"
                  : "ISSUED"}
              </strong>
            </span>
          </div>
        </article>

        <article className="proof-track proof-track-sim">
          <div className="proof-track-head">
            <EvidenceBadge kind="SIMULATED" />
            <span>{simulated.scope}</span>
          </div>

          <div className="proof-track-title">
            <FlaskConical size={24} />
            <div>
              <h2>Physics validation path</h2>
              <p>
                Synthetic truth tests inverse physics,
                uncertainty and robust control.
              </p>
            </div>
          </div>

          <div className="proof-sim-metrics">
            <div>
              <span>Recovery</span>
              <strong>
                {recovery.passed ? "PASS" : "FAIL"}
              </strong>
              <small>
                error{" "}
                {recovery.absolute_error_kg_s.toExponential(
                  2,
                )}{" "}
                kg/s
              </small>
            </div>
            <div>
              <span>Total outflow</span>
              <strong>
                {state.total_outflow_kg_s.center.toFixed(
                  3,
                )}
              </strong>
              <small>
                {state.total_outflow_kg_s.lower.toFixed(
                  3,
                )}
                {" – "}
                {state.total_outflow_kg_s.upper.toFixed(
                  3,
                )}{" "}
                kg/s
              </small>
            </div>
            <div>
              <span>Scenarios</span>
              <strong>{action.scenario_count}</strong>
              <small>bounded physical states</small>
            </div>
            <div>
              <span>Action validity</span>
              <strong>
                {Math.round(action.valid_for_seconds)} s
              </strong>
              <small>then re-optimize</small>
            </div>
          </div>

          <div className="proof-command-panel">
            <div className="proof-command-head">
              <div>
                <span className="proof-overline">
                  ROBUST FIRST ACTION
                </span>
                <h3>
                  Passed independent safety gate
                </h3>
              </div>
              <div className="proof-safe-pill">
                <CheckCircle2 size={16} />
                ROBUST SAFE
              </div>
            </div>

            <div className="proof-command-grid">
              {action.commands.map(
                ([compressorId, value]) => (
                  <div
                    key={compressorId}
                    className="proof-command"
                  >
                    <span>{compressorId}</span>
                    <strong>
                      {commandLabel(
                        compressorId,
                        value,
                      )}
                    </strong>
                  </div>
                ),
              )}
            </div>

            <div className="proof-safety-grid">
              <span>
                Worst min pressure
                <strong>
                  {action.worst_case_min_pressure_bar_g.toFixed(
                    3,
                  )}{" "}
                  bar(g)
                </strong>
              </span>
              <span>
                Worst max pressure
                <strong>
                  {action.worst_case_max_pressure_bar_g.toFixed(
                    3,
                  )}{" "}
                  bar(g)
                </strong>
              </span>
              <span>
                Minimum reserve
                <strong>
                  {action.minimum_reserve_kg_s.toFixed(
                    3,
                  )}{" "}
                  kg/s
                </strong>
              </span>
              <span>
                Horizon energy
                <strong>
                  {action.horizon_energy_kwh.toFixed(
                    3,
                  )}{" "}
                  kWh
                </strong>
              </span>
            </div>
          </div>

          <div className="proof-disconnect">
            <AlertTriangle size={20} />
            <div>
              <strong>
                Not connected to the MetroPT incident
              </strong>
              <p>
                This simulated action validates the
                method only. It is not a command,
                energy result or physical estimate for
                the real incident.
              </p>
            </div>
          </div>
        </article>
      </section>

      <section className="proof-claims-grid">
        <article className="proof-claims proof-claims-allowed">
          <div className="proof-claims-head">
            <CheckCircle2 size={20} />
            <div>
              <span className="proof-overline">
                ALLOWED CLAIMS
              </span>
              <h2>What the evidence supports</h2>
            </div>
          </div>
          <ol>
            {data.allowed_claims.map((claim) => (
              <li key={claim}>{claim}</li>
            ))}
          </ol>
        </article>

        <article className="proof-claims proof-claims-forbidden">
          <div className="proof-claims-head">
            <CircleStop size={20} />
            <div>
              <span className="proof-overline">
                FORBIDDEN CLAIMS
              </span>
              <h2>What AeroXAI will not say</h2>
            </div>
          </div>
          <ol>
            {data.forbidden_claims.map((claim) => (
              <li key={claim}>{claim}</li>
            ))}
          </ol>
        </article>
      </section>

      <section className="proof-integrity-strip">
        <ShieldCheck size={20} />
        <div>
          <strong>
            Evidence-class separation and artifact
            integrity are product behaviors.
          </strong>
          <span>
            REAL and SIMULATED results remain separate,
            while SHA-256 and semantic-rebuild checks
            bind this view to the frozen source artifacts.
          </span>
        </div>
        <code>{bundle.bundle_id}</code>
      </section>
    </div>
  );
}
