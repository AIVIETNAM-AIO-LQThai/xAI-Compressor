import { useEffect, useState } from "react";

import { ErrorPanel } from "../components/ErrorPanel";
import { LoadingPanel } from "../components/LoadingPanel";
import { getCase, getManifest } from "../syrus/client";
import { EvidencePane } from "../syrus/EvidencePane";
import { Inspector, type Ref } from "../syrus/Inspector";
import { SyrusPane } from "../syrus/SyrusPane";
import type { CaseFile, Manifest, Mode } from "../syrus/types";

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export function dateLabel(alertTime: string) {
  const [y, m, d] = alertTime.slice(0, 10).split("-").map(Number);
  return `${d} ${MONTHS[m - 1]} ${y}`;
}

interface Props {
  caseId?: string;
  mode: Mode;
  go: (caseId: string, mode: Mode) => void;
}

export function Investigation({ caseId, mode, go }: Props) {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [file, setFile] = useState<CaseFile | null>(null);
  const [error, setError] = useState("");
  const [target, setTarget] = useState<Ref | null>(null);
  const [tab, setTab] = useState<"syrus" | "evidence">("syrus");
  const [controls, setControls] = useState(false);

  useEffect(() => {
    const ac = new AbortController();
    getManifest(ac.signal).then(setManifest, (e: Error) => e.name !== "AbortError" && setError("The case list could not be loaded."));
    return () => ac.abort();
  }, []);

  const cases = manifest ? [...manifest.cases].sort((a, b) => a.alert_time.localeCompare(b.alert_time)) : [];
  const active = caseId && cases.some((c) => c.case_id === caseId) ? caseId : cases[1]?.case_id ?? cases[0]?.case_id;

  useEffect(() => {
    if (!active) return;
    setFile(null);
    setTarget(null);
    const ac = new AbortController();
    getCase(active, mode, ac.signal).then(setFile, (e: Error) => e.name !== "AbortError" && setError("This case could not be loaded."));
    return () => ac.abort();
  }, [active, mode]);

  if (error) return <ErrorPanel error={error} />;
  if (!manifest || !active) return <LoadingPanel label="Loading cases…" />;
  const number = cases.findIndex((c) => c.case_id === active) + 1;

  return (
    <div className="page-shell investigation">
      <header className="inv-header">
        <div>
          <div className="eyebrow">
            INCIDENT INVESTIGATION / INCIDENT {String(number).padStart(2, "0")}
          </div>
          <h1>Why did the detector alert?</h1>
          <p className="muted">
            MetroPT-3 air-production unit · {dateLabel(cases[number - 1].alert_time)} · {mode === "retro" ? "Retrospective review" : "Past-only view"}
          </p>
        </div>
        <span className="badge-outline">Verified detector evidence</span>
      </header>

      <button className="case-toggle" aria-expanded={controls} onClick={() => setControls(!controls)}>
        <span>
          Incident {number} · {mode === "retro" ? "Retrospective" : "Past-only"}
        </span>
        <small>{controls ? "Hide" : "Change"}</small>
      </button>
      <div className={controls ? "inv-controls shown" : "inv-controls"}>
        <div className="segmented" role="group" aria-label="Incident">
          {cases.map((c, i) => (
            <button key={c.case_id} className={c.case_id === active ? "active" : ""} aria-pressed={c.case_id === active} onClick={() => go(c.case_id, mode)}>
              <span>Incident {i + 1}</span>
              <small>{dateLabel(c.alert_time)}</small>
            </button>
          ))}
        </div>
        <div className="segmented compact" role="group" aria-label="Time mode">
          <button className={mode === "asof" ? "active" : ""} aria-pressed={mode === "asof"} onClick={() => go(active, "asof")}>
            Past-only
          </button>
          <button className={mode === "retro" ? "active" : ""} aria-pressed={mode === "retro"} onClick={() => go(active, "retro")}>
            Retrospective
          </button>
        </div>
      </div>

      {!file ? (
        <LoadingPanel label="Loading evidence…" />
      ) : (
        <>
          <div className="mobile-summary" aria-label="Alert summary">
            <span>
              Smoothed <b>{typeof file.records.find((r) => r.id === "alert.smoothed")?.display === "string" ? (file.records.find((r) => r.id === "alert.smoothed")?.display as string) : "n/a"}</b>
            </span>
            <span>
              Above threshold <b>{typeof file.records.find((r) => r.id === "alert.margin")?.display === "string" ? (file.records.find((r) => r.id === "alert.margin")?.display as string) : "n/a"}</b>
            </span>
            <em>Read-only</em>
          </div>
          <div className="mobile-tabs" role="tablist">
            <button role="tab" aria-selected={tab === "syrus"} className={tab === "syrus" ? "active" : ""} onClick={() => setTab("syrus")}>
              Syrus
            </button>
            <button role="tab" aria-selected={tab === "evidence"} className={tab === "evidence" ? "active" : ""} onClick={() => setTab("evidence")}>
              Evidence
            </button>
          </div>
          <div className={`inv-grid show-${tab}`}>
            <EvidencePane c={file} onInspect={setTarget} />
            <SyrusPane key={`${active}:${mode}`} c={file} caseId={active} mode={mode} onInspect={setTarget} />
          </div>
          <Inspector c={file} target={target} onClose={() => setTarget(null)} onOpen={setTarget} />
        </>
      )}
    </div>
  );
}
