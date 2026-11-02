import { useEffect, useState } from "react";
import { Sparkles } from "lucide-react";

import { ErrorPanel } from "../components/ErrorPanel";
import { LoadingPanel } from "../components/LoadingPanel";
import { recallIncident, remember, type Selection } from "../selection";
import { EvidencePane } from "../syrus/EvidencePane";
import { Inspector, type Ref } from "../syrus/Inspector";
import { dateLabel, title } from "../syrus/labels";
import { openSyrus } from "../syrus/ui";
import { useCaseFile, useCases, type CaseEntry } from "../syrus/useCase";

interface Props {
  selection: Extract<Selection, { kind: "incident" }> | null;
  go: (s: Selection) => void;
}

/** One selected documented incident. Without a selection it asks the user to choose; it never picks one. */
export function Investigation({ selection, go }: Props) {
  const { cases, error } = useCases();
  const entry = selection && cases?.find((c) => c.caseId === selection.caseId);

  useEffect(() => {
    if (selection && entry) remember(selection);
  }, [selection, entry]);

  if (error) return <ErrorPanel error="The case list could not be loaded." />;
  if (!cases) return <LoadingPanel label="Loading cases…" />;
  if (!selection || !entry) return <Picker cases={cases} go={go} unknown={!!selection} />;
  return <Workspace key={`${selection.caseId}:${selection.mode}`} selection={selection} entry={entry} cases={cases} go={go} />;
}

function Picker({ cases, go, unknown }: { cases: CaseEntry[]; go: (s: Selection) => void; unknown: boolean }) {
  const last = recallIncident();
  const lastEntry = last && cases.find((c) => c.caseId === last.caseId);
  return (
    <div className="page-shell investigation">
      <header className="inv-header">
        <div>
          <div className="eyebrow">INVESTIGATION</div>
          <h1>Choose what to investigate</h1>
          <p className="muted">{unknown ? "That case is not available. " : ""}Four documented incidents have reviewed evidence and Syrus.</p>
        </div>
      </header>
      {last && lastEntry ? (
        <button className="resume" onClick={() => go(last)}>
          Continue with Incident {lastEntry.number} · {last.mode === "retro" ? "Retrospective" : "Past-only"}
        </button>
      ) : null}
      <ul className="pick-list">
        {cases.map((c) => (
          <li key={c.caseId}>
            <button onClick={() => go({ kind: "incident", caseId: c.caseId, mode: "retro" })}>
              <strong>Incident {c.number}</strong>
              <span>{dateLabel(c.alertTime)}</span>
              <small>Alert {c.alertTime.slice(11, 16)} · dataset local time</small>
            </button>
          </li>
        ))}
      </ul>
      <p className="muted pick-note">Alert episodes without a documented incident and arbitrary recorded intervals come with the data explorer.</p>
    </div>
  );
}

function Workspace({ selection, entry, cases, go }: { selection: Extract<Selection, { kind: "incident" }>; entry: CaseEntry; cases: CaseEntry[]; go: (s: Selection) => void }) {
  const { file, error } = useCaseFile(selection.caseId, selection.mode);
  const [target, setTarget] = useState<Ref | null>(null);
  const [controls, setControls] = useState(false);
  const { mode } = selection;
  const ask = (r: Ref) => {
    setTarget(null);
    openSyrus(r.kind === "finding" ? `What does the finding "${title(r.id)}" show for this alert, and what does it not show?` : `What does "${title(r.id)}" show for this alert?`);
  };

  if (error) return <ErrorPanel error="This case could not be loaded." />;
  return (
    <div className="page-shell investigation">
      <header className="inv-header">
        <div>
          <div className="eyebrow">INVESTIGATION / INCIDENT {String(entry.number).padStart(2, "0")}</div>
          <h1>Why did the detector alert?</h1>
          <p className="muted">
            MetroPT-3 air-production unit · {dateLabel(entry.alertTime)} · {mode === "retro" ? "Retrospective review" : "Past-only view"}
          </p>
        </div>
        {mode === "retro" ? (
          <button className="primary-button ask-syrus" onClick={() => openSyrus()}>
            <Sparkles size={16} aria-hidden /> Ask Syrus
          </button>
        ) : (
          <span className="badge-outline">Past-only · chat off</span>
        )}
      </header>

      <button className="case-toggle" aria-expanded={controls} onClick={() => setControls(!controls)}>
        <span>
          Incident {entry.number} · {mode === "retro" ? "Retrospective" : "Past-only"}
        </span>
        <small>{controls ? "Hide" : "Change"}</small>
      </button>
      <a className="all-link" href="#investigate">
        All incidents
      </a>
      <div className={controls ? "inv-controls shown" : "inv-controls"}>
        <div className="segmented" role="group" aria-label="Incident">
          {cases.map((c) => (
            <button key={c.caseId} className={c.caseId === entry.caseId ? "active" : ""} aria-pressed={c.caseId === entry.caseId} onClick={() => go({ kind: "incident", caseId: c.caseId, mode })}>
              <span>Incident {c.number}</span>
              <small>{dateLabel(c.alertTime)}</small>
            </button>
          ))}
        </div>
        <div className="segmented compact" role="group" aria-label="Time mode">
          <button className={mode === "asof" ? "active" : ""} aria-pressed={mode === "asof"} onClick={() => go({ ...selection, mode: "asof" })}>
            Past-only
          </button>
          <button className={mode === "retro" ? "active" : ""} aria-pressed={mode === "retro"} onClick={() => go({ ...selection, mode: "retro" })}>
            Retrospective
          </button>
        </div>
      </div>

      {!file ? (
        <LoadingPanel label="Loading evidence…" />
      ) : (
        <>
          <EvidencePane c={file} onInspect={setTarget} />
          <Inspector c={file} target={target} onClose={() => setTarget(null)} onOpen={setTarget} onAsk={mode === "retro" ? ask : undefined} />
        </>
      )}
    </div>
  );
}
