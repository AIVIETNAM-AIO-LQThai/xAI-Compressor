import { useEffect, useRef, useState } from "react";
import { Sparkles, X } from "lucide-react";

import type { Selection } from "../selection";
import { Inspector, type Ref } from "./Inspector";
import { dateLabel } from "./labels";
import { SyrusPane } from "./SyrusPane";
import { closeSyrus, openSyrus, useSyrusUi } from "./ui";
import { useCaseFile, useCases } from "./useCase";

/**
 * Syrus as a companion: a fixed launcher, and a drawer that talks about the selected incident only. Its context is the selection's
 * case and mode, never a default; without a selection it asks the user to choose one.
 */
export function SyrusCompanion({ selection, choose }: { selection: Selection | null; choose: (s: Selection) => void }) {
  const ui = useSyrusUi();
  const { cases } = useCases();
  const incident = selection?.kind === "incident" ? selection : null;
  const entry = incident && cases?.find((c) => c.caseId === incident.caseId);
  const other = selection && selection.kind !== "incident" ? selection : null;
  const context = entry ? `Incident ${entry.number} · ${incident!.mode === "retro" ? "Retrospective" : "Past-only"}` : other ? (other.kind === "episode" ? "Alert episode · no case file" : "Interval · no case file") : "No incident selected";

  // Focus moves into the drawer when it opens and back to the launcher when it closes. An offered question focuses the composer instead.
  const launcher = useRef<HTMLButtonElement>(null);
  const close = useRef<HTMLButtonElement>(null);
  const was = useRef(ui.open);
  useEffect(() => {
    if (ui.open && !was.current && ui.prefill === null) close.current?.focus();
    if (!ui.open && was.current) launcher.current?.focus();
    was.current = ui.open;
  }, [ui.open]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!ui.open) return;
    const key = (e: KeyboardEvent) => e.key === "Escape" && closeSyrus();
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [ui.open]);

  if (!ui.open) {
    return (
      <button ref={launcher} className="syrus-launcher" onClick={() => openSyrus()} aria-label={`Open Syrus. Context: ${context}`}>
        <span className="syrus-mark" aria-hidden>
          <Sparkles size={20} />
        </span>
        <span className="launcher-text">
          <b>Syrus</b>
          <small>{context}</small>
        </span>
      </button>
    );
  }

  return (
    <aside className="syrus-drawer" role="dialog" aria-modal="false" aria-label="Syrus">
      <header className="drawer-head">
        <span className="syrus-mark" aria-hidden>
          <Sparkles size={20} />
        </span>
        <div>
          <strong>Syrus</strong>
          <span>
            {entry ? `Incident ${entry.number} · ${dateLabel(entry.alertTime)} · ${incident!.mode === "retro" ? "Retrospective" : "Past-only"}` : context}
          </span>
        </div>
        <button ref={close} className="icon-button" onClick={closeSyrus} aria-label="Close Syrus">
          <X size={18} />
        </button>
      </header>
      {incident && entry ? (
        <Conversation key={`${incident.caseId}:${incident.mode}`} selection={incident} />
      ) : (
        <div className="drawer-empty">
          {other ? (
            <p>
              Syrus is off for this {other.kind === "episode" ? "alert episode" : "interval"}: it has no reviewed case file, and it never borrows another incident's findings. The charts and values on the page are
              the evidence. To talk about a documented incident, choose one:
            </p>
          ) : (
            <p>Syrus answers about one documented incident at a time. Choose one:</p>
          )}
          <ul>
            {cases?.map((c) => (
              <li key={c.caseId}>
                <button onClick={() => choose({ kind: "incident", caseId: c.caseId, mode: "retro" })}>
                  <b>Incident {c.number}</b>
                  <small>{dateLabel(c.alertTime)} · alert {c.alertTime.slice(11, 16)}</small>
                </button>
              </li>
            ))}
          </ul>
          {other ? null : <p className="muted">Alert episodes without a documented incident, recorded intervals and simulations have no reviewed Syrus contract yet.</p>}
        </div>
      )}
    </aside>
  );
}

function Conversation({ selection }: { selection: Extract<Selection, { kind: "incident" }> }) {
  const { file, error } = useCaseFile(selection.caseId, selection.mode);
  const [target, setTarget] = useState<Ref | null>(null);
  if (error) return <p className="notice error">This case could not be loaded.</p>;
  if (!file) return <p className="muted drawer-loading">Loading the case…</p>;
  return (
    <>
      <SyrusPane c={file} caseId={selection.caseId} mode={selection.mode} onInspect={setTarget} />
      <Inspector c={file} target={target} onClose={() => setTarget(null)} onOpen={setTarget} />
    </>
  );
}
