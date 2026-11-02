import { X } from "lucide-react";

import type { Ref } from "./Inspector";
import { useDialog } from "./useDialog";
import { title } from "./labels";
import type { CaseFile } from "./types";

const GROUPS: Array<[string, string[]]> = [
  ["Alert", ["F-MARGIN", "F-EWMA-MEMORY", "F-GAP", "F-TRAIN-RANGE"]],
  ["Evidence", ["F-EVIDENCE-TIMING", "F-CF-NONUNIQUE", "F-CF-PERSIST"]],
  ["Candidates and limits", ["F-CANDIDATES", "F-IDENT", "F-BRIDGE"]],
  ["Retrospective context", ["F-LABEL", "F-TIMING", "F-SELECTION", "F-ALERT-BURDEN", "F-DISCRIMINATION"]],
];

/** Every finding that applies to the case, in full, grouped. Nothing is shortened here. */
export function FindingsDrawer({ c, onClose, onInspect }: { c: CaseFile; onClose: () => void; onInspect: (r: Ref) => void }) {
  const ref = useDialog<HTMLElement>(onClose);
  const byId = new Map(c.findings.map((f) => [f.id, f]));
  const known = new Set(GROUPS.flatMap(([, ids]) => ids));
  const groups: Array<[string, string[]]> = [...GROUPS, ["Other", c.findings.map((f) => f.id).filter((id) => !known.has(id))]];
  const silent = c.findings.filter((f) => !f.fires).length;
  return (
    <div className="inspector-backdrop" onClick={onClose}>
      <aside ref={ref} className="inspector wide" role="dialog" aria-modal="true" aria-label="All verified findings" onClick={(e) => e.stopPropagation()}>
        <header>
          <div>
            <span className="insp-kicker">Verified findings</span>
            <h2>All findings</h2>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="Close findings">
            <X size={18} />
          </button>
        </header>
        {groups.map(([name, ids]) => {
          const rows = ids.map((id) => byId.get(id)).filter((f) => f?.fires);
          if (!rows.length) return null;
          return (
            <section key={name} className="finding-group">
              <h3>{name}</h3>
              {rows.map((f) => (
                <article key={f!.id} className="finding-full">
                  <h4>{title(f!.id)}</h4>
                  <p>{f!.text}</p>
                  <button className="link-button" onClick={() => onInspect({ kind: "finding", id: f!.id })}>
                    Rule and sources
                  </button>
                </article>
              ))}
            </section>
          );
        })}
        {silent > 0 ? <p className="insp-note">{silent} more do not apply to this case.</p> : null}
      </aside>
    </div>
  );
}
