import { X } from "lucide-react";

import { groupName, title } from "./labels";
import { useDialog } from "./useDialog";
import type { CaseFile, CaseRecord } from "./types";

export interface Ref {
  kind: "record" | "finding" | "test";
  id: string;
}

export function refFor(id: string, c: CaseFile): Ref {
  if (id.startsWith("F-")) return { kind: "finding", id };
  if (id.startsWith("t_")) return { kind: "test", id };
  return { kind: "record", id };
}

export function Values({ value }: { value: CaseRecord["display"] }) {
  if (typeof value === "string") return <p className="insp-text">{value}</p>;
  return (
    <dl className="insp-values">
      {Object.entries(value).map(([k, v]) => (
        <div key={k}>
          <dt>{groupName(k)}</dt>
          <dd>{typeof v === "object" ? JSON.stringify(v) : String(v)}</dd>
        </div>
      ))}
    </dl>
  );
}

function Sources({ record }: { record: CaseRecord }) {
  if (!record.source.length) return <p className="insp-note">No file source. This statement describes what is not loaded.</p>;
  return (
    <ul className="insp-sources">
      {record.source.map((s, i) => (
        <li key={i}>
          <code>{s.file}</code>
          {s.path || s.symbol ? <span> {s.path ?? s.symbol}</span> : null}
          {s.sha256 ? <small> sha256 {s.sha256.slice(0, 12)}</small> : null}
        </li>
      ))}
    </ul>
  );
}

/** Side panel on desktop, bottom sheet on a phone. Shows only what the loaded case file contains. */
export function Inspector({ c, target, onClose, onOpen }: { c: CaseFile; target: Ref | null; onClose: () => void; onOpen: (r: Ref) => void }) {
  return target ? <InspectorBody c={c} target={target} onClose={onClose} onOpen={onOpen} /> : null;
}

function InspectorBody({ c, target, onClose, onOpen }: { c: CaseFile; target: Ref; onClose: () => void; onOpen: (r: Ref) => void }) {
  const dialog = useDialog<HTMLElement>(onClose);
  const record = target.kind === "record" ? c.records.find((r) => r.id === target.id) : undefined;
  const finding = target.kind === "finding" ? c.findings.find((f) => f.id === target.id) : undefined;
  const test = target.kind === "test" ? c.tests.find((t) => t.id === target.id) : undefined;
  const missing = !record && !finding && !test;

  return (
    <div className="inspector-backdrop" onClick={onClose}>
      <aside ref={dialog} className="inspector" role="dialog" aria-modal="true" aria-label="Evidence inspector" onClick={(e) => e.stopPropagation()}>
        <header>
          <div>
            <span className="insp-kicker">{target.kind === "finding" ? "Verified finding" : target.kind === "test" ? "Discriminating test" : "Record"}</span>
            <h2>{title(target.id)}</h2>
            <code>{target.id}</code>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="Close inspector">
            <X size={18} />
          </button>
        </header>

        {missing ? <p className="insp-note">This item is not part of the {c.mode === "asof" ? "past-only" : "retrospective"} case file.</p> : null}

        {record ? (
          <>
            <Values value={record.display} />
            {record.caveat ? <p className="insp-caveat">{record.caveat}</p> : null}
            <h3>Source</h3>
            <Sources record={record} />
          </>
        ) : null}

        {finding ? (
          <>
            <p className="insp-text">{finding.text}</p>
            {finding.detail ? <p className="insp-caveat">{finding.detail}</p> : null}
            {!finding.fires ? <p className="insp-note">This finding does not apply to this case.</p> : null}
            <h3>Rule</h3>
            <p className="insp-note">{finding.rule}</p>
            <h3>Built from</h3>
            <ul className="insp-inputs">
              {finding.inputs
                .filter((i) => c.records.some((r) => r.id === i))
                .map((i) => (
                  <li key={i}>
                    <button className="link-button" onClick={() => onOpen({ kind: "record", id: i })}>
                      {title(i)}
                    </button>
                  </li>
                ))}
            </ul>
          </>
        ) : null}

        {test ? (
          <>
            <p className="insp-text">{test.follow_up.en}</p>
            <dl className="insp-values">
              <div>
                <dt>Status</dt>
                <dd>
                  {test.maturity_text.en}; {test.result_text.en}
                </dd>
              </div>
              {test.observations.map((o, i) => (
                <div key={i}>
                  <dt>Needs</dt>
                  <dd>
                    {o.text.en}
                    <br />
                    <small>{o.availability_text.en}</small>
                  </dd>
                </div>
              ))}
              <div>
                <dt>Compares</dt>
                <dd>{test.compares.map((p) => p.map(groupName).join(" vs ")).join("; ")}</dd>
              </div>
            </dl>
          </>
        ) : null}
      </aside>
    </div>
  );
}
