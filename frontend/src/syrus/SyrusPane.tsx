import { useEffect, useRef, useState } from "react";
import { Send, Sparkles } from "lucide-react";

import { getHealth } from "./client";
import { refFor, type Ref } from "./Inspector";
import { DEPTHS, SUGGESTIONS, notice, title } from "./labels";
import { ask, cancel, conversationKey, useConversation } from "./store";
import type { Answer, CaseFile, Depth, Health, Mode, TurnOutcome } from "./types";

const MAX_CHARS = 500;

function Countdown({ seconds, onDone }: { seconds: number; onDone: () => void }) {
  const [left, setLeft] = useState(Math.ceil(seconds));
  useEffect(() => {
    if (left <= 0) {
      onDone();
      return;
    }
    const t = setTimeout(() => setLeft(left - 1), 1000);
    return () => clearTimeout(t);
  }, [left, onDone]);
  return <span>{left > 0 ? ` Retry in ${left} s.` : ""}</span>;
}

function AnswerView({ a, ms, c, onOpen, onAsk }: { a: Answer; ms: number; c: CaseFile; onOpen: (r: Ref) => void; onAsk: (q: string) => void }) {
  const added = a.provenance.validator === "passed_with_added_findings";
  // Sources are numbered by first citation and listed once; each claim carries only the numbers of what it rests on.
  const sources = new Map<string, string>();
  for (const b of a.blocks) for (const id of [...b.findings, ...b.records]) if (!sources.has(title(id))) sources.set(title(id), id);
  const list = [...sources.entries()];
  const numberOf = (id: string) => list.findIndex(([t]) => t === title(id)) + 1;
  return (
    <article className="answer" lang={a.lang}>
      <div className="answer-head">
        <strong>Syrus</strong>
        <span>{DEPTHS.find((d) => d.id === a.depth)?.label}</span>
      </div>
      {a.blocks.map((b, i) => (
        <section key={i} className="claim">
          <p>
            {b.text}
            <span className="refs">
              {[...new Set([...b.findings, ...b.records].map(numberOf))].map((n) => (
                <button key={n} className="ref" aria-label={`Source ${n}: ${list[n - 1][0]}`} onClick={() => onOpen(refFor(list[n - 1][1], c))}>
                  {n}
                </button>
              ))}
            </span>
          </p>
          {b.qualifications.map((q) => (
            <p key={q.id} className={`qualification ${q.severity}${q.short ? " short" : ""}`}>
              <span className="note-label">Evidence note</span>
              {q.text}
            </p>
          ))}
        </section>
      ))}
      {a.also_relevant.length > 0 ? (
        <section className="claim also">
          <h4>Also relevant</h4>
          {a.also_relevant.map((x) => (
            <p key={x.finding_id}>
              <button className="link-button" onClick={() => onOpen({ kind: "finding", id: x.finding_id })}>
                {title(x.finding_id)}
              </button>{" "}
              {x.text}
            </p>
          ))}
        </section>
      ) : null}
      {a.narrowed ? (
        <p className="qualification context">
          <span className="note-label">Scope</span>
          {a.narrowed}
        </p>
      ) : null}
      {a.operator_note ? (
        <p className="qualification context">
          <span className="note-label">Your statement</span>
          {a.operator_note}
        </p>
      ) : null}
      {a.next_question ? (
        <div className="next-question">
          <span>Next useful question</span>
          <button onClick={() => onAsk(a.next_question!.text)}>{a.next_question.text}</button>
        </div>
      ) : null}
      {list.length > 0 ? (
        <details className="sources">
          <summary>Sources ({list.length})</summary>
          <ol>
            {list.map(([t, id]) => (
              <li key={id}>
                <button className="link-button" onClick={() => onOpen(refFor(id, c))}>
                  {t}
                </button>
              </li>
            ))}
          </ol>
        </details>
      ) : null}
      <footer className="answer-foot">
        <span className={added ? "status-chip added" : "status-chip passed"}>{added ? "Checked · server added findings" : "Checked against the evidence"}</span>
        {a.refusal === "diagnosis_request" ? <span className="status-chip neutral">No diagnosis claimed</span> : null}
        <span className="foot-note">{(ms / 1000).toFixed(1)} s</span>
      </footer>
    </article>
  );
}

function Problem({ outcome, retryable, onRetry }: { outcome: Exclude<TurnOutcome, { kind: "answer" }>; retryable: boolean; onRetry: () => void }) {
  const n = notice(outcome);
  const codes = outcome.kind === "failed_checks" ? outcome.codes : [];
  const [wait, setWait] = useState(n.retryAfter ?? 0);
  return (
    <div className={`notice ${n.tone}`} role="status">
      <p>
        {n.text}
        {n.retryAfter ? <Countdown seconds={n.retryAfter} onDone={() => setWait(0)} /> : null}
      </p>
      {codes.length > 0 ? <small>Checks that failed: {codes.join(", ")}</small> : null}
      {n.canRetry && retryable ? (
        <button className="secondary-button" onClick={onRetry} disabled={wait > 0}>
          Try again
        </button>
      ) : null}
    </div>
  );
}

export function SyrusPane({ c, caseId, mode, onInspect }: { c: CaseFile; caseId: string; mode: Mode; onInspect: (r: Ref) => void }) {
  const key = conversationKey(caseId, mode);
  const conv = useConversation(key);
  const [draft, setDraft] = useState("");
  const [depth, setDepth] = useState<Depth>("investigation");
  const [health, setHealth] = useState<Health | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const retro = mode === "retro";

  useEffect(() => {
    const ac = new AbortController();
    getHealth(ac.signal).then(setHealth, () => setHealth(null));
    return () => ac.abort();
  }, []);
  useEffect(
    () => () => {
      cancel(key);
    },
    [key],
  );
  useEffect(() => {
    const box = scroller.current;
    if (box) box.scrollTop = box.scrollHeight; // scroll the conversation only, never the page
  }, [conv.turns.length, conv.pending]);

  const send = (text: string) => {
    const message = text.trim();
    if (!message || message.length > MAX_CHARS || conv.pending || !retro) return;
    setDraft("");
    void ask({ caseId, mode, message, depth });
  };
  const last = conv.turns[conv.turns.length - 1];
  const lastUser = [...conv.turns].reverse().find((t) => t.role === "user");
  const highlights = c.findings.filter((f) => f.fires && ["F-MARGIN", "F-EWMA-MEMORY"].includes(f.id));
  const blocked = health?.status === "offline" ? health.reason : null;

  return (
    <section className="syrus-pane" aria-label="Syrus">
      <header className="syrus-head">
        <div className="syrus-mark" aria-hidden>
          <Sparkles size={20} />
        </div>
        <div>
          <strong>Syrus</strong>
          <span>Evidence-backed engineering conversation</span>
        </div>
        <span className="mode-badge">{retro ? "Retrospective" : "Past-only"}</span>
      </header>

      {health?.mock ? <div className="notice warn">Test fixture: answers are scripted for interface checks, not produced by Syrus.</div> : null}

      <div className="depth-tabs" role="group" aria-label="Explanation depth">
        {DEPTHS.map((d) => (
          <button key={d.id} className={depth === d.id ? "active" : ""} aria-pressed={depth === d.id} onClick={() => setDepth(d.id)}>
            {d.label}
          </button>
        ))}
      </div>
      <p className="depth-hint">{DEPTHS.find((d) => d.id === depth)?.hint}</p>

      <div className="transcript" aria-live="polite" ref={scroller}>
        {conv.turns.length === 0 ? (
          <div className="highlights">
            <h3>Verified highlights</h3>
            {highlights.length === 0 ? <p className="muted">No highlight applies to this case.</p> : null}
            {highlights.map((f) => (
              <p key={f.id}>
                {f.text}{" "}
                <button className="link-button" onClick={() => onInspect({ kind: "finding", id: f.id })}>
                  Inspect
                </button>
              </p>
            ))}
            {retro ? (
              <div className="suggestions">
                {SUGGESTIONS.map((q) => (
                  <button key={q} onClick={() => send(q)} disabled={!!blocked}>
                    {q}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}

        {conv.turns.map((t, i) =>
          t.role === "user" ? (
            <div key={i} className="bubble user">
              {t.text}
            </div>
          ) : t.outcome.kind === "answer" ? (
            <AnswerView key={i} a={t.outcome.answer} ms={t.outcome.ms} c={c} onOpen={onInspect} onAsk={send} />
          ) : (
            <Problem key={i} outcome={t.outcome} retryable={t === last && !!lastUser} onRetry={() => lastUser && void ask({ caseId, mode, message: (lastUser as { text: string }).text, depth, retry: true })} />
          ),
        )}
        {conv.pending ? <div className="bubble thinking">Syrus is checking the evidence…</div> : null}
        {conv.endedSession && !conv.pending ? <p className="muted center">The previous session ended; the next question starts a new one.</p> : null}
      </div>

      {blocked ? <div className="notice warn">{notice({ kind: "offline", reason: blocked, retryAfter: null }).text}</div> : null}

      <form
        className="composer"
        onSubmit={(e) => {
          e.preventDefault();
          send(draft);
        }}
      >
        {retro ? (
          <>
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send(draft);
                }
              }}
              placeholder="Ask Syrus about this case…"
              aria-label="Question for Syrus"
              rows={2}
              maxLength={MAX_CHARS}
              disabled={conv.pending}
            />
            <button type="submit" className="send-button" aria-label="Send" disabled={conv.pending || !draft.trim() || !!blocked}>
              <Send size={18} />
            </button>
            <small className="count">
              {draft.length}/{MAX_CHARS}
            </small>
          </>
        ) : (
          <p className="composer-off">Chat opens in Retrospective mode. Past-only mode shows the evidence available at the alert.</p>
        )}
      </form>
    </section>
  );
}
