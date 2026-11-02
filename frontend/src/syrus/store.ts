import { useSyncExternalStore } from "react";

import { createSession, SessionError, sendTurn } from "./client";
import type { Depth, Mode, Turn } from "./types";

/**
 * In-memory conversations, one per case and mode, so that moving between pages keeps a conversation and a different case or mode never
 * sees another's messages or token. Nothing is written to browser storage: a reload ends the session.
 */
export interface Conversation {
  turns: Turn[];
  pending: boolean;
  token: string | null;
  expiresAt: number;
  endedSession: boolean;
}

const EMPTY: Conversation = { turns: [], pending: false, token: null, expiresAt: 0, endedSession: false };
const conversations = new Map<string, Conversation>();
const controllers = new Map<string, AbortController>();
const listeners = new Set<() => void>();

export const conversationKey = (caseId: string, mode: Mode) => `${caseId}:${mode}`;

function put(key: string, patch: Partial<Conversation>) {
  conversations.set(key, { ...(conversations.get(key) ?? EMPTY), ...patch });
  listeners.forEach((l) => l());
}

function get(key: string): Conversation {
  return conversations.get(key) ?? EMPTY;
}

export function useConversation(key: string): Conversation {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => get(key),
  );
}

export function cancel(key: string) {
  controllers.get(key)?.abort();
  controllers.delete(key);
  if (get(key).pending) put(key, { pending: false });
}

interface Ask {
  caseId: string;
  mode: Mode;
  message: string;
  depth: Depth;
  /** Send the last user message again after a failed answer instead of adding a new one. */
  retry?: boolean;
}

export async function ask({ caseId, mode, message, depth, retry }: Ask) {
  const key = conversationKey(caseId, mode);
  const conv = get(key);
  if (conv.pending || mode !== "retro") return;
  let turns = conv.turns;
  if (retry) {
    const last = turns[turns.length - 1];
    if (last?.role === "syrus") turns = turns.slice(0, -1);
  } else {
    turns = [...turns, { role: "user", text: message, at: Date.now() }];
  }
  put(key, { turns, pending: true });
  const controller = new AbortController();
  controllers.set(key, controller);
  const finish = (outcome: Extract<Turn, { role: "syrus" }>["outcome"], patch: Partial<Conversation> = {}) => {
    if (controller.signal.aborted) return;
    controllers.delete(key);
    put(key, { turns: [...get(key).turns, { role: "syrus", outcome, at: Date.now() }], pending: false, ...patch });
  };
  try {
    let { token, expiresAt } = get(key);
    if (!token || Date.now() > expiresAt - 5000) {
      try {
        const s = await createSession(caseId, mode, controller.signal);
        token = s.token;
        expiresAt = Date.now() + s.expires_in * 1000;
        put(key, { token, expiresAt, endedSession: false });
      } catch (e) {
        if ((e as Error).name === "AbortError") throw e;
        if (e instanceof SessionError) return finish({ kind: "error", code: e.code, status: e.status });
        return finish({ kind: "network" });
      }
    }
    const outcome = await sendTurn(token, { message, depth, caseId, mode }, controller.signal);
    if (outcome.kind === "error" && ["session_expired", "token_invalid", "token_missing"].includes(outcome.code)) {
      return finish(outcome, { token: null, expiresAt: 0, endedSession: true });
    }
    finish(outcome);
  } catch (e) {
    if ((e as Error).name !== "AbortError") finish({ kind: "network" });
  }
}
