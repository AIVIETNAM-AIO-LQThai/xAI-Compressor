import { useSyncExternalStore } from "react";

/** Whether the Syrus drawer is open, and a question offered from elsewhere in the interface. A prefill is never sent on its own. */
interface UiState {
  open: boolean;
  prefill: string | null;
}

let state: UiState = { open: false, prefill: null };
const listeners = new Set<() => void>();

function set(next: Partial<UiState>) {
  state = { ...state, ...next };
  listeners.forEach((l) => l());
}

export const openSyrus = (prefill: string | null = null) => set({ open: true, prefill });
export const closeSyrus = () => set({ open: false, prefill: null });
export const takePrefill = () => {
  const p = state.prefill;
  if (p !== null) set({ prefill: null });
  return p;
};

export function useSyrusUi(): UiState {
  return useSyncExternalStore(
    (cb) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    () => state,
  );
}
