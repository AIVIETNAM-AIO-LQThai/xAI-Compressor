import { useEffect, useState } from "react";

import { getCase, getManifest } from "./client";
import type { CaseFile, Manifest, Mode } from "./types";

// Case files are integrity-checked and immutable on the server, so one fetch per case and mode serves every view in this tab.
const files = new Map<string, Promise<CaseFile>>();
let manifest: Promise<Manifest> | null = null;

export interface CaseEntry {
  caseId: string;
  number: number;
  alertTime: string;
}

export function useCases(): { cases: CaseEntry[] | null; error: boolean } {
  const [cases, setCases] = useState<CaseEntry[] | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    manifest ??= getManifest();
    let live = true;
    manifest.then(
      (m) =>
        live &&
        setCases(
          [...m.cases]
            .sort((a, b) => a.alert_time.localeCompare(b.alert_time))
            .map((c, i) => ({ caseId: c.case_id, number: i + 1, alertTime: c.alert_time })),
        ),
      () => {
        manifest = null;
        if (live) setError(true);
      },
    );
    return () => {
      live = false;
    };
  }, []);
  return { cases, error };
}

export function useCaseFile(caseId: string | undefined, mode: Mode): { file: CaseFile | null; error: boolean } {
  const [state, setState] = useState<{ key: string; file: CaseFile | null; error: boolean }>({ key: "", file: null, error: false });
  const key = caseId ? `${caseId}:${mode}` : "";
  useEffect(() => {
    if (!caseId) return;
    if (!files.has(key)) files.set(key, getCase(caseId, mode));
    let live = true;
    files.get(key)!.then(
      (file) => live && setState({ key, file, error: false }),
      () => {
        files.delete(key);
        if (live) setState({ key, file: null, error: true });
      },
    );
    return () => {
      live = false;
    };
  }, [key, caseId, mode]);
  // Never hand back the previous case's file while the new one loads.
  return state.key === key ? { file: state.file, error: state.error } : { file: null, error: false };
}
