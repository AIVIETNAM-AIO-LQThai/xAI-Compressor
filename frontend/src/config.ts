import { useEffect, useState } from "react";

import { STATIC, staticUrl } from "./staticData";

/** What this deployment serves, from the server's /config. Anything missing reads as not served. */
export interface DeploymentConfig {
  mode: "local" | "preview" | "public" | "showcase";
  exported_at?: string;
  features: { syrus_chat: boolean; lab: "live" | "replay" | "off"; twin: boolean; optimization: boolean };
}

const BASE: string = import.meta.env?.VITE_API_BASE_URL ?? "";
const LOCKED: DeploymentConfig = { mode: "public", features: { syrus_chat: false, lab: "off", twin: false, optimization: false } };
let once: Promise<DeploymentConfig> | null = null;

export function useConfig(): DeploymentConfig | null {
  const [config, setConfig] = useState<DeploymentConfig | null>(null);
  useEffect(() => {
    once ??= fetch(STATIC ? staticUrl("/config") : `${BASE}/config`)
      .then((r) => (r.ok ? (r.json() as Promise<DeploymentConfig>) : LOCKED))
      .catch(() => LOCKED);
    let live = true;
    once.then((c) => live && setConfig(c));
    return () => {
      live = false;
    };
  }, []);
  return config;
}
