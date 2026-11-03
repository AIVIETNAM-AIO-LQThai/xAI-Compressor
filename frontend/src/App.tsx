import { useEffect, useRef, useState } from "react";
import { Activity, Cpu, Database, FlaskConical, LayoutDashboard, Menu, Search, ShieldCheck, X, Zap } from "lucide-react";

import { useConfig, type DeploymentConfig } from "./config";
import { Explorer } from "./pages/Explorer";
import { DigitalTwinLab } from "./pages/DigitalTwinLab";
import { EnergyRecommendation } from "./pages/EnergyRecommendation";
import { IncidentReplay } from "./pages/IncidentReplay";
import { Investigation } from "./pages/Investigation";
import { Overview } from "./pages/Overview";
import { ProofManifest } from "./pages/ProofManifest";
import { ScenarioLab } from "./pages/ScenarioLab";
import { hashFor, parse, recall, recallIncident, recallView, remember, type Route, type Selection, type Where } from "./selection";
import { SyrusCompanion } from "./syrus/Companion";
import { useSyrusUi } from "./syrus/ui";

// The older Simulation Lab is reached by link only: the Scenario Lab's generic twin replaces it.
const NAV_HIDDEN: Route[] = ["twin"];
const SERVED: Partial<Record<Route, (c: DeploymentConfig) => boolean>> = {
  lab: (c) => c.features.lab !== "off",
  twin: (c) => c.features.twin,
  recommendation: (c) => c.features.optimization,
};
const MODE_NAME: Record<DeploymentConfig["mode"], string> = { local: "local copy", preview: "preview", public: "public site", showcase: "showcase" };

const ROUTES: Array<{ id: Route; label: string; icon: typeof Activity }> = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "investigate", label: "Investigation", icon: Search },
  { id: "explore", label: "Data Explorer", icon: Database },
  { id: "incident", label: "Detector Replay", icon: Activity },
  { id: "proof", label: "Evidence Proof", icon: ShieldCheck },
  { id: "lab", label: "Scenario Lab", icon: FlaskConical },
  { id: "twin", label: "Simulation Lab", icon: Cpu },
  { id: "recommendation", label: "Advisory", icon: Zap },
];

export default function App() {
  const [where, setWhere] = useState<Where>(() => parse(window.location.hash));
  const [mobileOpen, setMobileOpen] = useState(false);
  const syrus = useSyrusUi();
  const config = useConfig();
  const served = (id: Route) => !config || (SERVED[id]?.(config) ?? true);
  const main = useRef<HTMLElement>(null);

  useEffect(() => {
    // The earlier `#investigation/<case>/<mode>` links still work; the address bar is rewritten to the current form.
    const current = () => {
      const w = parse(window.location.hash);
      if (window.location.hash.startsWith("#investigation")) history.replaceState(null, "", w.selection ? `#${hashFor(w.selection)}` : "#investigate");
      return w;
    };
    current();
    const onHash = () => {
      setWhere(current());
      setMobileOpen(false);
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  // Each page names itself in the tab and to screen readers.
  useEffect(() => {
    document.title = `${ROUTES.find((r) => r.id === where.route)?.label ?? "Overview"} · AeroXAI`;
  }, [where.route]);

  const open = (hash: string) => {
    window.location.hash = hash;
  };
  const choose = (s: Selection) => {
    remember(s);
    open(hashFor(s));
  };
  // Syrus follows what is on screen; on other pages it keeps the last thing the user chose in this tab.
  const selection = where.route === "investigate" || where.route === "explore" ? where.selection : recall();
  const incident = selection?.kind === "incident" ? selection : recallIncident();
  const view = selection && selection.kind !== "incident" ? selection : recallView();
  const target = (id: Route) => (id === "investigate" && incident ? hashFor(incident) : id === "explore" && view ? hashFor(view) : id);

  const content = !served(where.route) ? (
    <NotServed mode={config ? MODE_NAME[config.mode] : "deployment"} open={open} />
  ) : where.route === "investigate" ? (
      <Investigation selection={where.selection?.kind === "interval" ? null : where.selection} go={choose} />
    ) : where.route === "explore" ? (
      <Explorer selection={where.selection?.kind === "incident" ? null : where.selection} go={choose} />
    ) : where.route === "incident" ? (
      <IncidentReplay />
    ) : where.route === "lab" ? (
      <ScenarioLab />
    ) : where.route === "twin" ? (
      <DigitalTwinLab />
    ) : where.route === "recommendation" ? (
      <EnergyRecommendation />
    ) : where.route === "proof" ? (
      <ProofManifest />
    ) : (
      <Overview open={open} choose={choose} />
    );

  return (
    <div className={syrus.open ? "app syrus-open" : "app"}>
      <button className="skip-link" onClick={() => main.current?.focus()}>
        Skip to content
      </button>
      <button className="mobile-menu-button" onClick={() => setMobileOpen(!mobileOpen)} aria-label="Toggle navigation" aria-expanded={mobileOpen}>
        {mobileOpen ? <X /> : <Menu />}
      </button>

      <aside className={mobileOpen ? "sidebar sidebar-open" : "sidebar"}>
        <div className="brand">
          <div>
            <strong>
              Aero<span>XAI</span>
            </strong>
            <span>INDUSTRIAL INTELLIGENCE</span>
          </div>
        </div>

        <nav aria-label="Main">
          {ROUTES.filter((item) => !NAV_HIDDEN.includes(item.id) && served(item.id)).map((item) => {
            const Icon = item.icon;
            return (
              <button key={item.id} className={where.route === item.id ? "nav-item active" : "nav-item"} aria-current={where.route === item.id ? "page" : undefined} onClick={() => open(target(item.id))}>
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <p>Historical replay · advisory only · no equipment control</p>
          <p className="credit">
            Data:{" "}
            <a href="https://doi.org/10.24432/C5VW3R" target="_blank" rel="noreferrer">
              MetroPT-3
            </a>{" "}
            (Davari, Veloso, Ribeiro &amp; Gama, 2021), UCI, CC BY 4.0
          </p>
        </div>
      </aside>

      <main className="main-content" ref={main} tabIndex={-1}>
        <div className="ambient-grid" />
        {content}
      </main>

      <SyrusCompanion selection={selection} choose={choose} simulation={where.route === "lab" || where.route === "twin"} remembered={where.route !== "investigate" && where.route !== "explore"} />
    </div>
  );
}

function NotServed({ mode, open }: { mode: string; open: (hash: string) => void }) {
  return (
    <div className="page-shell">
      <div className="eyebrow">NOT IN THIS {mode.toUpperCase()}</div>
      <h1>This page is not part of this {mode}</h1>
      <p className="muted">It runs simulation or optimisation on the server, which this {mode} does not serve.</p>
      <button className="secondary-button" onClick={() => open("overview")}>
        Go to the Overview
      </button>
    </div>
  );
}
