import { useEffect, useState } from "react";
import { Activity, Cpu, LayoutDashboard, Menu, Search, ShieldCheck, X, Zap } from "lucide-react";

import { DigitalTwinLab } from "./pages/DigitalTwinLab";
import { EnergyRecommendation } from "./pages/EnergyRecommendation";
import { IncidentReplay } from "./pages/IncidentReplay";
import { Investigation } from "./pages/Investigation";
import { Overview } from "./pages/Overview";
import { ProofManifest } from "./pages/ProofManifest";
import type { Mode } from "./syrus/types";

type Route = "overview" | "investigation" | "incident" | "proof" | "twin" | "recommendation";

const ROUTES: Array<{ id: Route; label: string; icon: typeof Activity }> = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "investigation", label: "Investigation", icon: Search },
  { id: "incident", label: "Detector Replay", icon: Activity },
  { id: "proof", label: "Evidence Proof", icon: ShieldCheck },
  { id: "twin", label: "Simulation Lab", icon: Cpu },
  { id: "recommendation", label: "Advisory", icon: Zap },
];

interface Where {
  route: Route;
  caseId?: string;
  mode: Mode;
}

/** `#investigation/<case>/<mode>`; anything unknown falls back to the overview. */
function parse(hash: string): Where {
  const [head, caseId, mode] = hash.replace(/^#/, "").split("/");
  const route = ROUTES.find((r) => r.id === head)?.id ?? "overview";
  return { route, caseId: route === "investigation" && /^c-[a-z0-9]+$/.test(caseId ?? "") ? caseId : undefined, mode: mode === "asof" ? "asof" : "retro" };
}

export default function App() {
  const [where, setWhere] = useState<Where>(() => parse(window.location.hash));
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const onHash = () => {
      setWhere(parse(window.location.hash));
      setMobileOpen(false);
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const open = (hash: string) => {
    window.location.hash = hash;
  };

  const content =
    where.route === "investigation" ? (
      <Investigation caseId={where.caseId} mode={where.mode} go={(c, m) => open(`investigation/${c}/${m}`)} />
    ) : where.route === "incident" ? (
      <IncidentReplay />
    ) : where.route === "twin" ? (
      <DigitalTwinLab />
    ) : where.route === "recommendation" ? (
      <EnergyRecommendation />
    ) : where.route === "proof" ? (
      <ProofManifest />
    ) : (
      <Overview open={open} />
    );

  return (
    <div className="app">
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
          {ROUTES.map((item) => {
            const Icon = item.icon;
            return (
              <button key={item.id} className={where.route === item.id ? "nav-item active" : "nav-item"} aria-current={where.route === item.id ? "page" : undefined} onClick={() => open(item.id)}>
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <p>Historical replay · advisory only · no equipment control</p>
          <p className="syrus-line">
            <b>Syrus</b> · evidence-backed conversation
          </p>
        </div>
      </aside>

      <main className="main-content">
        <div className="ambient-grid" />
        {content}
      </main>
    </div>
  );
}
