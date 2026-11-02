import { useEffect, useRef, useState } from "react";
import { Activity, Cpu, LayoutDashboard, Menu, Search, ShieldCheck, X, Zap } from "lucide-react";

import { DigitalTwinLab } from "./pages/DigitalTwinLab";
import { EnergyRecommendation } from "./pages/EnergyRecommendation";
import { IncidentReplay } from "./pages/IncidentReplay";
import { Investigation } from "./pages/Investigation";
import { Overview } from "./pages/Overview";
import { ProofManifest } from "./pages/ProofManifest";
import { hashFor, parse, recall, remember, type Route, type Selection, type Where } from "./selection";
import { SyrusCompanion } from "./syrus/Companion";
import { useSyrusUi } from "./syrus/ui";

const ROUTES: Array<{ id: Route; label: string; icon: typeof Activity }> = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "investigate", label: "Investigation", icon: Search },
  { id: "incident", label: "Detector Replay", icon: Activity },
  { id: "proof", label: "Evidence Proof", icon: ShieldCheck },
  { id: "twin", label: "Simulation Lab", icon: Cpu },
  { id: "recommendation", label: "Advisory", icon: Zap },
];

export default function App() {
  const [where, setWhere] = useState<Where>(() => parse(window.location.hash));
  const [mobileOpen, setMobileOpen] = useState(false);
  const syrus = useSyrusUi();
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

  const open = (hash: string) => {
    window.location.hash = hash;
  };
  const choose = (s: Selection) => {
    remember(s);
    open(hashFor(s));
  };
  // Syrus follows what is on screen; elsewhere it keeps the last incident the user chose in this tab.
  const selection = where.route === "investigate" ? where.selection : recall();

  const content =
    where.route === "investigate" ? (
      <Investigation selection={where.selection} go={choose} />
    ) : where.route === "incident" ? (
      <IncidentReplay />
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
          {ROUTES.map((item) => {
            const Icon = item.icon;
            const target = item.id === "investigate" && selection ? hashFor(selection) : item.id;
            return (
              <button key={item.id} className={where.route === item.id ? "nav-item active" : "nav-item"} aria-current={where.route === item.id ? "page" : undefined} onClick={() => open(target)}>
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <p>Historical replay · advisory only · no equipment control</p>
        </div>
      </aside>

      <main className="main-content" ref={main} tabIndex={-1}>
        <div className="ambient-grid" />
        {content}
      </main>

      <SyrusCompanion selection={selection} choose={choose} />
    </div>
  );
}
