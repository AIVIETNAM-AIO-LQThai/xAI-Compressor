import { useEffect, useState } from "react";
import {
  Activity,
  Cpu,
  Gauge,
  LayoutDashboard,
  Menu,
  X,
  Zap,
} from "lucide-react";

import { DigitalTwinLab } from "./pages/DigitalTwinLab";
import { EnergyRecommendation } from "./pages/EnergyRecommendation";
import { IncidentReplay } from "./pages/IncidentReplay";
import { Overview } from "./pages/Overview";

type Route =
  | "overview"
  | "incident"
  | "twin"
  | "recommendation";

const ROUTES: Array<{
  id: Route;
  label: string;
  icon: typeof Activity;
}> = [
  {
    id: "overview",
    label: "Overview",
    icon: LayoutDashboard,
  },
  {
    id: "incident",
    label: "Incident Replay",
    icon: Activity,
  },
  {
    id: "twin",
    label: "Digital Twin Lab",
    icon: Cpu,
  },
  {
    id: "recommendation",
    label: "Energy Recommendation",
    icon: Zap,
  },
];

function routeFromHash(): Route {
  const hash = window.location.hash.slice(1);

  return ROUTES.some((route) => route.id === hash)
    ? (hash as Route)
    : "overview";
}

export default function App() {
  const [route, setRoute] = useState<Route>(
    routeFromHash,
  );
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const onHash = () => {
      setRoute(routeFromHash());
      setMobileOpen(false);
    };

    window.addEventListener("hashchange", onHash);

    return () =>
      window.removeEventListener(
        "hashchange",
        onHash,
      );
  }, []);

  function navigate(next: Route) {
    window.location.hash = next;
  }

  const content =
    route === "incident" ? (
      <IncidentReplay />
    ) : route === "twin" ? (
      <DigitalTwinLab />
    ) : route === "recommendation" ? (
      <EnergyRecommendation />
    ) : (
      <Overview />
    );

  return (
    <div className="app">
      <button
        className="mobile-menu-button"
        onClick={() => setMobileOpen(!mobileOpen)}
        aria-label="Toggle navigation"
      >
        {mobileOpen ? <X /> : <Menu />}
      </button>

      <aside
        className={
          mobileOpen
            ? "sidebar sidebar-open"
            : "sidebar"
        }
      >
        <div className="brand">
          <div className="brand-mark">
            <Gauge size={23} />
          </div>
          <div>
            <strong>AeroXAI</strong>
            <span>ENERGY COPILOT</span>
          </div>
        </div>

        <div className="system-live">
          <i />
          <span>ADVISORY MODE</span>
        </div>

        <nav>
          {ROUTES.map((item) => {
            const Icon = item.icon;

            return (
              <button
                key={item.id}
                className={
                  route === item.id
                    ? "nav-item active"
                    : "nav-item"
                }
                onClick={() => navigate(item.id)}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-footer">
          <div className="tiny-grid">
            <span>CTRL</span>
            <strong>NO WRITE</strong>
            <span>MODEL</span>
            <strong>EXPLAINED</strong>
          </div>
          <p>
            Vendor-agnostic prototype · ESIC 2026
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
