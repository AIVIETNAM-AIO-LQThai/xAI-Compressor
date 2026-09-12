import { useState } from "react";
import ReactECharts from "echarts-for-react";

import { simulateTwin } from "../api";
import { EvidenceBadge } from "../components/EvidenceBadge";
import { MetricCard } from "../components/MetricCard";
import { PageHeader } from "../components/PageHeader";
import type {
  TwinRequest,
  TwinResponse,
} from "../types";

const DEFAULTS: TwinRequest = {
  initial_pressure_bar_g: 7.0,
  mass_flow_in_kg_s: 0.095,
  demand_mass_flow_kg_s: 0.09,
  leak_mass_flow_kg_s: 0.005,
  duration_seconds: 600,
  timestep_seconds: 1,
};

export function DigitalTwinLab() {
  const [form, setForm] =
    useState<TwinRequest>(DEFAULTS);
  const [result, setResult] =
    useState<TwinResponse | null>(null);
  const [error, setError] = useState("");
  const [running, setRunning] = useState(false);

  async function run() {
    setError("");
    setRunning(true);

    try {
      setResult(await simulateTwin(form));
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Simulation failed.",
      );
    } finally {
      setRunning(false);
    }
  }

  function update(
    key: keyof TwinRequest,
    value: number,
  ) {
    setForm((current) => ({
      ...current,
      [key]: value,
    }));
  }

  const chart = result
    ? {
        backgroundColor: "transparent",
        grid: {
          left: 55,
          right: 24,
          top: 25,
          bottom: 42,
        },
        xAxis: {
          type: "value",
          name: "time (s)",
          nameTextStyle: { color: "#7f96a8" },
          axisLabel: { color: "#7f96a8" },
          axisLine: {
            lineStyle: { color: "#304656" },
          },
          splitLine: {
            lineStyle: {
              color: "rgba(130,160,180,.08)",
            },
          },
        },
        yAxis: {
          type: "value",
          name: "bar(g)",
          nameTextStyle: { color: "#7f96a8" },
          axisLabel: { color: "#7f96a8" },
          axisLine: {
            lineStyle: { color: "#304656" },
          },
          splitLine: {
            lineStyle: {
              color: "rgba(130,160,180,.08)",
            },
          },
        },
        tooltip: { trigger: "axis" },
        series: [
          {
            type: "line",
            showSymbol: false,
            smooth: false,
            data: result.trajectory.map((point) => [
              point.time_seconds,
              point.pressure_bar_g,
            ]),
            lineStyle: {
              width: 2,
              color: "#55d6be",
            },
            areaStyle: {
              color: "rgba(85,214,190,.08)",
            },
            markLine: {
              symbol: "none",
              data: [
                {
                  yAxis: 6.5,
                  lineStyle: { color: "#f59e0b" },
                  label: {
                    formatter: "safety min 6.5",
                    color: "#fbbf24",
                  },
                },
                {
                  yAxis: 7.5,
                  lineStyle: { color: "#f59e0b" },
                  label: {
                    formatter: "safety max 7.5",
                    color: "#fbbf24",
                  },
                },
              ],
            },
          },
        ],
      }
    : null;

  return (
    <div className="page-shell">
      <PageHeader
        eyebrow="DIGITAL TWIN LAB"
        title="Change the physics inputs and watch receiver pressure respond."
        description="This lab uses the lumped isothermal ideal-gas receiver model. Every result here is simulated evidence."
        badge={<EvidenceBadge kind="SIMULATED" />}
      />

      <section className="lab-layout">
        <aside className="panel control-panel">
          <span className="panel-kicker">
            SCENARIO INPUTS
          </span>
          <h2>Receiver experiment</h2>

          {(
            [
              [
                "initial_pressure_bar_g",
                "Initial pressure",
                "bar(g)",
                0.1,
              ],
              [
                "mass_flow_in_kg_s",
                "Compressor inflow",
                "kg/s",
                0.005,
              ],
              [
                "demand_mass_flow_kg_s",
                "Demand",
                "kg/s",
                0.005,
              ],
              [
                "leak_mass_flow_kg_s",
                "Leak",
                "kg/s",
                0.001,
              ],
              [
                "duration_seconds",
                "Duration",
                "s",
                60,
              ],
            ] as const
          ).map(([key, label, unit, step]) => (
            <label className="field" key={key}>
              <span>
                {label}
                <small>{unit}</small>
              </span>
              <input
                type="number"
                value={form[key]}
                step={step}
                min={0}
                onChange={(event) =>
                  update(
                    key,
                    Number(event.target.value),
                  )
                }
              />
            </label>
          ))}

          <button
            className="primary-button"
            disabled={running}
            onClick={run}
          >
            {running
              ? "Simulating..."
              : "Run physics simulation"}
          </button>

          {error ? (
            <div className="inline-error">{error}</div>
          ) : null}
        </aside>

        <div className="lab-results">
          {result ? (
            <>
              <div className="metrics-grid metrics-grid-3">
                <MetricCard
                  label="Final pressure"
                  value={`${result.final_pressure_bar_g.toFixed(
                    3,
                  )} bar(g)`}
                />
                <MetricCard
                  label="Minimum"
                  value={`${result.minimum_pressure_bar_g.toFixed(
                    3,
                  )} bar(g)`}
                  accent={
                    result.minimum_pressure_bar_g < 6.5
                      ? "red"
                      : "lime"
                  }
                />
                <MetricCard
                  label="Maximum"
                  value={`${result.maximum_pressure_bar_g.toFixed(
                    3,
                  )} bar(g)`}
                  accent={
                    result.maximum_pressure_bar_g > 7.5
                      ? "red"
                      : "cyan"
                  }
                />
              </div>

              <article className="panel chart-panel">
                <span className="panel-kicker">
                  PRESSURE TRAJECTORY
                </span>
                <ReactECharts
                  option={chart!}
                  style={{ height: 390 }}
                />
              </article>
            </>
          ) : (
            <article className="panel empty-state">
              <div className="twin-vessel">
                <div className="vessel-fill" />
              </div>
              <h2>Digital twin ready</h2>
              <p>
                Start with balanced flow, then increase
                leak or demand to see how stored
                pressure changes.
              </p>
            </article>
          )}
        </div>
      </section>
    </div>
  );
}
