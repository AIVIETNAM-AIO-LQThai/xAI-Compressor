import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";

import { getDetectionEvidence } from "../api";
import { ErrorPanel } from "../components/ErrorPanel";
import { EvidenceBadge } from "../components/EvidenceBadge";
import { LoadingPanel } from "../components/LoadingPanel";
import { MetricCard } from "../components/MetricCard";
import { PageHeader } from "../components/PageHeader";
import type {
  DetectionEvidence,
  DetectionIncident,
} from "../types";

function timingCopy(incident: DetectionIncident) {
  if (incident.timing === "pre_onset") {
    return `${incident.lead_hours?.toFixed(2)} h before onset`;
  }

  return `${incident.delay_hours?.toFixed(2)} h after onset`;
}

export function IncidentReplay() {
  const [data, setData] =
    useState<DetectionEvidence | null>(null);
  const [selectedId, setSelectedId] = useState(1);
  const [error, setError] = useState("");

  useEffect(() => {
    getDetectionEvidence()
      .then(setData)
      .catch((err: Error) => setError(err.message));
  }, []);

  const selected = useMemo(
    () =>
      data?.incidents.find(
        (incident) => incident.id === selectedId,
      ) ?? null,
    [data, selectedId],
  );

  if (error) {
    return <ErrorPanel error={error} />;
  }

  if (!data || !selected) {
    return <LoadingPanel label="Loading incident evidence..." />;
  }

  const chart = {
    backgroundColor: "transparent",
    grid: {
      left: 110,
      right: 26,
      top: 18,
      bottom: 28,
    },
    xAxis: {
      type: "value",
      min: 0,
      max: 1,
      axisLabel: {
        color: "#7f96a8",
        formatter: (value: number) =>
          `${Math.round(value * 100)}%`,
      },
      splitLine: {
        lineStyle: { color: "rgba(130,160,180,.10)" },
      },
    },
    yAxis: {
      type: "category",
      data: selected.top_groups
        .map((item) => item.group)
        .reverse(),
      axisLabel: { color: "#bed0dc" },
      axisLine: { show: false },
      axisTick: { show: false },
    },
    series: [
      {
        type: "bar",
        data: selected.top_groups
          .map((item) => item.share)
          .reverse(),
        barWidth: 14,
        itemStyle: {
          borderRadius: [0, 8, 8, 0],
          color: "#4fd1c5",
        },
      },
    ],
    tooltip: {
      trigger: "axis",
      formatter: (params: any) => {
        const point = params[0];
        return `${point.name}: ${(point.value * 100).toFixed(
          2,
        )}%`;
      },
    },
  };

  return (
    <div className="page-shell">
      <PageHeader
        eyebrow="INCIDENT REPLAY"
        title="See what the detector saw — without pretending it proved root cause."
        description="Each incident uses the same causal EWMA alert pipeline and exact PCA residual contributions. Contributions identify signals associated with anomaly evidence."
        badge={<EvidenceBadge kind="REAL" />}
      />

      <div className="incident-tabs">
        {data.incidents.map((incident) => (
          <button
            key={incident.id}
            className={
              selectedId === incident.id
                ? "incident-tab active"
                : "incident-tab"
            }
            onClick={() => setSelectedId(incident.id)}
          >
            <small>Incident {incident.id}</small>
            <strong>
              {incident.timing === "pre_onset"
                ? "Pre-onset"
                : "Post-onset"}
            </strong>
          </button>
        ))}
      </div>

      <section className="metrics-grid metrics-grid-3">
        <MetricCard
          label="Alert timing"
          value={timingCopy(selected)}
          accent={
            selected.timing === "pre_onset"
              ? "lime"
              : "amber"
          }
        />
        <MetricCard
          label="Smoothed PCA score"
          value={selected.smoothed_score.toLocaleString(
            undefined,
            { maximumFractionDigits: 0 },
          )}
          note={`threshold ${Number(
            data.detector.threshold,
          ).toLocaleString(undefined, {
            maximumFractionDigits: 0,
          })}`}
        />
        <MetricCard
          label="Dominant signal group"
          value={selected.top_groups[0].group}
          note={`${(
            selected.top_groups[0].share * 100
          ).toFixed(1)}% of smoothed alert score`}
        />
      </section>

      <section className="split-grid">
        <article className="panel">
          <span className="panel-kicker">
            SIGNAL CONTRIBUTIONS
          </span>
          <h2>Operator-level groups</h2>
          <ReactECharts
            option={chart}
            style={{ height: 330 }}
          />
        </article>

        <article className="panel">
          <span className="panel-kicker">
            EVENT CONTEXT
          </span>
          <h2>Replay metadata</h2>

          <dl className="details-list">
            <div>
              <dt>Documented onset</dt>
              <dd>{selected.incident_start}</dd>
            </div>
            <div>
              <dt>Explanation timestamp</dt>
              <dd>{selected.explanation_timestamp}</dd>
            </div>
            <div>
              <dt>Detection relationship</dt>
              <dd>{timingCopy(selected)}</dd>
            </div>
            <div>
              <dt>Causal root-cause claim</dt>
              <dd className="negative">No</dd>
            </div>
          </dl>

          <div className="callout callout-neutral">
            <strong>Interpretation boundary</strong>
            <p>{data.explanation_basis}</p>
          </div>
        </article>
      </section>
    </div>
  );
}
