import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ShieldCheck,
} from "lucide-react";

import { getDetectionEvidence } from "../api";
import { ErrorPanel } from "../components/ErrorPanel";
import { EvidenceBadge } from "../components/EvidenceBadge";
import { LoadingPanel } from "../components/LoadingPanel";
import { MetricCard } from "../components/MetricCard";
import { PageHeader } from "../components/PageHeader";
import type {
  CounterfactualRepair,
  DetectionEvidence,
  DetectionIncident,
  VerificationIncident,
} from "../types";

function timingCopy(incident: DetectionIncident) {
  if (incident.timing === "pre_onset") {
    return `${incident.lead_hours?.toFixed(2)} h before onset`;
  }

  return `${incident.delay_hours?.toFixed(2)} h after onset`;
}

function percent(value: number, digits = 1) {
  return `${(value * 100).toFixed(digits)}%`;
}

function score(value: number) {
  return value.toLocaleString(undefined, {
    maximumFractionDigits: 0,
  });
}

function topRepair(
  incident: VerificationIncident,
): CounterfactualRepair | null {
  const topGroup =
    incident.ranked_groups[0]?.group;

  if (!topGroup) {
    return null;
  }

  return (
    incident.single_group_repairs.find(
      (repair) =>
        repair.repaired_groups.length === 1 &&
        repair.repaired_groups[0] === topGroup,
    ) ?? null
  );
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

  const verified = useMemo(
    () =>
      data?.verification.incidents.find(
        (incident) => incident.id === selectedId,
      ) ?? null,
    [data, selectedId],
  );

  if (error) {
    return <ErrorPanel error={error} />;
  }

  if (!data || !selected || !verified) {
    return (
      <LoadingPanel label="Loading verified incident evidence..." />
    );
  }

  const ranked = verified.ranked_groups;
  const displayedGroups = ranked.slice(0, 8);
  const dominant = ranked[0];
  const repair = topRepair(verified);

  const topThreeShare = ranked
    .slice(0, 3)
    .reduce(
      (total, item) => total + item.share,
      0,
    );

  const contributionChart = {
    backgroundColor: "transparent",
    grid: {
      left: 125,
      right: 25,
      top: 18,
      bottom: 28,
      containLabel: true,
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
        lineStyle: {
          color: "rgba(130,160,180,.10)",
        },
      },
    },
    yAxis: {
      type: "category",
      data: displayedGroups
        .map((item) => item.group)
        .reverse(),
      axisLabel: {
        color: "#bed0dc",
      },
      axisLine: { show: false },
      axisTick: { show: false },
    },
    series: [
      {
        type: "bar",
        data: displayedGroups
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

        return `${point.name}: ${(
          point.value * 100
        ).toFixed(3)}% of alert score`;
      },
    },
  };

  const temporalGroups =
    verified.temporal_evidence.top_groups;

  const temporalRows =
    verified.temporal_evidence.rows;

  const heatmapData = temporalRows.flatMap(
    (row, xIndex) =>
      temporalGroups.map((group, yIndex) => [
        xIndex,
        yIndex,
        row.percentiles[group] ?? 0,
      ]),
  );

  const temporalChart = {
    backgroundColor: "transparent",
    grid: {
      left: 125,
      right: 25,
      top: 18,
      bottom: 62,
      containLabel: true,
    },
    xAxis: {
      type: "category",
      data: temporalRows.map((row) =>
        new Date(row.timestamp).toLocaleTimeString(
          [],
          {
            hour: "2-digit",
            minute: "2-digit",
          },
        ),
      ),
      axisLabel: {
        color: "#7f96a8",
        rotate: 40,
      },
      axisLine: {
        lineStyle: {
          color: "#304656",
        },
      },
    },
    yAxis: {
      type: "category",
      data: temporalGroups,
      axisLabel: {
        color: "#bed0dc",
      },
      axisLine: { show: false },
      axisTick: { show: false },
    },
    visualMap: {
      min: 0,
      max: 1,
      orient: "horizontal",
      left: "center",
      bottom: 0,
      text: [
        "unusual",
        "normal",
      ],
      textStyle: {
        color: "#7f96a8",
        fontSize: 10,
      },
      calculable: false,
      inRange: {
        color: [
          "#10222f",
          "#24505b",
          "#3b8d83",
          "#8de26b",
          "#fbbf24",
        ],
      },
    },
    tooltip: {
      formatter: (params: any) => {
        const [x, y, value] =
          params.value;

        return [
          temporalGroups[y],
          temporalRows[x]?.timestamp,
          `${(value * 100).toFixed(1)}th percentile`,
        ].join("<br/>");
      },
    },
    series: [
      {
        type: "heatmap",
        data: heatmapData,
        emphasis: {
          itemStyle: {
            borderColor: "#e8f0f5",
            borderWidth: 1,
          },
        },
      },
    ],
  };

  return (
    <div className="page-shell">
      <PageHeader
        eyebrow="VERIFIED INCIDENT REPLAY"
        title="See what drove the alert — then test whether the detector actually depended on it."
        description="AeroXAI combines exact PCA attribution, calibration-context normality, causal temporal evidence, and model-space counterfactual verification. None of these establish physical root cause."
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
            onClick={() =>
              setSelectedId(incident.id)
            }
          >
            <small>
              Incident {incident.id}
            </small>
            <strong>
              {incident.timing === "pre_onset"
                ? "Pre-onset"
                : "Post-onset"}
            </strong>
          </button>
        ))}
      </div>

      <section className="metrics-grid">
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
          label="Dominant evidence"
          value={dominant.group}
          note={`${percent(
            dominant.share,
          )} of smoothed alert score`}
        />

        <MetricCard
          label="Normality context"
          value={`${(
            dominant.calibration_percentile * 100
          ).toFixed(2)}th`}
          note="percentile of calibration contribution distribution"
          accent="cyan"
        />

        <MetricCard
          label="Counterfactual result"
          value={
            repair?.alert_cleared
              ? "Alert clears"
              : "Alert remains"
          }
          note={
            repair
              ? `${score(
                  repair.original_smoothed_score,
                )} → ${score(
                  repair.repaired_smoothed_score,
                )}`
              : "repair evidence unavailable"
          }
          accent={
            repair?.alert_cleared
              ? "lime"
              : "amber"
          }
        />
      </section>

      <section className="xai-verification-grid">
        <article className="panel">
          <div className="panel-title-row">
            <div>
              <span className="panel-kicker">
                1 · WHAT CONTRIBUTED?
              </span>
              <h2>
                Operator-level signal groups
              </h2>
            </div>

            <Activity size={24} />
          </div>

          <ReactECharts
            option={contributionChart}
            style={{ height: 350 }}
          />

          <div className="xai-footnote">
            All {ranked.length} signal groups were
            evaluated. The chart shows the eight
            largest. The top three account for{" "}
            <strong>
              {percent(topThreeShare, 2)}
            </strong>{" "}
            of this alert score.
          </div>
        </article>

        <article className="panel">
          <span className="panel-kicker">
            2 · HOW UNUSUAL?
          </span>
          <h2>
            Contribution vs calibration
          </h2>

          <div className="xai-group-list">
            {ranked.slice(0, 5).map(
              (item) => (
                <div
                  className="xai-group-row"
                  key={item.group}
                >
                  <div>
                    <strong>
                      {item.group}
                    </strong>
                    <small>
                      {percent(
                        item.share,
                        3,
                      )}{" "}
                      of alert score
                    </small>
                  </div>

                  <span>
                    {(
                      item.calibration_percentile *
                      100
                    ).toFixed(2)}
                    th
                  </span>
                </div>
              ),
            )}
          </div>

          <div className="callout callout-neutral">
            <strong>
              Read both values together
            </strong>
            <p>
              A high percentile means the
              detector contribution is unusual
              relative to calibration. It does
              not mean the signal has a high
              probability of being the physical
              fault.
            </p>
          </div>
        </article>
      </section>

      <section className="panel xai-temporal-panel">
        <div className="panel-title-row">
          <div>
            <span className="panel-kicker">
              3 · WHEN DID THE EVIDENCE EMERGE?
            </span>
            <h2>
              Temporal contribution context
            </h2>
          </div>

          {verified.temporal_evidence
            .window_complete ? (
            <div className="xai-status good">
              <CheckCircle2 size={16} />
              {
                verified.temporal_evidence
                  .window_rows_observed
              }
              /
              {
                verified.temporal_evidence
                  .window_bins_requested
              }{" "}
              bins
            </div>
          ) : (
            <div className="xai-status warn">
              <AlertTriangle size={16} />
              {
                verified.temporal_evidence
                  .window_rows_observed
              }
              /
              {
                verified.temporal_evidence
                  .window_bins_requested
              }{" "}
              bins · telemetry gap
            </div>
          )}
        </div>

        <ReactECharts
          option={temporalChart}
          style={{ height: 390 }}
        />

        <div className="xai-footnote">
          Percentiles are calculated against the
          calibration-only contribution
          distribution. This chart describes the
          emergence of detector evidence, not the
          physical onset time of the fault.
        </div>
      </section>

      <section className="xai-verification-grid">
        <article className="panel">
          <div className="panel-title-row">
            <div>
              <span className="panel-kicker">
                4 · COUNTERFACTUAL VERIFICATION
              </span>
              <h2>
                Does the detector depend on{" "}
                {dominant.group}?
              </h2>
            </div>

            <ShieldCheck size={24} />
          </div>

          {repair ? (
            <>
              <div className="counterfactual-scores">
                <div>
                  <small>
                    Original smoothed score
                  </small>
                  <strong>
                    {score(
                      repair.original_smoothed_score,
                    )}
                  </strong>
                </div>

                <span>→</span>

                <div>
                  <small>
                    After model-space repair
                  </small>
                  <strong>
                    {score(
                      repair.repaired_smoothed_score,
                    )}
                  </strong>
                </div>
              </div>

              <div className="threshold-line">
                Frozen threshold:{" "}
                <strong>
                  {score(verified.threshold)}
                </strong>
              </div>

              <div
                className={
                  repair.alert_cleared
                    ? "xai-verdict pass"
                    : "xai-verdict warn"
                }
              >
                {repair.alert_cleared ? (
                  <CheckCircle2 size={20} />
                ) : (
                  <AlertTriangle size={20} />
                )}

                <div>
                  <strong>
                    {repair.alert_cleared
                      ? "Alert clears after repair"
                      : "Alert remains after repair"}
                  </strong>

                  <p>
                    Under this model-space
                    counterfactual, removing the
                    anomalous evidence associated
                    with {dominant.group}{" "}
                    {repair.alert_cleared
                      ? "is sufficient to remove the frozen alert."
                      : "is not sufficient to remove the frozen alert."}
                  </p>
                </div>
              </div>
            </>
          ) : null}
        </article>

        <article className="panel">
          <span className="panel-kicker">
            KNOWLEDGE LIMIT
          </span>
          <h2>
            What AeroXAI can and cannot say
          </h2>

          <dl className="details-list">
            <div>
              <dt>
                Documented onset
              </dt>
              <dd>
                {selected.incident_start}
              </dd>
            </div>

            <div>
              <dt>
                Explanation timestamp
              </dt>
              <dd>
                {
                  selected.explanation_timestamp
                }
              </dd>
            </div>

            <div>
              <dt>
                Detector dependence
              </dt>
              <dd className="positive">
                Verified in model space
              </dd>
            </div>

            <div>
              <dt>
                Physical root-cause claim
              </dt>
              <dd className="negative">
                No
              </dd>
            </div>
          </dl>

          <div className="callout callout-neutral">
            <strong>
              Interpretation boundary
            </strong>
            <p>
              Counterfactual repair changes PCA
              feature-space evidence. It does not
              simulate a physical repair and does
              not prove that the dominant signal
              is the underlying physical cause.
            </p>
          </div>
        </article>
      </section>
    </div>
  );
}