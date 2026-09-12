import type { ReactNode } from "react";

interface Props {
  label: string;
  value: ReactNode;
  note?: string;
  accent?: "cyan" | "lime" | "amber" | "red";
}

export function MetricCard({
  label,
  value,
  note,
  accent = "cyan",
}: Props) {
  return (
    <article className={`metric-card accent-${accent}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {note ? (
        <div className="metric-note">{note}</div>
      ) : null}
    </article>
  );
}
