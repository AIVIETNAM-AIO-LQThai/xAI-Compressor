import type { EvidenceClass } from "../types";

interface Props {
  kind: EvidenceClass;
}

export function EvidenceBadge({ kind }: Props) {
  const label =
    kind === "REAL"
      ? "REAL TELEMETRY"
      : kind === "SIMULATED"
        ? "SIMULATED DIGITAL TWIN"
        : "LITERATURE";

  return (
    <span
      className={`evidence-badge evidence-${kind.toLowerCase()}`}
    >
      <span className="evidence-dot" />
      {label}
    </span>
  );
}
