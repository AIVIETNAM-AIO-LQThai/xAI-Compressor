/** Where a number comes from, in the same words on every page. Reviewed incident findings keep their own "verified" label. */
export type Source = "RECORDED" | "MODEL INFERENCE" | "SIMULATED" | "UNREVIEWED";

const HINT: Record<Source, string> = {
  RECORDED: "Values recorded on the MetroPT-3 air-production unit",
  "MODEL INFERENCE": "Computed by the frozen detector from recorded data",
  SIMULATED: "Produced by a simulation, not measured",
  UNREVIEWED: "Computed on request; not reviewed as a case file",
};

export function SourceTag({ source }: { source: Source }) {
  return (
    <span className={`source-tag source-${source.toLowerCase().replace(" ", "-")}`} title={HINT[source]}>
      {source}
    </span>
  );
}
