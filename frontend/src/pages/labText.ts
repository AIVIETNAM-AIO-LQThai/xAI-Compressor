const ms = (t: string) => Date.parse(`${t.replace(" ", "T")}Z`);

/** What the detector did, in words. "No alert" is a result like any other and is never softened into a near miss. */
export function responseText(first: { baseline: string | null; perturbed: string | null }, onset: string): string {
  if (first.baseline) return `The unchanged recording already alerts at ${first.baseline.slice(11, 16)}; choose a quieter background to see the effect of the change.`;
  if (!first.perturbed) return "No alert: the detector did not respond to this change in a way that meets its alert rule.";
  const after = (ms(first.perturbed) - ms(onset)) / 3_600_000;
  const lead = after >= 0 ? `${after.toFixed(1)} h after the change began` : `${(-after).toFixed(1)} h before the change began`;
  return `Alert at ${first.perturbed.slice(11, 16)}, ${lead}.`;
}

/** How a severity reads for each fault kind. Channel units are not loaded, so sizes are in training standard deviations or shares. */
export function severityText(fault: string, severity: number, std: number | null): string {
  if (fault === "dropout") return `${Math.round(severity * 100)}% of samples removed`;
  if (fault === "flatline" || fault === "stuck") return "no size: the channel is held at one value";
  const base = `${severity} training standard deviations`;
  return std === null ? base : `${base} (${Number((severity * std).toPrecision(3))} recorded units)`;
}
