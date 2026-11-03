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

interface TwinLike {
  scenario: string;
  inputs: { added_demand_kg_s: number[]; leak_kg_s: number[]; total_outflow_kg_s: number[] };
  response: { pressure_bar_g: number[]; compressor_flow_kg_s: number[]; power_kw: number[]; safety_violation: boolean[] };
  summary: { energy_kwh: number; starts: number; safety_violations: number; added_demand_kg: number; leak_kg: number };
}

const gap = (a: number[], b: number[]) => (a.length !== b.length ? Infinity : a.reduce((m, v, i) => Math.max(m, Math.abs(v - b[i])), 0));

/** Compares two generic-plant runs from their own numbers: whether the totals and the observable responses match, and how the inputs differ. */
export function compareTwins(a: TwinLike, b: TwinLike) {
  const response = {
    pressure: gap(a.response.pressure_bar_g, b.response.pressure_bar_g),
    flow: gap(a.response.compressor_flow_kg_s, b.response.compressor_flow_kg_s),
    power: gap(a.response.power_kw, b.response.power_kw),
    energy: Math.abs(a.summary.energy_kwh - b.summary.energy_kwh),
  };
  const sameTotal = gap(a.inputs.total_outflow_kg_s, b.inputs.total_outflow_kg_s) < 1e-9;
  const sameResponse =
    response.pressure < 1e-9 && response.flow < 1e-9 && response.power < 1e-9 && response.energy < 1e-9 && a.summary.starts === b.summary.starts && a.summary.safety_violations === b.summary.safety_violations;
  const sameInputs = gap(a.inputs.added_demand_kg_s, b.inputs.added_demand_kg_s) < 1e-12 && gap(a.inputs.leak_kg_s, b.inputs.leak_kg_s) < 1e-12;
  return { sameTotal, sameResponse, sameInputs, response };
}
