import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { compareTwins, responseText, severityText } from "../src/pages/labText.ts";

const onset = "2020-03-01 13:00:00";

describe("detector response wording", () => {
  it("says plainly when there was no alert", () => {
    assert.match(responseText({ baseline: null, perturbed: null }, onset), /^No alert/);
  });

  it("gives the delay after the change began", () => {
    assert.equal(responseText({ baseline: null, perturbed: "2020-03-01 13:30:00" }, onset), "Alert at 13:30, 0.5 h after the change began.");
  });

  it("never credits the change when the unchanged recording already alerts", () => {
    assert.match(responseText({ baseline: "2020-03-01 12:00:00", perturbed: "2020-03-01 12:00:00" }, onset), /already alerts/);
  });
});

describe("severity wording", () => {
  it("uses training standard deviations and never claims units", () => {
    assert.equal(severityText("offset", 3, 0.554), "3 training standard deviations (1.66 recorded units)");
    assert.equal(severityText("noise", 2, null), "2 training standard deviations");
    assert.equal(severityText("dropout", 0.6, null), "60% of samples removed");
    assert.match(severityText("stuck", 1, null), /held at one value/);
  });
});

describe("twin comparison", () => {
  const run = (added: number[], leak: number[], pressure: number[]) => ({
    scenario: "x",
    inputs: { added_demand_kg_s: added, leak_kg_s: leak, total_outflow_kg_s: added.map((a, i) => 0.09 + a + leak[i]) },
    response: { pressure_bar_g: pressure, compressor_flow_kg_s: [0.1, 0.11], power_kw: [20, 22], safety_violation: [false, false] },
    summary: { energy_kwh: 1, starts: 1, safety_violations: 0, added_demand_kg: added[1], leak_kg: leak[1] },
  });

  it("finds the same response and total from the numbers, with different declared inputs", () => {
    const c = compareTwins(run([0, 0.02], [0, 0], [7, 6.9]), run([0, 0], [0, 0.02], [7, 6.9]));
    assert.ok(c.sameTotal && c.sameResponse && !c.sameInputs);
  });

  it("reports a difference in response instead of assuming equality", () => {
    const c = compareTwins(run([0, 0.02], [0, 0], [7, 6.9]), run([0, 0], [0, 0.02], [7, 6.8]));
    assert.ok(c.sameTotal && !c.sameResponse);
    assert.ok(Math.abs(c.response.pressure - 0.1) < 1e-9);
  });
});
