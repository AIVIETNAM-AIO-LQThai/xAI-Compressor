import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { responseText, severityText } from "../src/pages/labText.ts";

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
