import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { runs } from "../src/pages/runs.ts";

const t = [0, 300_000, 600_000, 900_000, 1_200_000];

describe("alert runs", () => {
  it("joins consecutive bins and gives a single bin one bin of width", () => {
    assert.deepEqual(runs(t, [false, true, true, false, true]), [
      [300_000, 900_000],
      [1_200_000, 1_500_000],
    ]);
  });

  it("treats a gap or a period without alerts as not in an alert", () => {
    assert.deepEqual(runs(t, [null, null, true, null, null]), [[600_000, 900_000]]);
    assert.deepEqual(runs(t, [null, null, null, null, null]), []);
  });
});
