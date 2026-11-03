import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { days, staticUrl, stitch } from "../src/staticData.ts";

describe("showcase files", () => {
  it("maps every read-only path the interface uses to a file", () => {
    assert.equal(staticUrl("/api/cases/c-h3vz?mode=retro"), "/data/api/cases/c-h3vz.retro.json");
    assert.equal(staticUrl("/explorer/episode?start=2020-05-29%2021%3A20"), "/data/explorer/episode/2020-05-29T21-20.json");
    assert.equal(staticUrl("/evidence/proof-bundle/verify"), "/data/evidence/proof-bundle/verify.json");
    assert.equal(staticUrl("/config"), "/data/config.json");
  });

  it("lists every day a window touches", () => {
    assert.deepEqual(days("2020-05-29 18:00:00", "2020-05-31 02:00:00"), ["2020-05-29", "2020-05-30", "2020-05-31"]);
  });
});

describe("joining day files", () => {
  const day1 = { t: ["2020-05-29 23:50:00", "2020-05-29 23:55:00"], smoothed_score: [1, 2], channels: { tp3: { mean: [9, 9.1] } }, threshold: 5 };
  const day2 = { t: ["2020-05-30 00:20:00", "2020-05-30 00:25:00"], smoothed_score: [3, 4], channels: { tp3: { mean: [9.2, 9.3] } }, threshold: 5 };

  it("breaks the line at a gap that falls on a day boundary, with the server's rule", () => {
    const got = stitch([day1, day2], "2020-05-29 23:50:00", "2020-05-30 00:25:00", 300_000) as { t: string[]; smoothed_score: Array<number | null>; channels: { tp3: { mean: Array<number | null> } } };
    assert.deepEqual(got.t, ["2020-05-29 23:50:00", "2020-05-29 23:55:00", "2020-05-30 00:15:00", "2020-05-30 00:20:00", "2020-05-30 00:25:00"]);
    assert.deepEqual(got.smoothed_score, [1, 2, null, 3, 4]);
    assert.deepEqual(got.channels.tp3.mean, [9, 9.1, null, 9.2, 9.3]);
  });

  it("keeps only rows inside the window, with the raw end excluded as on the server", () => {
    const got = stitch([day1, day2], "2020-05-29 23:55:00", "2020-05-30 00:25:00", 300_000, true) as { t: string[] };
    assert.deepEqual(got.t, ["2020-05-29 23:55:00", "2020-05-30 00:15:00", "2020-05-30 00:20:00"]);
  });
});
