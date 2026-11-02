import assert from "node:assert/strict";
import { beforeEach, describe, it } from "node:test";

import { hashFor, parse, recall, recallIncident, recallView, remember } from "../src/selection.ts";

describe("selection from the address", () => {
  it("reads an incident deep link", () => {
    assert.deepEqual(parse("#investigate/incident/c-h3vz/asof"), { route: "investigate", selection: { kind: "incident", caseId: "c-h3vz", mode: "asof" } });
  });

  it("selects nothing without an explicit incident", () => {
    for (const h of ["", "#", "#overview", "#investigate", "#investigate/incident", "#investigate/incident/../x", "#nowhere"]) {
      assert.equal(parse(h).selection, null, h);
    }
    assert.equal(parse("#investigate").route, "investigate");
    assert.equal(parse("#nowhere").route, "overview");
  });

  it("understands the earlier investigation links", () => {
    assert.deepEqual(parse("#investigation/c-r6tq/asof"), { route: "investigate", selection: { kind: "incident", caseId: "c-r6tq", mode: "asof" } });
    assert.deepEqual(parse("#investigation"), { route: "investigate", selection: null });
  });

  it("round-trips through the hash", () => {
    const s = { kind: "incident" as const, caseId: "c-w8nk", mode: "retro" as const };
    assert.deepEqual(parse(`#${hashFor(s)}`).selection, s);
  });
});

describe("explorer selections", () => {
  it("reads an episode and an interval", () => {
    assert.deepEqual(parse("#explore/episode/2020-05-29T21:20"), { route: "explore", selection: { kind: "episode", start: "2020-05-29T21:20" } });
    assert.deepEqual(parse("#explore/interval/2020-05-29T00:00/2020-05-30T00:00"), { route: "explore", selection: { kind: "interval", start: "2020-05-29T00:00", end: "2020-05-30T00:00" } });
  });

  it("rejects malformed or reversed windows", () => {
    for (const h of ["#explore", "#explore/episode/x", "#explore/interval/2020-05-29T00:00", "#explore/interval/2020-05-30T00:00/2020-05-29T00:00", "#explore/interval/2020-05-29/2020-05-30"]) {
      assert.equal(parse(h).selection, null, h);
    }
  });

  it("round-trips through the hash", () => {
    for (const s of [{ kind: "episode" as const, start: "2020-03-04T05:15" }, { kind: "interval" as const, start: "2020-03-04T00:00", end: "2020-03-05T00:00" }]) {
      assert.deepEqual(parse(`#${hashFor(s)}`).selection, s);
    }
  });
});

describe("remembered selection", () => {
  const store = new Map<string, string>();
  beforeEach(() => {
    store.clear();
    (globalThis as { sessionStorage?: unknown }).sessionStorage = {
      getItem: (k: string) => store.get(k) ?? null,
      setItem: (k: string, v: string) => void store.set(k, v),
    };
  });

  it("returns the last explicit choice", () => {
    assert.equal(recall(), null);
    remember({ kind: "incident", caseId: "c-b5xm", mode: "asof" });
    assert.deepEqual(recall(), { kind: "incident", caseId: "c-b5xm", mode: "asof" });
  });

  it("keeps the last incident when an interval is chosen afterwards", () => {
    remember({ kind: "incident", caseId: "c-h3vz", mode: "retro" });
    remember({ kind: "interval", start: "2020-03-04T00:00", end: "2020-03-05T00:00" });
    assert.equal(recall()?.kind, "interval");
    assert.equal(recallIncident()?.caseId, "c-h3vz");
    assert.equal(recallView()?.kind, "interval");
  });

  it("ignores anything malformed", () => {
    for (const v of ["{", '{"kind":"incident","caseId":"x","mode":"retro"}', '{"kind":"incident","caseId":"c-h3vz","mode":"live"}', '{"kind":"episode","caseId":"c-h3vz","mode":"retro"}']) {
      store.set("aeroxai.selection", v);
      assert.equal(recall(), null, v);
    }
  });
});
