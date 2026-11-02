import assert from "node:assert/strict";
import { beforeEach, describe, it } from "node:test";

import { hashFor, parse, recall, remember } from "../src/selection.ts";

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

  it("ignores anything malformed", () => {
    for (const v of ["{", '{"kind":"incident","caseId":"x","mode":"retro"}', '{"kind":"incident","caseId":"c-h3vz","mode":"live"}', '{"kind":"episode","caseId":"c-h3vz","mode":"retro"}']) {
      store.set("aeroxai.selection", v);
      assert.equal(recall(), null, v);
    }
  });
});
