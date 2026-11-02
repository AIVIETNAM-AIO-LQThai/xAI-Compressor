"""A few targeted live smoke questions through the real Syrus service (owner-approved, spend-capped). Not a benchmark.

    python -m scripts.syrus_smoke                    # dry run: print the plan and the worst-case cost, call nothing
    python -m scripts.syrus_smoke --run [--only id]  # make the calls

Needs SYRUS_LLM_API_KEY and SYRUS_CAPS_CONFIRMED=1 in the environment. Calls go through the same budget guard as the app and are
spaced so each falls in its own minute of the provider's tokens-per-minute limit. The cost shown is worst case, from the size of
each question's actual prompt (see scripts/syrus_tokens.py) and the configured maximum output, not a fixed figure. Prints
statuses, coverage of the listed evidence, token use and the sanitised provider diagnostics; the answers are for the owner's
manual review.
"""
import argparse
import dataclasses
import json
import time

import yaml
from fastapi.testclient import TestClient

from backend.syrus_app.app import create_app
from backend.syrus_app.context import CaseStore
from backend.syrus_app.records import ROOT
from backend.syrus_app.settings import load_settings
from scripts.syrus_tokens import rows

FIXTURE = ROOT / "tests" / "fixtures" / "syrus_acceptance.yaml"
SMOKE = ["why_alert", "definitely_a_leak", "dv_pressure_cause", "energy_wasted", "maintenance_log_vi", "operator_statement"]
SPACING = 65  # seconds between calls: keeps each request, and a retry after it, inside its own minute of the provider limit


def main() -> None:
    ap = argparse.ArgumentParser(description="Live smoke questions for Syrus.")
    ap.add_argument("--run", action="store_true", help="make the calls; without it nothing is sent")
    ap.add_argument("--only", help="run a single fixture item")
    args = ap.parse_args()
    items = [i for i in yaml.safe_load(FIXTURE.read_text("utf-8"))["items"] if i["id"] in ([args.only] if args.only else SMOKE)]
    settings = load_settings()
    settings = dataclasses.replace(settings, diagnostics=True)  # sanitised provider and token diagnostics in each response
    cfg, store = settings.cfg, CaseStore()
    by_alert = {c["alert_time"][:10]: c["case_id"] for c in store.manifest["cases"]}
    price, out_max, cap = cfg["price_per_million"], cfg["limits"]["max_output_tokens"], settings.input_cap
    cost = lambda tin, tout: tin * price["input"] / 1e6 + tout * price["output"] / 1e6
    plan = rows(store, settings, only=[i["id"] for i in items])
    worst = {}
    for r in plan:
        if r["after"] is None:
            raise SystemExit(f"not running: {r['item']} needs more context than the {cap}-token input cap ({r['error']})")
        retry = r["after"] + out_max + 100  # the failed reply and the correction are appended to the retry
        worst[r["item"], r["turn"]] = cost(r["after"], out_max) + (cost(retry, out_max) if retry <= cap else 0)
    print(f"model {settings.model}; {len(items)} questions; input cap {cap} estimated tokens; worst case ${sum(worst.values()):.3f} with a retry on every one")
    print(f"caps ${cfg['caps_usd']['daily']}/day, ${cfg['caps_usd']['total']} total; offline reason: {settings.offline_reason()}")
    for r in plan:
        print(f"  {r['item']} turn {r['turn']}: ~{r['after']} input tokens (whole ledger: {r['before']}), focus {', '.join(r['focus'])}, worst ${worst[r['item'], r['turn']]:.4f}")
    if not args.run:
        return
    if settings.offline_reason():
        raise SystemExit(f"not running: {settings.offline_reason()}")
    c = TestClient(create_app(settings, store=store))
    for n, item in enumerate(items):
        if n:
            time.sleep(SPACING)
        token = c.post("/api/session", json={"case_id": by_alert[item["alert"]], "mode": "retro"}).json()["token"]
        for t in item["turns"]:
            r = c.post("/api/syrus/turn", json={"message": t["text"], "depth": item["depth"]}, headers={"Authorization": f"Bearer {token}"})
            body = r.json()
            cited = {x for b in body.get("blocks", []) for x in (*b["records"], *b["findings"])} | {a["finding_id"] for a in body.get("also_relevant", [])}
            print(f"\n=== {item['id']} [{body['status']}{' ' + body.get('reason', '') if body['status'] == 'offline' else ''}] {t['text']}")
            if body["status"] == "ok":
                print(body["answer"])
                print(f"-- purposes {body['purposes']}, refusal {body['refusal']} (expected {t['refusal']}), missing essentials {sorted(set(t['essentials']) - cited)}, "
                      f"validator {body['provenance']['validator']}")
            else:
                print(body.get("message") or {k: v for k, v in body.items() if k != "diagnostics"})
            print("-- diagnostics", json.dumps(body.get("diagnostics")))
            if body["status"] == "offline":
                raise SystemExit(1)
    spent = c.app.state.svc.budget.state
    print(f"\nrecorded spend: total ${spent['total']:.4f}")


if __name__ == "__main__":
    main()
