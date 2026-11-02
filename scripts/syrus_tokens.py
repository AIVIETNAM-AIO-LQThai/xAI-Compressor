"""Offline prompt-size report for the Syrus acceptance questions: estimated input tokens before selection (the whole ledger) and
after it (the question's slice), by part, with what the slice leaves out. Makes no model call and needs no key.

    python -m scripts.syrus_tokens             # table of every acceptance question
    python -m scripts.syrus_tokens --parts     # add the estimated tokens by part
    python -m scripts.syrus_tokens --markdown  # the same as a Markdown table

The estimate is the pessimistic character-based one in backend/syrus_app/tokens.py; the live usage reported by the provider is the
check on it. A follow-up turn is measured with a synthetic earlier answer of the largest size its depth allows.
"""
import argparse
from collections import deque

import yaml

from backend.syrus_app.context import CaseStore
from backend.syrus_app.prompt import baseline, build
from backend.syrus_app.records import ROOT
from backend.syrus_app.selection import SelectionError
from backend.syrus_app.sessions import Thread
from backend.syrus_app.settings import load_settings

FIXTURE = ROOT / "tests" / "fixtures" / "syrus_acceptance.yaml"
ANSWER_CHARS = {"quick": 3 * 260, "investigation": 6 * 260, "technical": 10 * 260, "auto": 3 * 260}  # units a depth allows times a typical unit


def items() -> list[dict]:
    return yaml.safe_load(FIXTURE.read_text("utf-8"))["items"]


def thread_after(pack, history: list[tuple[str, str]], focus: list[str], depth: str, history_max: int) -> Thread:
    t = Thread("report", pack.case_id, "retro", 0, exchanges=deque(maxlen=history_max))
    for q in history:
        t.exchanges.append((q, "a" * ANSWER_CHARS[depth]))
    t.turns, t.focus = len(history), focus
    return t


def rows(store: CaseStore, cfg_settings, only: list[str] | None = None, cases: list[str] | None = None) -> list[dict]:
    """One row per question turn: the baseline and the selected prompt, with what the selection left out."""
    by_alert = {c["alert_time"][:10]: c["case_id"] for c in store.manifest["cases"]}
    cap, hist = cfg_settings.input_cap, cfg_settings.limits["history_exchanges"]
    out = []
    for item in items():
        if only and item["id"] not in only:
            continue
        for case_id in cases or [by_alert[item["alert"]]]:
            pack, asked, focus = store.pack(case_id), [], []
            for n, turn in enumerate(item["turns"]):
                th = thread_after(pack, asked, focus, item["depth"], hist)
                before = baseline(pack, th, turn["text"], item["depth"], keep=len(th.exchanges))
                row = {"item": item["id"], "turn": n + 1, "case": case_id, "text": turn["text"], "before": before.est, "before_parts": before.parts, "cap": cap}
                try:
                    built = build(pack, th, turn["text"], item["depth"], cap, hist)
                except SelectionError as e:
                    out.append({**row, "error": e.reason, "after": None})
                    break
                v = built.view
                fired = {f for f, x in pack.findings.items() if x["fires"]}
                essentials = turn["essentials"]
                out.append({**row, "after": built.est, "after_parts": built.parts, "focus": list(v.focus), "dropped": list(v.dropped), "purposes": list(v.purposes),
                            "history": built.keep, "records": len(v.records), "records_all": len(pack.records), "findings": sorted(v.findings),
                            "findings_left_out": sorted(fired - set(v.findings)), "tests": len(v.tests),
                            "essentials_missing": [e for e in essentials if e not in v.ids()],
                            "required": list(v.required), "required_missing": sorted(set(v.required) - set(v.findings))})
                asked.append(turn["text"])
                focus = list(v.focus)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline prompt-size report for the Syrus acceptance questions.")
    ap.add_argument("--parts", action="store_true", help="add the estimated tokens by part")
    ap.add_argument("--markdown", action="store_true", help="print a Markdown table")
    ap.add_argument("--all-cases", action="store_true", help="measure every question on all four cases")
    args = ap.parse_args()
    store, settings = CaseStore(), load_settings({})
    cases = [c["case_id"] for c in store.manifest["cases"]] if args.all_cases else None
    data = rows(store, settings, cases=cases)
    sep = " | " if args.markdown else "  "
    head = ["question", "turn", "before", "after", "focus", "left out of the view", "essentials", "must convey"]
    if args.markdown:
        print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for r in data:
        cells = [r["item"] + (f" ({r['case']})" if args.all_cases else ""), str(r["turn"]), str(r["before"]), str(r["after"] if r["after"] is not None else r["error"]),
                 ", ".join(r.get("focus", [])) + (f" (dropped {', '.join(r['dropped'])})" if r.get("dropped") else ""),
                 f"{r['records_all'] - r['records']} of {r['records_all']} records; findings {', '.join(r['findings_left_out']) or 'none'}" if r["after"] else "",
                 "all present" if r["after"] and not r["essentials_missing"] else str(r.get("essentials_missing", "")),
                 ", ".join(r.get("required", [])) or "-"]
        print(("| " + sep.join(cells) + " |") if args.markdown else sep.join(cells))
        if args.parts and r["after"]:
            print(f"    before {r['before_parts']}\n    after  {r['after_parts']}")
    if data:
        fits = [r["after"] for r in data if r["after"]]
        print(f"\ncap {data[0]['cap']} estimated input tokens; largest after {max(fits)}; before {min(r['before'] for r in data)}-{max(r['before'] for r in data)}")


if __name__ == "__main__":
    main()
