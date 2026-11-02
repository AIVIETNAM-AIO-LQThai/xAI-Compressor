"""Findings ledger: deterministic rules over one case's records.

Each finding has a stable ID, the rule in words, the record IDs it reads, a structured output, whether it fires,
its visibility, a short English text and an optional technical `detail`. Both texts are built only from the input
records' display strings. Qualifications that must accompany a claim are not part of the ledger; they are applied
when an answer is assembled (docs/syrus/LEDGER_REVIEW.md).
"""
from itertools import pairwise

from backend.syrus_app.records import listing, minutes


def finding(id, rule, inputs, output, text, fires=True, vis="both", detail=None):
    return {"id": id, "rule": rule, "inputs": inputs, "output": output, "fires": fires, "visibility": vis, "text": text, "detail": detail}


def _lower(label: str) -> str:
    return label[0].lower() + label[1:]


def _pc(row: dict, g: str) -> str:
    return row["display"]["percentiles"][g]


def margin(R, ctx):
    d = {i: R[i]["display"] for i in ("alert.smoothed", "alert.threshold", "alert.margin")}
    text = f"The smoothed score was {d['alert.smoothed']} against a threshold of {d['alert.threshold']} ({d['alert.margin']})."
    return finding("F-MARGIN", "margin = smoothed / threshold - 1", list(d), {"margin": R["alert.margin"]["value"]}, text)


def ewma_memory(R, ctx):
    raw, prev = R["alert.raw"], R["alert.prev_smoothed"]
    falling = raw["value"] < prev["value"]["score"]
    at, carried, mem = prev["display"]["time"], prev["display"]["score"], R["alert.memory"]["display"]
    if falling:
        text = (f"The alert bin's own raw score ({raw['display']}) was below the smoothed score carried from {at} ({carried}), so the "
                f"smoothed score was falling at the alert, and the smoothing state carried from earlier bins accounts for {mem} of the current smoothed score.")
    else:
        text = (f"The alert bin's own raw score ({raw['display']}) was above the smoothed score carried from {at} ({carried}), so the "
                f"smoothed score was still rising at the alert, and the smoothing state carried from earlier bins accounts for {mem} of the current smoothed score.")
    rule = R["detector.ewma"]["display"]["rule"]
    detail = (f"Smoothing rule: {rule}. The share from earlier bins is one minus the alert bin's own term. This shows how the smoothing "
              "works; it does not show which earlier bins carried the evidence or that a condition persisted, and one large raw score "
              "can by itself produce several smoothed hits.")
    return finding("F-EWMA-MEMORY", "fires when the alert bin's raw score is below the previous smoothed score (smoothed score falling)",
                   ["alert.raw", "alert.prev_smoothed", "alert.memory", "detector.ewma"],
                   {"falling": falling, "memory_share": R["alert.memory"]["value"]}, text, fires=falling, detail=detail)


def evidence_timing(R, ctx):
    rows, groups = [R[i] for i in ctx["temporal_ids"]], ctx["display_groups"]
    first, last = rows[0], rows[-1]
    out, parts = {}, []
    for g in groups:
        steps = [(b["value"]["percentiles"][g] - a["value"]["percentiles"][g], a, b) for a, b in pairwise(rows)]
        rise, a, b = max(steps, key=lambda s: s[0])
        top = next((r for r in rows if r["value"]["percentiles"][g] == 1), None)
        gap = minutes(a["value"]["time"], b["value"]["time"]) > ctx["bin_minutes"]
        clause = f"{g} {_pc(first, g)} at {first['display']['time']} and {_pc(last, g)} at {last['display']['time']}"
        if rise > 0:
            clause += f", largest one-bin rise {_pc(a, g)} to {_pc(b, g)} between {a['display']['time']} and {b['display']['time']}" + (" across a data gap" if gap else "")
        clause += (", above every calibration bin from the window start" if top is first else
                   f", first above every calibration bin at {top['display']['time']}" if top else ", never above every calibration bin in this window")
        parts.append(clause)
        out[g] = {"start": first["value"]["percentiles"][g], "end": last["value"]["percentiles"][g],
                  "rise": {"from": a["value"]["time"], "to": b["value"]["time"], "size": rise, "across_gap": gap} if rise > 0 else None,
                  "first_above_all": top["value"]["time"] if top else None}
    text = ("Calibration percentile of each group's smoothed contribution: " + "; ".join(parts)
            + ". These rank the detector's smoothed evidence; they do not show when a fault began.")
    return finding("F-EVIDENCE-TIMING", "per displayed group: percentile at window start and at the alert, largest one-bin rise, "
                   "first bin above every calibration bin; no thresholds", ["temporal.window", *ctx["temporal_ids"]], out, text)


def cf_nonunique(R, ctx):
    cfs = [R[i] for i in ctx["cf_ids"]]
    clear = [c for c in cfs if c["value"]["alert_cleared"]]
    thr, t = R["alert.threshold"]["display"], R["alert.time"]["display"][11:16]
    names = [c["value"]["group"] for c in clear]
    scores = listing([c["display"]["repaired_smoothed"] for c in clear]) if clear else ""
    if len(clear) > 1:
        text = (f"Replacing any one of {listing(names, 'or')} with the PCA model's reconstruction and re-scoring removes the alert at {t} "
                f"(repaired smoothed scores {scores} against the {thr} threshold), so the detector's decision does not single out one "
                "signal group. This tests the detector, not the machine.")
    elif clear:
        others = [c["value"]["group"] for c in cfs if not c["value"]["alert_cleared"]]
        text = (f"Only the {names[0]} repair removes the alert at {t} (repaired smoothed score {scores} against the {thr} threshold); "
                f"repairs of {listing(others)} leave it in place. This tests the detector, not the machine.")
    else:
        text = f"None of the tested single-group repairs removes the alert at {t}. This tests the detector, not the machine."
    w, pre = cfs[0]["display"]["window"], R["cf.prewindow"]["display"]
    ratios = [f"{c['value']['group']} {c['display']['ratio']}" for c in clear if c["display"]["ratio"].endswith("×")]
    detail = (f"Tested groups, each replaced on its own over {w}: {listing([c['value']['group'] for c in cfs])}, the groups with the largest "
              f"smoothed contributions. Combinations and other groups were not tested, and clearing is checked at {t} only.")
    if ratios:
        detail += (" Re-projection changes other groups' residuals too, so a repair is not the same as removing the group's own "
                   f"contribution. The smoothed score falls by a multiple of the group's own smoothed contribution: {listing(ratios)}.")
    detail += f" The repair does not edit the smoothing state from before the window; its share of the {t} smoothed score: {pre}."
    return finding("F-CF-NONUNIQUE", "fires when two or more tested single-group repairs clear the alert at the alert bin",
                   ["alert.threshold", "alert.time", "cf.prewindow", *ctx["cf_ids"]], {"clearing": names}, text, fires=len(clear) > 1, detail=detail)


def cf_persistence(R, ctx):
    thr, t = R["alert.threshold"], R["alert.time"]["display"][11:16]
    per = R["detector.persistence"]["value"]
    hits = [R[i] for i in ctx["cf_ids"] if R[i]["value"]["path"] == "persistence"]
    if hits:
        text = " ".join(f"Replacing {c['value']['group']} leaves the {t} smoothed score at {c['display']['repaired_smoothed']}, above the "
                        f"{thr['display']} threshold, yet the alert clears because fewer than {per['hits']} of the last {per['window']} "
                        "smoothed scores stay at or above it." for c in hits)
    else:
        text = "Every repair that clears this alert also brings the smoothed score below the threshold."
    return finding("F-CF-PERSIST", "fires when a clearing repair leaves the smoothed score at or above the threshold",
                   ["alert.threshold", "alert.time", "detector.persistence", *ctx["cf_ids"]], {"groups": [c["value"]["group"] for c in hits]},
                   text, fires=bool(hits))


def train_range(R, ctx):
    rows, clauses = [], []
    for i in ctx["measured_ids"]:
        v, d = R[i]["value"], R[i]["display"]
        pos = "below" if v["value"] < v["q1"] else "above" if v["value"] > v["q3"] else "inside"
        rows.append({"feature": v["feature"], "position": pos})
        clauses.append(f"{v['feature']} ({d['value']}) was {pos} its training interquartile range ({d['q1']} to {d['q3']})")
    text = ("At the alert bin, " + "; ".join(clauses) + ". Only these features are exported for the alert bin; units are not loaded, "
            "and a value inside the range is not shown to be normal.")
    return finding("F-TRAIN-RANGE", "position of each exported measured feature against its training interquartile range",
                   ctx["measured_ids"], {"features": rows}, text)


def gap(R, ctx):
    w = R["temporal.window"]
    gaps, d = w["value"]["gaps"], w["display"]
    if not gaps:
        text = f"All bins of the past-only window are present ({d['bins']})."
    else:
        text = f"{d['bins']} bins are present; no bins between " + "; ".join(d["gaps"]) + "."
        if any(g["minutes"] > ctx["reset_gap"] for g in gaps):
            text += f" Smoothing and persistence restart after a gap over {ctx['reset_gap']} min."
    return finding("F-GAP", "fires when the past-only window is incomplete", ["temporal.window", "detector.ewma"], {"gaps": gaps}, text,
                   fires=bool(gaps))


def candidates(R, ctx):
    hyps = [R[i] for i in ctx["hyp_ids"]]
    label = {h["value"]["id"]: h["display"]["label"] for h in hyps}
    pairs = sorted({tuple(sorted((h["value"]["id"], o))) for h in hyps for o in h["value"]["not_separable_from"]})
    text = "All candidate explanations are open: no reviewed test has examined them, and Syrus does not rank them."
    text += "".join(f" {label[a]} and {_lower(label[b])} cannot be told apart with the data recorded for this asset." for a, b in pairs)
    return finding("F-CANDIDATES", "lists candidate statuses (open until a reviewed test returns a result) and pair limits; never ranks",
                   ctx["hyp_ids"], {"open": [h["value"]["id"] for h in hyps if h["value"]["status"] == "open"], "not_separable": [list(p) for p in pairs]},
                   text)


def identifiability(R, ctx):
    ins = ["unavailable:demand_measurement", "hyp.air_leakage", "hyp.downstream_demand"]
    text = ("Leakage and higher downstream demand both raise total outflow (total outflow = demand + leakage). Without an "
            "independent demand measurement this data cannot tell them apart.")
    return finding("F-IDENT", "always present: leak and demand are not separately identifiable without a demand observation", ins,
                   {"scope": "asset"}, text)


def bridge(R, ctx):
    b = R["bridge.status"]
    text = f"Real-asset physical reasoning and actions are withheld. Missing: {'; '.join(b['display']['missing'])}."
    return finding("F-BRIDGE", "always present while the physical bridge is not validated", ["bridge.status"],
                   {"status": b["value"]["status"], "scope": "asset"}, text)


def label(R, ctx):
    d = R["label.incident"]["display"]
    text = (f"The provider's failure list, as transcribed in the repository configuration, labels {d['interval']} as \"{d['condition']}\". "
            "AeroXAI's evidence does not establish this label.")
    return finding("F-LABEL", "states the provider label; retrospective", ["label.incident"], {}, text, vis="retro")


def timing(R, ctx):
    t, k = R["label.timing"], R["bench.preonset"]["display"]
    if t["value"]["timing"] == "pre_onset":
        text = (f"The alert came {t['display']} the documented onset. An alert preceded the documented onset; whether it was a "
                f"precursor is not established. {k} documented incidents had a pre-onset alert.")
    else:
        text = f"The alert came {t['display']} the documented onset. {k} documented incidents had a pre-onset alert."
    return finding("F-TIMING", "states alert timing against the documented onset; retrospective", ["label.timing", "bench.preonset"],
                   {"timing": t["value"]["timing"]}, text, vis="retro")


def selection(R, ctx):
    text = ("Documented incident in the provider's failure list. This alert was chosen for explanation because it falls in that "
            "incident's timely window, so the case set is selected by the labels.")
    return finding("F-SELECTION", "states the case status and how the case was selected; retrospective", ["case.status"], {}, text, vis="retro")


def alert_burden(R, ctx):
    d = {i: R[i]["display"] for i in ("bench.timely", "bench.false_rate", "bench.episodes", "bench.exposure")}
    e = d["bench.episodes"]
    text = (f"{d['bench.timely']} documented incidents had a timely alert. Over {d['bench.exposure']} of valid TEST exposure the detector "
            f"raised {d['bench.false_rate']} ({e['benchmark_false']} of {e['total']} episodes). Benchmark-false means not linked to a "
            "documented incident window; it is not a confirmed false alarm or confirmed healthy behaviour.")
    return finding("F-ALERT-BURDEN", "states timely recall and the benchmark-false alert rate over TEST; retrospective", list(d), {}, text, vis="retro")


def discrimination(R, ctx):
    d = {i: R[i]["display"] for i in ("bench.pr_auc", "bench.profiles", "bench.selection")}
    p = d["bench.profiles"]
    text = (f"PR-AUC {d['bench.pr_auc']}. Averaged over alert bins, the leading groups were {p['incident_linked']} in incident-linked "
            f"alerts and {p['benchmark_false']} in benchmark-false ones. {d['bench.selection']}, so these are not independent "
            "held-out estimates.")
    return finding("F-DISCRIMINATION", "states how well alerts separate documented incidents, with the selection caveat; retrospective",
                   list(d), {}, text, vis="retro")


RULES = [margin, ewma_memory, evidence_timing, cf_nonunique, cf_persistence, train_range, gap, candidates, identifiability, bridge,
         label, timing, selection, alert_burden, discrimination]


def ledger(R: dict, ctx: dict) -> list[dict]:
    return [rule(R, ctx) for rule in RULES]
