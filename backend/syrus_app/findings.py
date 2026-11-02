"""Findings ledger: deterministic rules over one case's records.

Each finding has a stable ID, the rule in words, the record IDs it reads, a structured output, whether it
fires, its visibility, and English text built only from the records' display strings.
"""
from itertools import pairwise

from backend.syrus_app.records import minutes, pct


def finding(id, rule, inputs, output, text, fires=True, vis="both"):
    return {"id": id, "rule": rule, "inputs": inputs, "output": output, "fires": fires, "visibility": vis, "text": text}


def _join(items, word="or"):
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {word} " + items[-1]


def margin(R, ctx):
    m, cut = R["alert.margin"]["value"], ctx["marginal_margin"]
    d = {i: R[i]["display"] for i in ("alert.smoothed", "alert.threshold", "alert.margin")}
    text = (f"The smoothed score ({d['alert.smoothed']}) exceeded the threshold ({d['alert.threshold']}) by "
            f"{d['alert.margin']}" + (", a marginal alert." if m < cut else "."))
    return finding("F-MARGIN", f"margin = smoothed / threshold - 1; called marginal below {pct(cut, 0)}",
                   list(d), {"margin": m, "marginal": m < cut}, text)


def carryover(R, ctx):
    raw, thr = R["alert.raw"]["value"], R["alert.threshold"]["value"]
    ins, out = ["alert.raw", "alert.threshold", "alert.raw_ratio"], {"raw_below_threshold": raw < thr}
    if raw < thr:
        text = (f"The alert bin's own raw score was {R['alert.raw']['display']} ({R['alert.raw_ratio']['display']}); "
                "the smoothed score above the threshold was carried from earlier bins by EWMA memory.")
    else:
        text = f"The alert bin's own raw score was {R['alert.raw']['display']} ({R['alert.raw_ratio']['display']}), so the alert does not rest on carried-over evidence alone."
    if prev := R.get("alert.prev_smoothed"):
        ins.append("alert.prev_smoothed")
        out["prev_above_threshold"] = prev["value"]["score"] >= thr
        text += f" The smoothed score one bin earlier, at {prev['display']['time']}, follows from the EWMA update as {prev['display']['score']}."
    return finding("F-CARRYOVER", "fires when the raw score at the alert bin is below the threshold", ins, out, text, fires=raw < thr)


def emerge(R, ctx):
    rows, groups = [R[i] for i in ctx["temporal_ids"]], ctx["display_groups"]
    steps = [(b["value"]["percentiles"][g] - a["value"]["percentiles"][g], g, a, b)
             for a, b in pairwise(rows) for g in groups]
    rise, g, a, b = max(steps, key=lambda s: s[0])
    first, dom = rows[0], groups[0]
    gap = "" if minutes(a["value"]["time"], b["value"]["time"]) <= ctx["bin_minutes"] else " across a data gap"
    if rise >= 0.5 and b["value"]["percentiles"][g] >= 0.99:
        jumps = [s for s in steps if s[2] is a and s[0] >= 0.5 and s[3]["value"]["percentiles"][s[1]] >= 0.99]
        parts = [f"{s[1]} went from percentile {a['display']['percentiles'][s[1]]} to {b['display']['percentiles'][s[1]]}" for s in jumps]
        out = {"pattern": "abrupt", "from": a["value"]["time"], "at": b["value"]["time"], "groups": [s[1] for s in jumps]}
        text = (f"Smoothed evidence rose abruptly between {a['display']['time']} and {b['display']['time']}{gap}: "
                + "; ".join(parts) + ". This shows when detector evidence appeared, not when a fault began.")
    elif first["value"]["percentiles"][dom] >= 0.99:
        out = {"pattern": "elevated_at_start", "at": first["value"]["time"], "groups": [dom]}
        text = (f"{dom} evidence was already at calibration percentile {first['display']['percentiles'][dom]} at the window start "
                f"({first['display']['time']}), so it did not emerge within this window.")
    else:
        out = {"pattern": "gradual", "from": a["value"]["time"], "at": b["value"]["time"], "groups": [g]}
        text = (f"No abrupt one-bin emergence: the largest rise was {g}, from percentile {a['display']['percentiles'][g]} at "
                f"{a['display']['time']} to {b['display']['percentiles'][g]} at {b['display']['time']}{gap}.")
    rule = ("largest one-bin rise in smoothed-contribution calibration percentile among displayed groups; abrupt when "
            "the rise is at least 0.50 and ends at or above 0.99; elevated at start when the dominant group starts at or above 0.99")
    return finding("F-EMERGE", rule, ["temporal.window", *ctx["temporal_ids"]], out, text)


def cf_specificity(R, ctx):
    ids = ctx["cf_ids"]
    clear = [R[i]["value"]["group"] for i in ids if R[i]["value"]["cleared"]]
    scores = [R[f"cf.{g}"]["display"]["repaired_smoothed"] for g in clear]
    thr = R["alert.threshold"]["display"]
    if len(clear) > 1:
        text = (f"Removing the model-space evidence of any one of {_join(clear)} clears this alert (repaired smoothed scores "
                f"{_join(scores, 'and')} against the {thr} threshold), so the detector's decision is not specific to {ctx['dominant']}.")
    else:
        text = f"Only the {clear[0]} repair clears this alert (repaired smoothed score {scores[0]} against the {thr} threshold)."
    text += " This tests detector dependence, not a physical repair."
    return finding("F-CF-SPEC", "lists single-group model-space repairs that clear the alert; non-specific when more than one does",
                   ["alert.threshold", *ids], {"clearing": clear}, text, fires=len(clear) > 1)


def cf_persistence(R, ctx):
    thr = R["alert.threshold"]["value"]
    hits = [i for i in ctx["cf_ids"] if R[i]["value"]["cleared"] and R[i]["value"]["repaired_smoothed"] >= thr]
    if hits:
        text = " ".join(f"Repairing {R[i]['value']['group']} left the smoothed score at {R[i]['display']['repaired_smoothed']}, "
                        f"still above the {R['alert.threshold']['display']} threshold, yet the alert cleared because fewer than 3 "
                        "of the last 4 bins stayed above the threshold." for i in hits)
    else:
        text = "Every repair that clears the alert also brings the smoothed score below the threshold."
    return finding("F-CF-PERSIST", "fires when a clearing repair leaves the smoothed score at or above the threshold",
                   ["alert.threshold", "detector.persistence", *ctx["cf_ids"]], {"groups": [R[i]["value"]["group"] for i in hits]}, text,
                   fires=bool(hits))


def hyp_tie(R, ctx):
    ids = ctx["hyp_ids"]
    text = "All candidate explanations in play remain open; the loaded evidence does not rank or separate them."
    return finding("F-HYP-TIE", "counts candidates in play; candidates are never ranked", ids, {"open": len(ids)}, text)


def scale(R, ctx):
    ms = [R[i] for i in ctx["measured_ids"]]
    rows = []
    for m in ms:
        v = m["value"]
        pos = "below" if v["value"] < v["q1"] else "above" if v["value"] > v["q3"] else "inside"
        rows.append({"feature": m["value"]["feature"], "iqr": v["q3"] - v["q1"], "position": pos})
    tight = min(ms, key=lambda m: m["value"]["q3"] - m["value"]["q1"])
    td = tight["display"]
    text = (f"Training-period ranges are narrow for some features (for example {tight['value']['feature']}: interquartile range "
            f"{td['q1']} to {td['q3']}), so small physical changes become large scaled residuals.")
    clauses = []
    for pos in ("inside", "above", "below"):
        if feats := [r["feature"] for r in rows if r["position"] == pos]:
            clauses.append(f"{_join(feats, 'and')} {'was' if len(feats) == 1 else 'were'} {pos} the training interquartile range")
    text += " At the alert bin, " + "; ".join(clauses) + "."
    return finding("F-SCALE", "reports training interquartile ranges of the case's measured-context features and where the alert-bin value falls",
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


def identifiability(R, ctx):
    ins = ["unavailable:demand_measurement", *(i for i in ("hyp.air_leakage", "hyp.downstream_demand") if i in R)]
    text = ("Leakage and higher downstream demand both raise total outflow (total outflow = demand + leakage). Without an "
            "independent demand measurement this data cannot tell them apart.")
    return finding("F-IDENT", "always present: leak and demand are not separately identifiable without a demand observation", ins, {}, text)


def bridge(R, ctx):
    b = R["bridge.status"]
    text = f"Real-asset physical reasoning and actions are withheld. Missing: {'; '.join(b['display']['missing'])}."
    return finding("F-BRIDGE", "always present while the physical bridge is not validated", ["bridge.status"], {"status": b["value"]["status"]}, text)


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


def status(R, ctx):
    text = ("Documented incident in the provider's failure list. This alert was chosen for explanation because it falls in that "
            "incident's timely window, so the case set is selected by the labels.")
    return finding("F-STATUS", "states the case status and how the case was selected; retrospective", ["case.status"], {}, text, vis="retro")


def bench(R, ctx):
    d = {i: R[i]["display"] for i in ("bench.false_rate", "bench.episodes", "bench.pr_auc", "bench.exposure", "bench.selection")}
    e = d["bench.episodes"]
    text = (f"Over {d['bench.exposure']} of valid TEST exposure the detector raised {d['bench.false_rate']} "
            f"({e['benchmark_false']} of {e['total']} episodes; PR-AUC {d['bench.pr_auc']}). Benchmark-false means not linked "
            f"to a documented incident, not physically normal. {d['bench.selection']}, so these figures are not an independent "
            "held-out estimate.")
    return finding("F-BENCH", "states the frozen TEST benchmark and its caveats; retrospective", list(d), {}, text, vis="retro")


RULES = [margin, carryover, emerge, cf_specificity, cf_persistence, hyp_tie, scale, gap, identifiability, bridge,
         label, timing, status, bench]


def ledger(R: dict, ctx: dict) -> list[dict]:
    return [rule(R, ctx) for rule in RULES]
