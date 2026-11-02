"""Deterministic case builder: frozen artifacts and Syrus configs in, one as-of and one retrospective case file per case out.

Reads only frozen JSON reports, YAML configs and (for code pointers) source files. It never imports `ml/` or
recomputes detector values; derived records are arithmetic or logic on frozen values and keep their formula and
input record IDs.
"""
from itertools import pairwise, product

from backend.syrus_app.findings import ledger
from backend.syrus_app.records import (
    SCHEMA,
    Sources,
    count,
    derived,
    hhmm,
    listing,
    minute,
    minutes,
    num,
    pct,
    pctile,
    record,
    resolve,
    score,
    sha,
    signed_pct,
)

XAI, VER, BENCH = "docs/xai_report.json", "docs/xai_verification_report.json", "docs/detector_benchmark.json"
PROOF = "docs/proof_manifest.json"
METRO, DET, BRIDGE = "configs/metropt.yaml", "configs/detection.yaml", "configs/evidence_action_bridge.yaml"
CASES, HYPS, DISCS = "configs/syrus_cases.yaml", "configs/syrus_hypotheses.yaml", "configs/syrus_discriminators.yaml"
FEATURES_PY, PCA_PY = "ml/data/features.py", "ml/explainability/pca.py"
ALERTS_PY, VERIFY_PY, XAI_NB = "ml/detection/alerts.py", "ml/explainability/verification.py", "notebooks/04_detector_xai.ipynb"

ANALOG_STATS, DIGITAL_STATS, RELATION_STATS = ["mean", "std", "min", "max", "last"], ["active_ratio", "transitions", "last"], ["mean", "std"]
RELATIONS = ["tp3_minus_reservoirs", "tp2_minus_tp3"]
CONDITIONS = {"air_leak_high_stress": "Air leak, high stress"}
MISSING = {
    "site_calibrated_receiver_and_compressor_model": "a site-calibrated receiver and compressor model",
    "calibrated_compressor_inflow_observation": "a calibrated compressor inflow observation",
    "receiver_pressure_trace": "a receiver pressure trace suitable for inverse physics",
}
UNAVAILABLE = {
    "unavailable:definitions": "Signal definitions and units are not loaded: the dataset description has not passed the document gate.",
    "unavailable:documents": "No OEM manual, SOP, maintenance ticket or other document for this asset is loaded.",
    "unavailable:raw_series": "Raw telemetry and the raw-score series around the alert are not in the frozen artifacts; only the alert bin's raw score is.",
    "unavailable:demand_measurement": "No independent measurement of downstream air demand is available for this asset, and the bridge configuration records no calibrated compressor inflow observation.",
}


def pair_limits(tests: list[dict]) -> list[dict]:
    """A pair is not separable here when every test that compares it needs an external observation."""
    by_pair = {}
    for t in tests:
        for a, b in t["compares"]:
            by_pair.setdefault(tuple(sorted((a, b))), []).append(t)
    return [{"pair": list(p), "limit": "not_separable", "tests": [t["id"] for t in ts]}
            for p, ts in by_pair.items() if all(o["availability"] == "external" for t in ts for o in t["observations"])]


def hit_note(times: list[str], known: dict[str, bool], need: int, window: int, reset: int, merge: int) -> dict:
    """What the persistence rule implies about threshold hits before an alert's first bin.

    `times` are the observed bins ending at the alert bin and `known` the bins whose hit is known. Every hit
    pattern over the bins since the last restart is kept if the last bin alerts and no bin within `merge`
    minutes before it does; a bin is reported only when all kept patterns agree on it. Conditions whose
    persistence history would reach before the first observed bin are skipped, which keeps the result sound.
    """
    start = max([0] + [k for k in range(1, len(times)) if minutes(times[k - 1], times[k]) > reset])
    bins, t = times[start:], times[-1]
    free = [b for b in bins if b not in known]

    def alerting(h, k):
        if start == 0 and k + 1 < window:
            return None
        return sum(h[b] for b in bins[max(0, k + 1 - window):k + 1]) >= need

    quiet = [k for k in range(len(bins) - 1) if minutes(bins[k], t) <= merge]
    kept = []
    for bits in product((False, True), repeat=len(free)):
        h = {**known, **dict(zip(free, bits, strict=True))}
        if alerting(h, len(bins) - 1) and not any(alerting(h, k) for k in quiet):
            kept.append(h)
    if not kept:
        raise ValueError(f"no hit pattern makes {t} the first alerting bin")
    forced = {b: kept[0][b] for b in bins if all(h[b] == kept[0][b] for h in kept)}
    recent = bins[-(window + 1):]
    open_ = [b for b in recent if b not in forced]
    counts = sorted({sum(h[b] for b in open_) for h in kept})
    return {"restart": times[start] if start else None, "hits": [b for b in recent if forced.get(b) is True],
            "misses": [b for b in recent if forced.get(b) is False], "undetermined": open_, "hit_counts": counts}


def hit_display(note: dict) -> str:
    parts = []
    if note["hits"]:
        parts.append(f"{listing([hhmm(b) for b in note['hits']])} {'was' if len(note['hits']) == 1 else 'were'} at or above the threshold")
    if note["undetermined"]:
        names = listing([hhmm(b) for b in note["undetermined"]])
        k = note["hit_counts"]
        parts.append(f"exactly {'one' if k == [1] else k[0]} of {names} {'was' if k == [1] else 'were'}" if len(k) == 1 else f"{names} cannot be determined")
    if note["misses"]:
        parts.append(f"{listing([hhmm(b) for b in note['misses']])} {'was' if len(note['misses']) == 1 else 'were'} below it")
    text = "; ".join(parts) + "."
    if note["restart"]:
        text += f" Persistence history restarted at {hhmm(note['restart'])} after a gap."
    return text[0].upper() + text[1:]


def group_features(g: str, cfg_metro: dict) -> list[str]:
    if g == "pressure_relationships":
        return [f"{r}__{s}" for r in RELATIONS for s in RELATION_STATS]
    stats = ANALOG_STATS if g in cfg_metro["dataset"]["analog_columns"] else DIGITAL_STATS
    return [f"{g}__{s}" for s in stats]


def build(case_cfg: dict, src: Sources) -> tuple[dict, dict]:
    cfg, hyp_cfg, test_cfg = src.load(CASES), src.load(HYPS), src.load(DISCS)
    cid, i = case_cfg["id"], case_cfg["source"]["incident_id"]
    inc = f"/incidents[id={i}]"
    R = {}

    def add(r):
        R[r["id"]] = r

    def got(rel, path):
        return src.get(rel, path)

    # Alert and detector mechanics.
    t, p = got(VER, f"{inc}/explanation_timestamp")
    add(record("alert.time", "model_evidence", t, f"{minute(t)} (dataset local time)",
               [p, src.code(XAI_NB, "we explain the first alert episode whose **start time** is inside the Phase 2 timely window")],
               caveat="Start of the first five-minute bin of this alert."))
    thr, p = got(VER, f"{inc}/threshold")
    add(record("alert.threshold", "model_evidence", thr, score(thr), [p], caveat="99.5th percentile of the smoothed score over the calibration period only."))
    s, p = got(VER, f"{inc}/original_smoothed_score")
    add(record("alert.smoothed", "model_evidence", s, score(s), [p]))
    raw, p = got(XAI, f"{inc}/raw_score")
    add(record("alert.raw", "model_evidence", raw, score(raw), [p],
               caveat="Mean squared PCA residual over 63 robust-scaled features for the alert bin alone, before smoothing."))
    add(derived("alert.margin", s / thr - 1, signed_pct(s / thr - 1), ["alert.smoothed", "alert.threshold"], "smoothed / threshold - 1", R))

    alpha, pa = got(DET, "/alerting/ewma_alpha")
    reset, pr = got(DET, "/alerting/reset_gap_minutes")
    add(record("detector.ewma", "model_evidence", {"alpha": alpha, "reset_gap_minutes": reset},
               {"alpha": f"{alpha:.2f}", "rule": f"smoothed = {alpha:.2f} × raw + {1 - alpha:.2f} × previous smoothed; restarts after a gap over {reset} min"},
               [pa, pr, src.code(ALERTS_PY, "causal_ewma")]))
    hits, ph = got(DET, "/alerting/persistence_hits")
    window, pw = got(DET, "/alerting/persistence_window")
    merge, pm = got(DET, "/alerting/merge_minutes")
    add(record("detector.persistence", "model_evidence", {"hits": hits, "window": window, "merge_minutes": merge},
               f"alert when {hits} of the last {window} smoothed scores are at or above the threshold; alerting bins up to {merge} min apart form one alert",
               [ph, pw, pm, src.code(ALERTS_PY, "persistent_alerts"), src.code(ALERTS_PY, "extract_alert_episodes")],
               caveat="The history of the last scores is cleared after a gap, so an alert needs that many hits since the last restart. Smoothed hits need not come from separate anomalous bins: one large raw score can produce several."))
    n_cal, p = got(BENCH, "/evaluation_protocol/calibration_rows")
    add(record("detector.calibration_bins", "model_evidence", n_cal, count(n_cal), [p], caveat="Valid five-minute bins in the calibration period."))

    # Contributions.
    ranked = resolve(src.load(VER), f"{inc}/ranked_groups")
    floor = cfg["display_floor"]
    shown = [g["group"] for g in ranked if g["share"] >= floor]
    for g in shown:
        base = f"{inc}/ranked_groups[group={g}]"
        (sh, p1), (pc, p2), (c, p3) = (got(VER, f"{base}/{k}") for k in ("share", "calibration_percentile", "smoothed_contribution"))
        add(record(f"contrib.{g}", "model_evidence", {"group": g, "share": sh, "percentile": pc, "contribution": c},
                   {"share": pct(sh), "percentile": pctile(pc), "contribution": score(c)}, [p1, p2, p3],
                   caveat="Share of the smoothed PCA anomaly-evidence score, and calibration percentile of this group's smoothed contribution. Not a probability or a cause."))
        k = pc * n_cal
        if abs(k - round(k)) < 1e-6 and pc < 1:
            hi = n_cal - round(k)
            add(derived(f"calrank.{g}", {"higher": hi, "n": n_cal}, f"{count(hi)} of {count(n_cal)}", [f"contrib.{g}", "detector.calibration_bins"],
                        "calibration_bins × (1 - percentile); mid-rank percentile with no exact ties", R,
                        caveat="Calibration bins whose smoothed contribution for this group was higher."))
        elif pc == 1:
            add(derived(f"calrank.{g}", {"higher": 0, "n": n_cal}, f"0 of {count(n_cal)}", [f"contrib.{g}", "detector.calibration_bins"],
                        "percentile 1.0: above every calibration bin", R, caveat="Calibration bins whose smoothed contribution for this group was higher."))
    rest = sum(g["share"] for g in ranked if g["share"] < floor)
    _, p = got(VER, f"{inc}/ranked_groups")
    add(record("contrib.others", "model_evidence", rest, pct(rest) if rest >= 0.001 else "<0.1%", [p],
               caveat=f"Combined share of groups below {pct(floor, 0)}; their percentiles are not meaningful at near-zero contribution."))
    for g in shown:
        add(record(f"group.{g}", "model_evidence", {"group": g, "features": group_features(g, src.load(METRO))},
                   {"features": ", ".join(group_features(g, src.load(METRO)))},
                   [got(METRO, "/dataset/analog_columns")[1], got(METRO, "/dataset/digital_columns")[1],
                    src.code(FEATURES_PY, "aggregate_causal_bins"), src.code(PCA_PY, "feature_group")],
                   caveat="Engineered five-minute features that make up this signal group.", definition="unavailable:definitions"))

    # Past-only temporal window.
    te = f"{inc}/temporal_evidence"
    rows = resolve(src.load(VER), f"{te}/rows")
    times = [r["timestamp"] for r in rows]
    gaps = [{"after": a, "before": b, "minutes": minutes(a, b)} for a, b in pairwise(times) if minutes(a, b) > 5]
    (req, p1), (obs, p2) = got(VER, f"{te}/window_bins_requested"), got(VER, f"{te}/window_rows_observed")
    add(record("temporal.window", "model_evidence",
               {"requested": req, "observed": obs, "start": times[0], "end": times[-1], "groups": shown, "gaps": gaps},
               {"window": f"{hhmm(times[0])} to {hhmm(times[-1])}", "bins": f"{obs} of {req}",
                "gaps": [f"{hhmm(g['after'])} and {hhmm(g['before'])} ({g['minutes']} min apart)" for g in gaps]}, [p1, p2],
               caveat="Past-only window of smoothed-contribution percentiles ending at the alert bin. It shows smoothed evidence, not when a fault began."))
    temporal_ids = []
    for ts in times:
        tid = f"temporal.{ts[11:13]}{ts[14:16]}"
        vals = {g: got(VER, f"{te}/rows[timestamp={ts}]/percentiles/{g}") for g in shown}
        add(record(tid, "model_evidence", {"time": ts, "percentiles": {g: v for g, (v, _) in vals.items()}},
                   {"time": hhmm(ts), "percentiles": {g: pctile(v) for g, (v, _) in vals.items()}}, [p for _, p in vals.values()]))
        temporal_ids.append(tid)

    # Smoothing memory. An alert bin always follows a bin within the reset gap: after a restart one hit cannot alert.
    if minutes(times[-2], times[-1]) > reset:
        raise ValueError(f"{cid}: the bin before the alert is more than {reset} min earlier")
    prev = (s - alpha * raw) / (1 - alpha)
    add(derived("alert.prev_smoothed", {"time": times[-2], "score": prev}, {"time": hhmm(times[-2]), "score": score(prev)},
                ["alert.smoothed", "alert.raw", "detector.ewma", temporal_ids[-2]],
                "(smoothed - alpha × raw) / (1 - alpha); valid because the previous bin is within the reset gap", R,
                caveat="Arithmetic on frozen values, not an exported score."))
    mem = 1 - alpha * raw / s
    add(derived("alert.memory", mem, pct(mem, 2), ["alert.raw", "alert.smoothed", "detector.ewma"], "1 - alpha × raw / smoothed", R,
                caveat=f"Share of the alert-bin smoothed score carried from earlier bins by smoothing. {pct(1 - alpha, 0)} means the score is neither rising nor falling. It does not show which earlier bins carried the evidence."))
    note = hit_note(times, {times[-1]: s >= thr, times[-2]: prev >= thr}, hits, window, reset, merge)
    add(derived("alert.hits", note, hit_display(note),
                ["alert.time", "alert.smoothed", "alert.threshold", "alert.prev_smoothed", "detector.persistence", "temporal.window"],
                "hit patterns consistent with the persistence rule, the known hits, and this bin being the first alerting bin", R,
                caveat="Technical note. Inferred from the persistence rule; which earlier bin was a hit is not exported."))

    # Counterfactual repairs.
    (meth, p1), (interp, p2) = got(VER, "/method/counterfactual_repair"), got(VER, "/method/counterfactual_interpretation")
    add(record("detector.repair", "model_evidence", {"method": meth, "interpretation": interp}, f"{meth} {interp}",
               [p1, p2, src.code(VERIFY_PY, "verify_counterfactual_alert"), src.code(VERIFY_PY, "score_with_feature_group_repair")],
               caveat="Only the groups with the largest smoothed contributions at the alert bin are repaired, one at a time."))
    cf_ids = []
    for c in resolve(src.load(VER), f"{inc}/single_group_repairs"):
        g = c["repaired_groups"][0]
        base = f"{inc}/single_group_repairs[repaired_groups={g}]"
        keys = ("window_start", "window_end", "window_rows", "original_raw_score", "original_smoothed_score", "original_alert",
                "repaired_raw_score", "repaired_smoothed_score", "repaired_alert", "alert_cleared")
        vals = {k: got(VER, f"{base}/{k}") for k in keys}
        own, po = got(VER, f"{inc}/ranked_groups[group={g}]/smoothed_contribution")
        share, ps = got(VER, f"{inc}/ranked_groups[group={g}]/share")
        v = {k: x for k, (x, _) in vals.items()}
        rs, cl = v["repaired_smoothed_score"], v["alert_cleared"]
        path = ("score" if rs < thr else "persistence") if cl else None
        red = (v["original_smoothed_score"] - rs) / own if own else None
        add(record(f"cf.{g}", "model_evidence",
                   {"group": g, **v, "own_contribution": own, "share": share, "path": path, "reduction_ratio": red},
                   {"window": f"{hhmm(v['window_start'])} to {hhmm(v['window_end'])} ({v['window_rows']} bins)",
                    "repaired_smoothed": score(rs), "repaired_raw": score(v["repaired_raw_score"]), "own_contribution": score(own),
                    "result": {"score": "clears the alert: smoothed score below the threshold",
                               "persistence": "clears the alert: persistence rule no longer met", None: "alert remains"}[path],
                    "ratio": f"{red:.2f}×" if share >= floor else "not shown: contribution below the display floor"},
                   [*(p for _, p in vals.values()), po, ps],
                   caveat="Model-space repair over the window shown: this group's robust-scaled coordinates are replaced by the PCA "
                          "reconstruction, each bin is re-projected and rescored, and the series passes through the same smoothing, "
                          "threshold and persistence rule. Re-projection changes other groups' residuals too. Checked at the alert "
                          "bin only. Tests the detector, not a physical repair.",
                   derivation={"formula": "path = score when the repaired smoothed score is below the threshold, persistence when the alert "
                                          "clears with it at or above; reduction_ratio = (original smoothed - repaired smoothed) / own smoothed contribution",
                               "inputs": ["alert.threshold", "detector.repair"]}))
        cf_ids.append(f"cf.{g}")
    restart = next((g["before"] for g in gaps if g["minutes"] > reset), None)
    low = min(R[c]["value"]["repaired_smoothed_score"] for c in cf_ids)
    bound = 0.0 if restart else low / s
    add(derived("cf.prewindow", {"max_share": bound, "restart": restart},
                f"none, because smoothing restarts at {hhmm(restart)} inside the window" if restart else f"at most {pct(bound)}",
                [*cf_ids, "alert.smoothed", "temporal.window", "detector.ewma"],
                "lowest repaired smoothed / smoothed, because the smoothing state before the window is not edited and repaired raw "
                "scores are non-negative; zero when smoothing restarts inside the window", R,
                caveat="Upper bound on how much of the alert-bin smoothed score comes from the unedited smoothing state before the repair window."))

    # Measured context at the alert bin.
    measured_ids = []
    for m in src.load(XAI)["diagnostics"]["physical_context"]:
        if m["incident_id"] != i:
            continue
        f = m["feature"]
        base = f"/diagnostics/physical_context[incident_id={i},feature={f}]"
        vals = {k: got(XAI, f"{base}/{k}") for k in ("value", "training_q1", "training_median", "training_q3")}
        v = {k.removeprefix("training_"): x for k, (x, _) in vals.items()}
        add(record(f"measured.{f}", "measured", {"feature": f, **v, "unit": None},
                   {**{k: num(x) for k, x in v.items()}, "unit": "unit not loaded"}, [p for _, p in vals.values()],
                   caveat="Five-minute feature value at the alert bin with training-period quartiles; only the three features with the largest "
                          "contributions at the alert bin are exported. Units are defined in the dataset description, which is not loaded.",
                   definition="unavailable:definitions"))
        measured_ids.append(f"measured.{f}")

    # Unavailable information.
    for uid, text in UNAVAILABLE.items():
        if uid == "unavailable:demand_measurement":
            v, p = got(BRIDGE, "/bridge/calibrated_compressor_inflow_available")
            add(record(uid, "unavailable", {"calibrated_compressor_inflow_available": v}, text, [p]))
        else:
            add(record(uid, "unavailable", None, text, []))

    # Physical bridge.
    (st, p1), (miss, p2), (avail, p3) = (got(PROOF, f"/real_evidence_path/{k}") for k in ("bridge_status", "missing_requirements", "physical_bridge_available"))
    flags = {k: got(BRIDGE, f"/bridge/{k}") for k in ("site_calibrated", "calibrated_compressor_inflow_available", "receiver_pressure_trace_available")}
    add(record("bridge.status", "bridge_status",
               {"status": st, "physical_bridge_available": avail, "missing": miss, **{k: v for k, (v, _) in flags.items()}},
               {"status": "Withheld", "missing": [MISSING[m] for m in miss]}, [p1, p2, p3, *(p for _, p in flags.values())],
               caveat="Asset-level gate from the bridge configuration and proof manifest; it applies to every MetroPT-3 case."))

    # Candidate explanations and tests: every candidate for every case, unranked, status open.
    tests = [{"id": t_["id"], "compares": t_["compares"],
              "observations": [{**o, "availability_text": test_cfg["availability"][o["availability"]]} for o in t_["observations"]],
              "maturity": t_["maturity"], "maturity_text": test_cfg["maturity"][t_["maturity"]], "rule": t_["rule"],
              "result": t_["result"], "result_text": test_cfg["results"][t_["result"]], "follow_up": t_["follow_up"]}
             for t_ in test_cfg["tests"]]
    limits = [{**lim, "text": test_cfg["pair_limits"][lim["limit"]]} for lim in pair_limits(tests)]
    hyp_ids = []
    for h in hyp_cfg["hypotheses"]:
        hid, groups = h["id"], h["groups"]
        related = [g for g in shown if groups != "any" and g in groups]
        partners = sorted(x for lim in limits if hid in lim["pair"] for x in lim["pair"] if x != hid)
        add(record(f"hyp.{hid}", "hypothesis",
                   {"id": hid, "layer": h["layer"], "status": "open", "groups": groups, "related": related,
                    "tests": [t_["id"] for t_ in tests if any(hid in pr for pr in t_["compares"])], "not_separable_from": partners},
                   {"label": h["label"]["en"], "layer": hyp_cfg["layers"][h["layer"]]["en"], "status": "open",
                    "related": ", ".join(related) or "no specific signal group"},
                   [got(HYPS, f"/hypotheses[id={hid}]")[1], {"file": DISCS, "path": "/tests", "sha256": src.hashes[DISCS]}],
                   caveat=f"Candidate explanation, unranked; {test_cfg['candidate_status']['open']['en']}. Not a diagnosis.",
                   derivation={"formula": "status open until a reviewed test returns a result; related = catalog groups among this case's displayed groups",
                               "inputs": [f"contrib.{g}" for g in related]}))
        hyp_ids.append(f"hyp.{hid}")

    # Retrospective records.
    lab, p = got(METRO, f"/dataset/reported_incidents[id={i}]")
    add(record("label.incident", "dataset_label", {"condition": lab["condition"], "start": lab["start"], "end": lab["end"]},
               {"condition": CONDITIONS[lab["condition"]], "interval": f"{minute(lab['start'])} to {minute(lab['end'])}"}, [p], vis="retro",
               caveat="Transcribed in configs/metropt.yaml from the provider's failure list. The original document is not loaded, so this record cannot cite it yet."))
    (tm, p1), (lead, p2), (delay, p3) = (got(XAI, f"{inc}/{k}") for k in ("timing", "lead_hours", "delay_hours"))
    hours = lead if tm == "pre_onset" else delay
    add(record("label.timing", "dataset_label", {"timing": tm, "lead_hours": lead, "delay_hours": delay},
               f"{hours:.2f} h {'before' if tm == 'pre_onset' else 'after'}", [p1, p2, p3], vis="retro"))
    ctype, p = got(CASES, f"/cases[id={cid}]/source/type")
    add(record("case.status", "case_status", ctype, "Documented incident", [p], vis="retro",
               caveat="The explained alerts were selected because they fall in documented incident windows."))

    b = "/details/pca_robust"
    n_inc = len(resolve(src.load(BENCH), f"{b}/incident_results"))
    for rid, key in (("bench.timely", "timely_incident_recall"), ("bench.preonset", "pre_onset_incident_recall")):
        v, p = got(BENCH, f"{b}/{key}")
        add(record(rid, "benchmark_stat", {"recall": v, "n": n_inc}, f"{round(v * n_inc)} of {n_inc}", [p, got(BENCH, f"{b}/incident_results")[1]], vis="retro"))
    v, p = got(BENCH, f"{b}/false_alerts_per_24h")
    add(record("bench.false_rate", "benchmark_stat", v, f"{v:.3f} benchmark-false alert episodes per 24 h", [p], vis="retro",
               caveat="Benchmark-false means not linked to a documented incident window; it is not a confirmed false alarm or confirmed healthy behaviour."))
    eps = {k: got(XAI, f"/episodes/{k}") for k in ("total", "incident_linked", "benchmark_false")}
    add(record("bench.episodes", "benchmark_stat", {k: x for k, (x, _) in eps.items()}, {k: count(x) for k, (x, _) in eps.items()},
               [p for _, p in eps.values()], vis="retro"))
    v, p = got(BENCH, f"{b}/pr_auc")
    add(record("bench.pr_auc", "benchmark_stat", v, f"{v:.3f}", [p], vis="retro"))
    v, p = got(BENCH, f"{b}/valid_exposure_days")
    add(record("bench.exposure", "benchmark_stat", v, f"{v:.1f} days", [p], vis="retro"))
    summary, p = got(BENCH, "/summary")
    add(record("bench.selection", "benchmark_stat", {"detectors": sorted(summary), "period": "TEST"},
               f"The same TEST period was used to compare {len(summary)} candidate detectors", [p], vis="retro"))
    prof = {k: got(XAI, f"/global_profiles/{k}_top_groups") for k in ("incident_linked", "benchmark_false")}
    add(record("bench.profiles", "benchmark_stat", {k: x[:3] for k, (x, _) in prof.items()},
               {k: ", ".join(f"{g['group']} {pct(g['share'])}" for g in x[:3]) for k, (x, _) in prof.items()},
               [p for _, p in prof.values()], vis="retro",
               caveat="Mean normalised smoothed group contributions over all alert bins of each episode type, so longer episodes weigh more. "
                      "Benchmark-false does not mean physically normal."))

    ctx = {"display_groups": shown, "temporal_ids": temporal_ids, "cf_ids": cf_ids, "hyp_ids": hyp_ids,
           "measured_ids": measured_ids, "bin_minutes": 5, "reset_gap": reset}
    F = ledger(R, ctx)

    head = {"schema": SCHEMA, "case_id": cid, "alert_time": t, "asset": cfg["asset"]}
    body = {"tests": tests, "pair_limits": limits}
    retro = {**head, "mode": "retro", "source": case_cfg["source"], "role": case_cfg["role"],
             "records": list(R.values()), "findings": F, **body}
    asof = {**head, "mode": "asof", "records": [_asof(r) for r in R.values() if r["visibility"] == "both"],
            "findings": [f for f in F if f["visibility"] == "both"], **body}
    return _sealed(asof), _sealed(retro)


def _asof(r: dict) -> dict:
    """As-of records drop source paths and symbols, which could reveal the incident number; values and file hashes stay."""
    return {**r, "source": [{"file": p["file"], "sha256": p["sha256"]} for p in r["source"]]}


def _sealed(case: dict) -> dict:
    return {**case, "sha256": sha(case)}


def build_all(src: Sources) -> tuple[list[tuple[dict, dict]], dict]:
    cfg = src.load(CASES)
    built = sorted((build(c, src) for c in cfg["cases"]), key=lambda pair: pair[0]["alert_time"])
    manifest = {
        "schema": "syrus.manifest.v2",
        "cases": [{"case_id": a["case_id"], "alert_time": a["alert_time"],
                   "files": {m: {"path": f"{a['case_id']}.{m}.json", "sha256": c["sha256"]} for m, c in (("asof", a), ("retro", r))}}
                  for a, r in built],
        "documents": {"loaded": False, "index_sha256": None},
        "inputs": dict(sorted(src.hashes.items())),
    }
    return built, _sealed(manifest)
