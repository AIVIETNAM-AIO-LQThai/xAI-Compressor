"""Deterministic case builder: frozen artifacts and Syrus configs in, one as-of and one retrospective case file per case out.

Reads only frozen JSON reports and YAML configs. It never imports `ml/` or recomputes detector values;
derived records are arithmetic on frozen values and keep their formula and input record IDs.
"""
from itertools import pairwise

from backend.syrus_app.findings import ledger
from backend.syrus_app.records import (
    SCHEMA,
    Sources,
    count,
    derived,
    hhmm,
    minute,
    minutes,
    num,
    pct,
    pctile,
    ratio,
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


def statuses(in_play: list[str], discs: list[dict]) -> dict:
    """untestable_here when every observation separating a candidate from another in-play candidate is absent."""
    pairs = {}
    for d in discs:
        for a, b in d["separates"]:
            pairs.setdefault(frozenset((a, b)), []).append(d["availability"])
    out = {}
    for h in in_play:
        blocked = [o for o in in_play if o != h and (av := pairs.get(frozenset((h, o)))) and all(a == "absent" for a in av)]
        out[h] = ("untestable_here", blocked) if blocked else ("not_discriminated", [])
    return out


def group_features(g: str, cfg_metro: dict) -> list[str]:
    if g == "pressure_relationships":
        return [f"{r}__{s}" for r in RELATIONS for s in RELATION_STATS]
    stats = ANALOG_STATS if g in cfg_metro["dataset"]["analog_columns"] else DIGITAL_STATS
    return [f"{g}__{s}" for s in stats]


def build(case_cfg: dict, src: Sources) -> tuple[dict, dict]:
    cfg, hyp_cfg, disc_cfg = src.load(CASES), src.load(HYPS), src.load(DISCS)
    cid, i = case_cfg["id"], case_cfg["source"]["incident_id"]
    inc = f"/incidents[id={i}]"
    R = {}

    def add(r):
        R[r["id"]] = r

    def got(rel, path):
        return src.get(rel, path)

    # Alert and detector mechanics.
    t, p = got(VER, f"{inc}/explanation_timestamp")
    add(record("alert.time", "model_evidence", t, f"{minute(t)} (dataset local time)", [p],
               caveat="Start of the first five-minute bin of this alert."))
    thr, p = got(VER, f"{inc}/threshold")
    add(record("alert.threshold", "model_evidence", thr, score(thr), [p], caveat="99.5th percentile of the smoothed score over the calibration period only."))
    s, p = got(VER, f"{inc}/original_smoothed_score")
    add(record("alert.smoothed", "model_evidence", s, score(s), [p]))
    raw, p = got(XAI, f"{inc}/raw_score")
    add(record("alert.raw", "model_evidence", raw, score(raw), [p],
               caveat="Mean squared PCA residual over 63 robust-scaled features for the alert bin alone, before smoothing."))
    add(derived("alert.margin", s / thr - 1, signed_pct(s / thr - 1), ["alert.smoothed", "alert.threshold"], "smoothed / threshold - 1", R))
    add(derived("alert.raw_ratio", raw / thr, ratio(raw / thr), ["alert.raw", "alert.threshold"], "raw / threshold", R))

    alpha, pa = got(DET, "/alerting/ewma_alpha")
    reset, pr = got(DET, "/alerting/reset_gap_minutes")
    add(record("detector.ewma", "model_evidence", {"alpha": alpha, "reset_gap_minutes": reset},
               {"alpha": f"{alpha:.2f}", "rule": f"smoothed = {alpha:.2f} × raw + {1 - alpha:.2f} × previous smoothed; restarts after a gap over {reset} min"},
               [pa, pr]))
    hits, ph = got(DET, "/alerting/persistence_hits")
    window, pw = got(DET, "/alerting/persistence_window")
    add(record("detector.persistence", "model_evidence", {"hits": hits, "window": window},
               f"alert when {hits} of the last {window} smoothed scores are at or above the threshold", [ph, pw]))
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
                   caveat="Share of the smoothed PCA anomaly-evidence score, and calibration percentile of this group's smoothed contribution. Not a probability."))
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

    # EWMA back-calculation for the previous bin, valid only when it is within the reset gap.
    if len(times) > 1 and minutes(times[-2], times[-1]) <= reset:
        prev = (s - alpha * raw) / (1 - alpha)
        add(derived("alert.prev_smoothed", {"time": times[-2], "score": prev}, {"time": hhmm(times[-2]), "score": score(prev)},
                    ["alert.smoothed", "alert.raw", "detector.ewma", temporal_ids[-2]],
                    "(smoothed - alpha × raw) / (1 - alpha); valid because the previous bin is within the reset gap", R,
                    caveat="Arithmetic on frozen values, not an exported score."))

    # Counterfactual repairs.
    cf_ids = []
    for c in resolve(src.load(VER), f"{inc}/single_group_repairs"):
        g = c["repaired_groups"][0]
        base = f"{inc}/single_group_repairs[repaired_groups={g}]"
        (rs, p1), (rr, p2), (ra, p3), (cl, p4) = (got(VER, f"{base}/{k}") for k in
                                                  ("repaired_smoothed_score", "repaired_raw_score", "repaired_alert", "alert_cleared"))
        add(record(f"cf.{g}", "model_evidence", {"group": g, "repaired_smoothed": rs, "repaired_raw": rr, "repaired_alert": ra, "cleared": cl},
                   {"repaired_smoothed": score(rs), "repaired_raw": score(rr), "result": "clears the alert" if cl else "alert remains"},
                   [p1, p2, p3, p4],
                   caveat="Model-space repair: this group's coordinates are replaced by the PCA reconstruction and rescored through the same EWMA, threshold and persistence rule. Detector dependence, not a physical repair."))
        cf_ids.append(f"cf.{g}")

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
                   caveat="Five-minute feature value at the alert bin with training-period quartiles. Units are defined in the dataset description, which is not loaded.",
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

    # Candidate explanations and discriminators.
    rule = cfg["evidence_rule"]
    ev = {g["group"] for g in ranked if g["share"] >= rule["min_share"] and g["calibration_percentile"] >= rule["min_percentile"]}
    in_play = [h for h in hyp_cfg["hypotheses"] if h["triggers"] == "any" or ev & set(h["triggers"])]
    st_map = statuses([h["id"] for h in in_play], disc_cfg["discriminators"])
    hyp_ids = []
    for h in in_play:
        hid = h["id"]
        matched = [g for g in shown if h["triggers"] != "any" and g in h["triggers"] and g in ev]
        status, against = st_map[hid]
        add(record(f"hyp.{hid}", "hypothesis", {"id": hid, "status": status, "against": against, "matched_groups": matched},
                   {"label": h["label"]["en"], "status": disc_cfg["statuses"][status]["en"]},
                   [got(HYPS, f"/hypotheses[id={hid}]")[1], {"file": DISCS, "path": "/discriminators", "sha256": src.hashes[DISCS]}],
                   caveat="Candidate explanation, unranked. Not a diagnosis.",
                   derivation={"formula": "in play when a trigger group passes the evidence rule, or always for `any`; status from the discriminator table",
                               "inputs": [f"contrib.{g}" for g in matched]}))
        hyp_ids.append(f"hyp.{hid}")
    play = {h["id"] for h in in_play}
    discs = []
    for d in disc_cfg["discriminators"]:
        pairs = [p for p in d["separates"] if set(p) <= play]
        if pairs:
            discs.append({"id": d["id"], "observation": d["observation"], "separates": pairs, "availability": d["availability"],
                          "availability_text": disc_cfg["availability"][d["availability"]], "question": d["question"]})

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
               caveat="Benchmark-false means not linked to a documented incident window; it does not mean physically normal."))
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
    add(record("bench.unmatched_profile", "benchmark_stat", {k: x[:3] for k, (x, _) in prof.items()},
               {k: ", ".join(f"{g['group']} {pct(g['share'])}" for g in x[:3]) for k, (x, _) in prof.items()},
               [p for _, p in prof.values()], vis="retro",
               caveat="Average contribution shares over all alert episodes of each type. Benchmark-false does not mean physically normal."))

    ctx = {"marginal_margin": cfg["marginal_margin"], "display_groups": shown, "dominant": shown[0], "temporal_ids": temporal_ids,
           "cf_ids": cf_ids, "hyp_ids": hyp_ids, "measured_ids": measured_ids, "bin_minutes": 5, "reset_gap": reset}
    F = ledger(R, ctx)

    head = {"schema": SCHEMA, "case_id": cid, "alert_time": t, "asset": cfg["asset"]}
    retro = {**head, "mode": "retro", "source": case_cfg["source"], "role": case_cfg["role"],
             "records": list(R.values()), "findings": F, "discriminators": discs}
    asof = {**head, "mode": "asof", "records": [_asof(r) for r in R.values() if r["visibility"] == "both"],
            "findings": [f for f in F if f["visibility"] == "both"], "discriminators": discs}
    return _sealed(asof), _sealed(retro)


def _asof(r: dict) -> dict:
    """As-of records drop source paths, which would reveal the incident number; values and file hashes stay."""
    return {**r, "source": [{"file": p["file"], "sha256": p["sha256"]} for p in r["source"]]}


def _sealed(case: dict) -> dict:
    return {**case, "sha256": sha(case)}


def build_all(src: Sources) -> tuple[list[tuple[dict, dict]], dict]:
    cfg = src.load(CASES)
    built = sorted((build(c, src) for c in cfg["cases"]), key=lambda pair: pair[0]["alert_time"])
    manifest = {
        "schema": "syrus.manifest.v1",
        "cases": [{"case_id": a["case_id"], "alert_time": a["alert_time"],
                   "files": {m: {"path": f"{a['case_id']}.{m}.json", "sha256": c["sha256"]} for m, c in (("asof", a), ("retro", r))}}
                  for a, r in built],
        "documents": {"loaded": False, "index_sha256": None},
        "inputs": dict(sorted(src.hashes.items())),
    }
    return built, _sealed(manifest)
