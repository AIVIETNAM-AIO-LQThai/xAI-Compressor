import json
import re

import pytest

from backend.syrus_app import cases
from backend.syrus_app.cases import hit_note, pair_limits
from backend.syrus_app.records import (
    RETRO_KINDS,
    ROOT,
    Sources,
    check_case,
    hhmm,
    resolve,
    sha,
)
from scripts.build_syrus_cases import OUT, build_files

FILES = build_files(Sources())
RETRO = {n: c for n, c in FILES.items() if n.endswith(".retro.json")}
ASOF = {n: c for n, c in FILES.items() if n.endswith(".asof.json")}
BY_INCIDENT = {c["source"]["incident_id"]: c for c in RETRO.values()}
FEATURED = next(c for c in RETRO.values() if c["role"] == "featured")
TOKEN = re.compile(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}|(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")


def recs(case):
    return {r["id"]: r for r in case["records"]}


def finding(case, fid):
    return next(f for f in case["findings"] if f["id"] == fid)


def leaves(x):
    if isinstance(x, dict):
        return [v for y in x.values() for v in leaves(y)]
    if isinstance(x, list):
        return [v for y in x for v in leaves(y)]
    return [x]


def tokens(text):
    return set(TOKEN.findall(text))


def resealed(case):
    body = {k: v for k, v in case.items() if k != "sha256"}
    return {**body, "sha256": sha(body)}


def test_committed_files_match_fresh_build():
    for name, case in FILES.items():
        assert json.loads((OUT / name).read_text("utf-8")) == case, f"{name} is stale; run python -m scripts.build_syrus_cases"
    assert sorted(p.name for p in OUT.glob("*.json")) == sorted(FILES)


def test_build_is_deterministic():
    assert build_files(Sources()) == FILES


def test_integrity_checks_reject_tampering_and_leaks():
    for case in [*RETRO.values(), *ASOF.values()]:
        check_case(case)
    tampered = json.loads(json.dumps(FEATURED))
    tampered["records"][0]["value"] = "2020-01-01 00:00:00"
    with pytest.raises(ValueError, match="sha256"):
        check_case(tampered)
    asof = json.loads(json.dumps(ASOF[f"{FEATURED['case_id']}.asof.json"]))
    leak = next(r for r in FEATURED["records"] if r["id"] == "label.timing")
    asof["records"].append({**leak, "source": []})
    with pytest.raises(ValueError, match="retrospective record"):
        check_case(resealed(asof))


def test_integrity_rejects_unreviewed_results_and_ranked_candidates():
    bad = json.loads(json.dumps(FEATURED))
    bad["tests"][0]["result"] = "supports"
    with pytest.raises(ValueError, match="reviewed rule"):
        check_case(resealed(bad))
    bad = json.loads(json.dumps(FEATURED))
    next(r for r in bad["records"] if r["kind"] == "hypothesis")["value"]["status"] = "supported"
    with pytest.raises(ValueError, match="stay open"):
        check_case(resealed(bad))


def test_every_pointer_resolves_to_the_recorded_value():
    src = Sources()
    for case in RETRO.values():
        R = recs(case)
        for r in case["records"]:
            inherited = {json.dumps(p, sort_keys=True) for i in r.get("derivation", {}).get("inputs", []) for p in R[i]["source"]}
            for p in r["source"]:
                src.load(p["file"])
                assert src.hashes[p["file"]] == p["sha256"]
                if "path" not in p:
                    assert "symbol" in p
                    continue
                v = resolve(src.load(p["file"]), p["path"])
                if json.dumps(p, sort_keys=True) not in inherited and not isinstance(v, (dict, list)):
                    assert v in leaves(r["value"]), (case["case_id"], r["id"], p["path"])


def test_code_pointers_name_real_symbols():
    for case in RETRO.values():
        for r in case["records"]:
            for p in r["source"]:
                if "symbol" in p:
                    assert p["symbol"] in (ROOT / p["file"]).read_text("utf-8"), (r["id"], p)


def test_derived_records_recompute():
    for case in RETRO.values():
        R = recs(case)
        thr, s, raw = (R[i]["value"] for i in ("alert.threshold", "alert.smoothed", "alert.raw"))
        a = R["detector.ewma"]["value"]["alpha"]
        prev = R["alert.prev_smoothed"]["value"]["score"]
        assert R["alert.margin"]["value"] == pytest.approx(s / thr - 1)
        assert a * raw + (1 - a) * prev == pytest.approx(s)
        assert R["alert.memory"]["value"] == pytest.approx((1 - a) * prev / s)
        n = R["detector.calibration_bins"]["value"]
        for rid, r in R.items():
            if rid.startswith("calrank."):
                p = R[rid.replace("calrank.", "contrib.")]["value"]["percentile"]
                assert r["value"]["higher"] == round(n * (1 - p))
        shown = sum(r["value"]["share"] for i, r in R.items() if i.startswith("contrib.") and i != "contrib.others")
        assert shown + R["contrib.others"]["value"] == pytest.approx(1.0)
        cfs = [r for i, r in R.items() if i.startswith("cf.") and i != "cf.prewindow"]
        assert len(cfs) == 5
        for c in cfs:
            v = c["value"]
            assert v["original_smoothed_score"] == pytest.approx(s) and v["original_alert"] is True
            assert v["alert_cleared"] == (not v["repaired_alert"])
            assert v["path"] == (("score" if v["repaired_smoothed_score"] < thr else "persistence") if v["alert_cleared"] else None)
            assert v["reduction_ratio"] == pytest.approx((s - v["repaired_smoothed_score"]) / v["own_contribution"])
        pre = R["cf.prewindow"]["value"]
        low = min(c["value"]["repaired_smoothed_score"] for c in cfs)
        assert pre["max_share"] == (0.0 if pre["restart"] else pytest.approx(low / s))


def test_featured_case_is_incident_2_with_frozen_values():
    R = recs(FEATURED)
    assert FEATURED["source"] == {"type": "documented_incident", "incident_id": 2}
    assert R["alert.time"]["value"] == "2020-05-29 21:20:00"
    assert (R["alert.smoothed"]["display"], R["alert.threshold"]["display"], R["alert.margin"]["display"]) == ("58,989", "52,942", "+11.4%")
    assert R["alert.raw"]["display"] == "35.8"
    assert R["alert.prev_smoothed"]["display"] == {"time": "21:15", "score": "73,727"}
    assert R["alert.memory"]["display"] == "99.99%"
    assert R["contrib.dv_pressure"]["display"]["share"] == "77.6%"
    assert R["calrank.dv_pressure"]["display"] == "8 of 1,796"
    assert [i for i in R if i.startswith("contrib.")] == ["contrib.dv_pressure", "contrib.tp2", "contrib.motor_current", "contrib.others"]
    cf = {g: R[f"cf.{g}"]["display"] for g in ("dv_pressure", "tp2", "motor_current")}
    assert {g: d["repaired_smoothed"] for g, d in cf.items()} == {"dv_pressure": "7,472", "tp2": "51,473", "motor_current": "48,374"}
    assert {g: d["ratio"] for g, d in cf.items()} == {"dv_pressure": "1.13×", "tp2": "1.01×", "motor_current": "1.84×"}
    assert cf["dv_pressure"]["window"] == "20:25 to 21:20 (12 bins)"
    assert R["cf.prewindow"]["display"] == "at most 12.7%"
    assert R["measured.dv_pressure__std"]["display"]["value"] == "0.00086"
    assert R["label.timing"]["display"] == "2.17 h before"
    assert R["bench.preonset"]["display"] == "2 of 4"


# incident: (memory fires, clearing repairs, persistence-path repairs, gap, (hits, undetermined, misses), restart)
EXPECTED = {
    1: (False, ["tp2"], [], True, (["00:30", "00:35", "00:40"], [], []), "00:30"),
    2: (True, ["dv_pressure", "tp2", "motor_current"], [], False, (["21:15", "21:20"], ["21:05", "21:10"], ["21:00"]), None),
    3: (False, ["tp2"], [], False, (["10:05", "10:10"], ["09:55", "10:00"], ["09:50"]), None),
    4: (True, ["dv_pressure", "tp2", "motor_current"], ["tp2"], False, (["08:40", "08:45"], ["08:30", "08:35"], ["08:25"]), None),
}
FINDING_IDS = ["F-MARGIN", "F-EWMA-MEMORY", "F-EVIDENCE-TIMING", "F-CF-NONUNIQUE", "F-CF-PERSIST", "F-TRAIN-RANGE", "F-GAP",
               "F-CANDIDATES", "F-IDENT", "F-BRIDGE", "F-LABEL", "F-TIMING", "F-SELECTION", "F-ALERT-BURDEN", "F-DISCRIMINATION"]


@pytest.mark.parametrize("incident", sorted(EXPECTED))
def test_findings_per_case(incident):
    case, R = BY_INCIDENT[incident], recs(BY_INCIDENT[incident])
    memory, clearing, persist, gap, (hits, undetermined, misses), restart = EXPECTED[incident]
    assert [f["id"] for f in case["findings"]] == FINDING_IDS
    assert finding(case, "F-EWMA-MEMORY")["fires"] is memory
    assert memory == (R["alert.raw"]["value"] < R["alert.prev_smoothed"]["value"]["score"])
    assert finding(case, "F-CF-NONUNIQUE")["output"]["clearing"] == clearing
    assert finding(case, "F-CF-NONUNIQUE")["fires"] is (len(clearing) > 1)
    assert finding(case, "F-CF-PERSIST")["output"]["groups"] == persist
    assert finding(case, "F-GAP")["fires"] is gap
    note = R["alert.hits"]["value"]
    assert [[hhmm(b) for b in note[k]] for k in ("hits", "undetermined", "misses")] == [hits, undetermined, misses]
    assert note["hit_counts"] == ([1] if undetermined else [0])
    assert (note["restart"] and hhmm(note["restart"])) == restart
    assert (R["cf.prewindow"]["value"]["restart"] and hhmm(R["cf.prewindow"]["value"]["restart"])) == restart


def test_hit_note_rule():
    ts = [f"2020-01-01 10:{m:02d}:00" for m in range(0, 60, 5)]
    note = hit_note(ts, {ts[-1]: True, ts[-2]: True}, 3, 4, 10, 30)
    assert (note["hits"], note["undetermined"], note["misses"], note["hit_counts"]) == (ts[-2:], ts[-4:-2], [ts[-5]], [1])
    note = hit_note(ts, {ts[-1]: True, ts[-2]: False}, 3, 4, 10, 30)
    assert (note["hits"], note["misses"], note["undetermined"]) == ([ts[-4], ts[-3], ts[-1]], [ts[-5], ts[-2]], [])
    gapped = [*ts[:5], "2020-01-01 10:40:00", "2020-01-01 10:45:00", "2020-01-01 10:50:00"]
    note = hit_note(gapped, {gapped[-1]: True, gapped[-2]: True}, 3, 4, 10, 30)
    assert (note["restart"], note["hits"], note["undetermined"]) == (gapped[5], gapped[5:], [])
    with pytest.raises(ValueError):
        hit_note(gapped, {gapped[-1]: True, gapped[-2]: False}, 3, 4, 10, 30)


def test_finding_numbers_trace_to_input_displays():
    for case in RETRO.values():
        R = recs(case)
        for f in case["findings"]:
            allowed = set().union(*(tokens(json.dumps(R[i]["display"], ensure_ascii=False)) for i in f["inputs"]))
            for text in filter(None, (f["text"], f["detail"])):
                assert tokens(text) <= allowed, (case["case_id"], f["id"], tokens(text) - allowed)


def test_repair_findings_never_equate_repair_with_removing_a_contribution():
    for case in RETRO.values():
        for fid in ("F-CF-NONUNIQUE", "F-CF-PERSIST"):
            f = finding(case, fid)
            for text in filter(None, (f["text"], f["detail"])):
                assert not re.search(r"contribution\W+(?:\w+\W+){0,3}(?:clears|removes) the alert", text), (case["case_id"], fid)
                assert not re.search(r"\b(?:fix|fixed|root cause)\b", text, re.IGNORECASE), (case["case_id"], fid)
        assert "not the machine" in finding(case, "F-CF-NONUNIQUE")["text"]


def test_timing_uses_owner_framing_and_never_prediction_language():
    for case in RETRO.values():
        text = finding(case, "F-TIMING")["text"]
        if case["source"]["incident_id"] in (2, 4):
            assert "whether it was a precursor is not established" in text
        assert not re.search(r"predict|early warning of the leak|caused", text, re.IGNORECASE)


def test_evidence_timing_uses_no_cutoffs():
    for case in RETRO.values():
        text = finding(case, "F-EVIDENCE-TIMING")["text"]
        assert not re.search(r"abrupt|gradual|emerg", text, re.IGNORECASE)
        assert text.endswith("they do not show when a fault began.")


ASOF_FORBIDDEN = ["incident", "onset", "lead_hours", "delay_hours", "air_leak_high_stress", "benchmark", "episode", "global_profiles",
                  "label.", "bench.", "case.status", "documented", "featured", "contrast", "/incidents"]


def test_asof_files_hold_no_retrospective_content():
    for name, case in ASOF.items():
        assert set(case) == {"schema", "case_id", "alert_time", "asset", "mode", "records", "findings", "tests", "pair_limits", "sha256"}
        assert all(r["visibility"] == "both" and r["kind"] not in RETRO_KINDS for r in case["records"])
        assert all(set(p) == {"file", "sha256"} for r in case["records"] for p in r["source"])
        # Source file names are checked by the pointer tests; scan everything else.
        body = {**case, "records": [{k: v for k, v in r.items() if k != "source"} for r in case["records"]]}
        text = json.dumps(body, ensure_ascii=False).lower()
        for word in ASOF_FORBIDDEN:
            assert word not in text, (name, word)


def test_asof_records_equal_their_retro_twins():
    for name, asof in ASOF.items():
        retro = RETRO[name.replace(".asof.", ".retro.")]
        R = recs(retro)
        for x in asof["records"]:
            y = R[x["id"]]
            assert {k: v for k, v in x.items() if k != "source"} == {k: v for k, v in y.items() if k != "source"}
            assert x["source"] == [{"file": p["file"], "sha256": p["sha256"]} for p in y["source"]]
        assert (asof["tests"], asof["pair_limits"]) == (retro["tests"], retro["pair_limits"])


def test_asof_files_share_one_structure():
    prefixes = {tuple(sorted({re.split(r"[.:]", r["id"])[0] for r in c["records"]})) for c in ASOF.values()}
    fields = {tuple(sorted({(r["kind"], tuple(sorted(r))) for r in c["records"]})) for c in ASOF.values()}
    assert len(prefixes) == 1 and len(fields) == 1
    assert len({tuple(f["id"] for f in c["findings"]) for c in ASOF.values()}) == 1


def test_manifest_lists_opaque_ids_by_alert_time():
    m = FILES["manifest.json"]
    times = [c["alert_time"] for c in m["cases"]]
    assert times == sorted(times) and len(times) == 4
    assert all(re.fullmatch(r"c-[a-z0-9]{4}", c["case_id"]) for c in m["cases"])
    assert "incident" not in json.dumps(m)
    assert m["documents"] == {"loaded": False, "index_sha256": None}


def test_candidates_are_open_unranked_and_layered():
    src = Sources()
    order = [h["id"] for h in src.load(cases.HYPS)["hypotheses"]]
    limit_text = src.load(cases.DISCS)["pair_limits"]["not_separable"]
    for case in RETRO.values():
        hyps = [r for r in case["records"] if r["kind"] == "hypothesis"]
        assert [r["value"]["id"] for r in hyps] == order
        assert {r["value"]["status"] for r in hyps} == {"open"}
        assert {r["value"]["layer"] for r in hyps} == {"process", "evidence"}
        apart = {r["value"]["id"]: r["value"]["not_separable_from"] for r in hyps}
        assert apart.pop("air_leakage") == ["downstream_demand"] and apart.pop("downstream_demand") == ["air_leakage"]
        assert not any(apart.values())
        assert case["pair_limits"] == [{"pair": ["air_leakage", "downstream_demand"], "limit": "not_separable",
                                        "tests": ["t_demand_record"], "text": limit_text}]


def test_pair_limit_rule():
    ext, raw = {"availability": "external"}, {"availability": "raw_dataset"}
    tests = [{"id": "x", "compares": [["a", "b"]], "observations": [ext]},
             {"id": "y", "compares": [["a", "c"]], "observations": [raw]},
             {"id": "z", "compares": [["c", "a"]], "observations": [ext]}]
    assert pair_limits(tests) == [{"pair": ["a", "b"], "limit": "not_separable", "tests": ["x"]}]


def test_catalogs_are_bilingual_and_consistent():
    src = Sources()
    hyp_cfg, test_cfg = src.load(cases.HYPS), src.load(cases.DISCS)
    ids = {h["id"] for h in hyp_cfg["hypotheses"]}
    assert ids == {"air_leakage", "downstream_demand", "pressure_regulation", "compressor_performance", "normal_operating_change",
                   "sensor_problem", "data_quality", "reference_coverage"}
    for h in hyp_cfg["hypotheses"]:
        assert set(h["label"]) == set(h["description"]) == {"en", "vi"} and h["layer"] in hyp_cfg["layers"]
        assert "persistent" not in h["label"]["en"].lower()
    for t in test_cfg["tests"]:
        assert (t["maturity"], t["rule"], t["result"]) == ("draft", None, "not_run")
        assert set(t["follow_up"]) == {"en", "vi"}
        assert all(o["availability"] in test_cfg["availability"] and set(o["text"]) == {"en", "vi"} for o in t["observations"])
        assert all(len(p) == 2 and set(p) <= ids for p in t["compares"])
    blocks = [hyp_cfg["layers"], *(test_cfg[k] for k in ("availability", "maturity", "results", "candidate_status", "pair_limits"))]
    for block in blocks:
        assert all(set(v) == {"en", "vi"} and all(isinstance(x, str) for x in v.values()) for v in block.values())


def test_group_records_match_feature_code():
    from ml.data.schema import ANALOG_COLUMNS, DIGITAL_COLUMNS
    from ml.explainability.pca import PRESSURE_RELATIONSHIPS, feature_group

    names = ([f"{s}__{k}" for s in ANALOG_COLUMNS for k in cases.ANALOG_STATS]
             + [f"{s}__{k}" for s in DIGITAL_COLUMNS for k in cases.DIGITAL_STATS]
             + [f"{s}__{k}" for s in cases.RELATIONS for k in cases.RELATION_STATS])
    assert len(names) == 63 and set(cases.RELATIONS) == PRESSURE_RELATIONSHIPS
    for case in RETRO.values():
        for r in case["records"]:
            if r["id"].startswith("group."):
                g = r["value"]["group"]
                assert r["value"]["features"] == [n for n in names if feature_group(n) == g]


def test_syrus_app_stays_slim():
    for p in (ROOT / "backend" / "syrus_app").glob("*.py"):
        text = p.read_text("utf-8")
        assert not re.search(r"^\s*(from|import)\s+(ml|pandas|scipy|torch|numpy|sklearn|backend\.app)\b", text, re.MULTILINE), p.name
