import json
import re

import pytest

from backend.syrus_app import cases
from backend.syrus_app.cases import statuses
from backend.syrus_app.records import (
    RETRO_KINDS,
    ROOT,
    Sources,
    check_case,
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
    asof = {k: v for k, v in asof.items() if k != "sha256"}
    with pytest.raises(ValueError, match="retrospective record"):
        check_case({**asof, "sha256": sha(asof)})


def test_every_pointer_resolves_to_the_recorded_value():
    src = Sources()
    for case in RETRO.values():
        for r in case["records"]:
            for p in r["source"]:
                src.load(p["file"])
                assert src.hashes[p["file"]] == p["sha256"]
                if "path" not in p:
                    assert "symbol" in p
                    continue
                v = resolve(src.load(p["file"]), p["path"])
                if "derivation" not in r and not isinstance(v, (dict, list)):
                    assert v in leaves(r["value"]), (case["case_id"], r["id"], p["path"])


def test_derived_records_recompute():
    for case in RETRO.values():
        R = recs(case)
        thr, s, raw = (R[i]["value"] for i in ("alert.threshold", "alert.smoothed", "alert.raw"))
        assert R["alert.margin"]["value"] == pytest.approx(s / thr - 1)
        assert R["alert.raw_ratio"]["value"] == pytest.approx(raw / thr)
        a = R["detector.ewma"]["value"]["alpha"]
        assert a * raw + (1 - a) * R["alert.prev_smoothed"]["value"]["score"] == pytest.approx(s)
        n = R["detector.calibration_bins"]["value"]
        for rid, r in R.items():
            if rid.startswith("calrank."):
                p = R[rid.replace("calrank.", "contrib.")]["value"]["percentile"]
                assert r["value"]["higher"] == round(n * (1 - p))
        shown = sum(r["value"]["share"] for i, r in R.items() if i.startswith("contrib.") and i != "contrib.others")
        assert shown + R["contrib.others"]["value"] == pytest.approx(1.0)


def test_featured_case_is_incident_2_with_frozen_values():
    R = recs(FEATURED)
    assert FEATURED["source"] == {"type": "documented_incident", "incident_id": 2}
    assert R["alert.time"]["value"] == "2020-05-29 21:20:00"
    assert (R["alert.smoothed"]["display"], R["alert.threshold"]["display"], R["alert.margin"]["display"]) == ("58,989", "52,942", "+11.4%")
    assert R["alert.raw"]["display"] == "35.8"
    assert R["alert.prev_smoothed"]["display"] == {"time": "21:15", "score": "73,727"}
    assert R["contrib.dv_pressure"]["display"]["share"] == "77.6%"
    assert R["calrank.dv_pressure"]["display"] == "8 of 1,796"
    assert [i for i in R if i.startswith("contrib.")] == ["contrib.dv_pressure", "contrib.tp2", "contrib.motor_current", "contrib.others"]
    assert {g: R[f"cf.{g}"]["display"]["repaired_smoothed"] for g in ("dv_pressure", "tp2", "motor_current")} == {
        "dv_pressure": "7,472", "tp2": "51,473", "motor_current": "48,374"}
    assert R["measured.dv_pressure__std"]["display"]["value"] == "0.00086"
    assert R["label.timing"]["display"] == "2.17 h before"
    assert R["bench.preonset"]["display"] == "2 of 4"


# incident: (marginal, carryover fires, emergence pattern, clearing repairs, persistence-only clear, gap)
EXPECTED = {
    1: (False, False, "abrupt", ["tp2"], [], True),
    2: (True, True, "abrupt", ["dv_pressure", "tp2", "motor_current"], [], False),
    3: (False, False, "gradual", ["tp2"], [], False),
    4: (True, True, "elevated_at_start", ["dv_pressure", "tp2", "motor_current"], ["tp2"], False),
}


@pytest.mark.parametrize("incident", sorted(EXPECTED))
def test_findings_per_case(incident):
    case = BY_INCIDENT[incident]
    marginal, carry, pattern, clearing, persist, gap = EXPECTED[incident]
    assert finding(case, "F-MARGIN")["output"]["marginal"] is marginal
    assert finding(case, "F-CARRYOVER")["fires"] is carry
    assert finding(case, "F-EMERGE")["output"]["pattern"] == pattern
    assert finding(case, "F-CF-SPEC")["output"]["clearing"] == clearing
    assert finding(case, "F-CF-PERSIST")["output"]["groups"] == persist
    assert finding(case, "F-GAP")["fires"] is gap
    assert [f["id"] for f in case["findings"]] == ["F-MARGIN", "F-CARRYOVER", "F-EMERGE", "F-CF-SPEC", "F-CF-PERSIST", "F-HYP-TIE",
                                                   "F-SCALE", "F-GAP", "F-IDENT", "F-BRIDGE", "F-LABEL", "F-TIMING", "F-STATUS", "F-BENCH"]


def test_finding_numbers_trace_to_input_displays():
    for case in RETRO.values():
        R = recs(case)
        for f in case["findings"]:
            allowed = set().union(*(tokens(json.dumps(R[i]["display"], ensure_ascii=False)) for i in f["inputs"]))
            assert tokens(f["text"]) <= allowed, (case["case_id"], f["id"], tokens(f["text"]) - allowed)


def test_timing_uses_owner_framing_and_never_prediction_language():
    for case in RETRO.values():
        text = finding(case, "F-TIMING")["text"]
        if case["source"]["incident_id"] in (2, 4):
            assert "whether it was a precursor is not established" in text
        assert not re.search(r"predict|early warning of the leak|caused", text, re.IGNORECASE)


ASOF_FORBIDDEN = ["incident", "onset", "lead_hours", "delay_hours", "air_leak_high_stress", "benchmark", "episode", "global_profiles",
                  "label.", "bench.", "case.status", "documented", "featured", "contrast", "/incidents"]


def test_asof_files_hold_no_retrospective_content():
    for name, case in ASOF.items():
        assert set(case) == {"schema", "case_id", "alert_time", "asset", "mode", "records", "findings", "discriminators", "sha256"}
        assert all(r["visibility"] == "both" and r["kind"] not in RETRO_KINDS for r in case["records"])
        assert all(set(p) == {"file", "sha256"} for r in case["records"] for p in r["source"])
        # Source file names are checked by the pointer tests; scan everything else.
        body = {**case, "records": [{k: v for k, v in r.items() if k != "source"} for r in case["records"]]}
        text = json.dumps(body, ensure_ascii=False).lower()
        for word in ASOF_FORBIDDEN:
            assert word not in text, (name, word)


def test_asof_records_equal_their_retro_twins():
    for name, asof in ASOF.items():
        R = recs(RETRO[name.replace(".asof.", ".retro.")])
        for x in asof["records"]:
            y = R[x["id"]]
            assert {k: v for k, v in x.items() if k != "source"} == {k: v for k, v in y.items() if k != "source"}
            assert x["source"] == [{"file": p["file"], "sha256": p["sha256"]} for p in y["source"]]


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


def test_hypotheses_are_unranked_with_table_statuses():
    for case in RETRO.values():
        hyps = [r for r in case["records"] if r["kind"] == "hypothesis"]
        assert len(hyps) == 7
        for r in hyps:
            assert set(r["value"]) == {"id", "status", "against", "matched_groups"}
        status = {r["value"]["id"]: r["value"]["status"] for r in hyps}
        assert status["air_leakage"] == status["downstream_demand"] == "untestable_here"
        assert {s for h, s in status.items() if h not in ("air_leakage", "downstream_demand")} == {"not_discriminated"}


def test_status_rule():
    discs = [{"separates": [["a", "b"]], "availability": "absent"},
             {"separates": [["a", "c"]], "availability": "raw_data_only"},
             {"separates": [["a", "c"]], "availability": "absent"}]
    assert statuses(["a", "b", "c"], discs) == {
        "a": ("untestable_here", ["b"]), "b": ("untestable_here", ["a"]), "c": ("not_discriminated", [])}


def test_catalogs_are_bilingual_and_consistent():
    src = Sources()
    hyps = src.load(cases.HYPS)["hypotheses"]
    disc = src.load(cases.DISCS)
    ids = {h["id"] for h in hyps}
    assert {"detector_artifact", "operating_transient", "downstream_demand"} <= ids
    for h in hyps:
        assert set(h["label"]) == set(h["description"]) == {"en", "vi"}
        assert "persistent" not in h["label"]["en"].lower()
    for d in disc["discriminators"]:
        assert d["availability"] in disc["availability"]
        assert set(d["question"]) == set(d["observation"]) == {"en", "vi"}
        assert all(len(p) == 2 and set(p) <= ids for p in d["separates"])
    for block in (disc["availability"], disc["statuses"]):
        assert all(set(v) == {"en", "vi"} and all(isinstance(t, str) for t in v.values()) for v in block.values())


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
