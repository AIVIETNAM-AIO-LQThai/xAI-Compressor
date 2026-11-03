"""The recorded-data explorer index and API: it reproduces the frozen detector, keeps gaps as gaps, never invents samples, gives alert
decisions only where the protocol makes them, and links episodes to incidents by the evaluation's own rule. Needs the local MetroPT-3
recording; skipped without it."""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.explorer import ROOT, Explorer, build_index, sha256
from backend.syrus_app.settings import load_settings
from backend.web import create_web

RAW = ROOT / "data" / "raw" / "MetroPT3(AirCompressor).csv"
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="local MetroPT-3 recording not present")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("explorer")
    return out, build_index(out)


@pytest.fixture(scope="module")
def ex(built):
    return Explorer(built[0])


@pytest.fixture(scope="module")
def api(built):
    from backend.app.routes import explorer as routes
    routes.explorer = Explorer(built[0])
    return TestClient(create_web(load_settings({"SYRUS_KILL_SWITCH": "1"}), dist=built[0] / "none"))


# --- the build ---------------------------------------------------------------------------------------

def test_the_build_reproduces_every_frozen_value_and_records_its_sources(built):
    _, m = built
    assert set(m["reproduced"]) == {"threshold", "incident_2_smoothed", "episodes_total", "episodes_not_linked"}
    assert m["raw_zip_matches_config"] and m["raw_rows"] == 1516948 and m["components"] == 3
    assert m["bins"] == {"train": 5271, "calibration": 1796, "test": 43221}
    assert {"data/raw/metropt3.zip", "data/processed/train.parquet", "configs/detection.yaml", "docs/xai_verification_report.json"} <= set(m["sources"])
    for name, digest in m["outputs"].items():
        assert sha256(built[0] / name) == digest


def test_the_build_is_deterministic(built):
    main = ROOT / "data" / "processed" / "explorer" / "manifest.json"
    if not main.exists():
        pytest.skip("no project index to compare with")
    # Parquet files record the writing library's version, so compare their content; episodes.json is compared byte for byte.
    assert built[1]["outputs"]["episodes.json"] == json.loads(main.read_text("utf-8"))["outputs"]["episodes.json"]
    for name in ("detector.parquet", "raw/2020-05-29.parquet"):
        pd.testing.assert_frame_equal(pd.read_parquet(built[0] / name), pd.read_parquet(main.parent / name), check_exact=True, obj=name)


def test_the_build_refuses_when_a_frozen_value_is_not_reproduced(tmp_path, monkeypatch):
    import backend.app.explorer as module
    real = module.causal_ewma
    monkeypatch.setattr(module, "causal_ewma", lambda s, *, alpha, reset_gap_minutes: real(s, alpha=0.25, reset_gap_minutes=reset_gap_minutes))
    with pytest.raises(RuntimeError, match="does not reproduce"):
        build_index(tmp_path)
    assert not (tmp_path / "manifest.json").exists()


# --- what the series contains -------------------------------------------------------------------------

def test_only_frozen_valid_bins_are_scored_and_nothing_is_added(ex):
    s, f = ex.detector(), ex.features()
    assert set(s.index) <= set(f.index) and len(s) == 50288 and s.index.is_monotonic_increasing
    assert (f.loc[s.index, "coverage_ratio"] >= 0.8).all()


def test_alert_decisions_exist_only_in_the_test_period(ex):
    s = ex.detector()
    assert s.loc[s.split != "test", "alert"].isna().all() and s.loc[s.split == "test", "alert"].notna().all()


def test_smoothing_restarts_after_every_gap_over_ten_minutes(ex):
    s = ex.detector()
    gaps = s.index.to_series().diff() > pd.Timedelta(minutes=10)
    assert (s.loc[gaps[gaps].index, "restart"]).all()
    restarted = s[s.restart]
    assert (restarted["smoothed_score"] == restarted["raw_score"]).all()  # a restart starts from the bin's own score


def test_threshold_hits_follow_the_frozen_threshold(ex):
    s, t = ex.detector(), ex.manifest()["threshold"]
    assert ((s.smoothed_score >= t) == s.threshold_hit).all()


def test_episodes_link_to_incidents_by_the_evaluation_rule(ex):
    eps, incidents = ex.episodes(), {i["id"]: i for i in ex.overview()["incidents"]}
    linked = [e for e in eps if e["linked_incident"] is not None]
    assert len(eps) == 142 and len(linked) == 9
    for e in linked:
        i = incidents[e["linked_incident"]]
        assert pd.Timestamp(e["start"]) <= pd.Timestamp(i["end"]) and pd.Timestamp(e["end"]) >= pd.Timestamp(i["start"]) - pd.Timedelta(hours=24)
    for e in eps:
        assert e["alert_bins"] >= 1 and pd.Timestamp(e["start"]) <= pd.Timestamp(e["peak_time"]) <= pd.Timestamp(e["end"])


# --- the API --------------------------------------------------------------------------------------------

def test_the_overview_states_coverage_and_never_calls_unlinked_episodes_false_alarms(api):
    o = api.get("/explorer/overview").json()
    assert o["bins"]["valid"] == 50301 and o["bins"]["expected"] == 61392 and o["episodes"] == {"total": 142, "linked": 9, "not_linked": 133}
    assert o["source"]["units"] is None and all(i["retrospective"] for i in o["incidents"])
    assert not any(k in json.dumps(o).lower() for k in ("false_alarm", "false alarm", "healthy"))
    assert len(o["activity"]["t"]) == len(o["activity"]["max_smoothed"]) == len(o["activity"]["alert"])


def test_a_range_across_a_gap_breaks_the_line_instead_of_joining_it(api, ex):
    s = ex.detector()
    gap_at = s.index.to_series().diff().idxmax()  # the bin after the longest gap in the recording
    before = s.index[s.index.get_loc(gap_at) - 1]
    j = api.get("/explorer/range", params={"start": str(before - pd.Timedelta(hours=1)), "end": str(gap_at + pd.Timedelta(hours=1))}).json()
    blanks = [n for n, x in enumerate(j["split"]) if x is None]
    assert len(blanks) == 1 and j["smoothed_score"][blanks[0]] is None and j["channels"]["tp2"]["mean"][blanks[0]] is None
    real = [t for t, x in zip(j["t"], j["split"], strict=True) if x is not None]
    assert set(pd.to_datetime(real)) <= set(s.index)


def test_the_range_returns_recorded_aggregates_unchanged(api, ex):
    j = api.get("/explorer/range", params={"start": "2020-05-29 20:00", "end": "2020-05-29 22:00"}).json()
    f = ex.features()
    for t, v in zip(j["t"], j["channels"]["dv_pressure"]["max"], strict=True):
        if v is not None:
            assert v == round(float(f.loc[pd.Timestamp(t), "dv_pressure__max"]), 4)
    assert j["alert"][j["t"].index("2020-05-29 21:20:00")] is True


def test_raw_samples_are_the_recording_itself(api):
    j = api.get("/explorer/raw", params={"start": "2020-05-29 21:00", "end": "2020-05-29 21:20"}).json()
    raw = pd.read_csv(RAW, usecols=["timestamp", "TP2"], parse_dates=["timestamp"])
    raw = raw[(raw.timestamp >= "2020-05-29 21:00") & (raw.timestamp < "2020-05-29 21:20")]
    got = [(t, v) for t, v in zip(j["t"], j["channels"]["tp2"], strict=True) if v is not None]
    assert [pd.Timestamp(t) for t, _ in got] == list(raw.timestamp) and all(abs(v - w) < 1e-3 for (_, v), w in zip(got, raw.TP2, strict=True))


def test_training_and_calibration_bins_carry_no_alert_state(api):
    b = api.get("/explorer/bin", params={"t": "2020-02-10 12:00"}).json()
    assert b["split"] == "train" and b["alert"] is None


@pytest.mark.parametrize(("path", "params", "status"), [
    ("/explorer/range", {"start": "2020-03-01", "end": "2020-04-01"}, 400),
    ("/explorer/range", {"start": "2020-03-02", "end": "2020-03-01"}, 400),
    ("/explorer/raw", {"start": "2020-03-01", "end": "2020-03-03"}, 400),
    ("/explorer/raw", {"start": "yesterday", "end": "today"}, 400),
    ("/explorer/bin", {"t": "2020-05-29 21:21"}, 404),
    ("/explorer/bin", {"t": "../../etc/passwd"}, 400),
])
def test_requests_outside_the_limits_are_refused(api, path, params, status):
    assert api.get(path, params=params).status_code == status


def test_a_missing_index_says_how_to_build_it(tmp_path):
    from backend.app.routes import explorer as routes
    saved, routes.explorer = routes.explorer, Explorer(tmp_path)
    try:
        c = TestClient(create_web(load_settings({"SYRUS_KILL_SWITCH": "1"}), dist=tmp_path / "none"))
        r = c.get("/explorer/overview")
        assert r.status_code == 503 and r.json()["detail"]["code"] == "explorer_index_missing"
    finally:
        routes.explorer = saved


def test_the_public_build_still_serves_the_read_only_explorer(built):
    c = TestClient(create_web(load_settings({"SYRUS_KILL_SWITCH": "1"}), dist=built[0] / "none", public=True))
    assert c.get("/explorer/episodes").status_code in (200, 503) and c.post("/twin/simulate", json={}).status_code in (404, 405)
