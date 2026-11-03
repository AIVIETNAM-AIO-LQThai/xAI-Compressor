"""The scenario lab: declared, seeded, labelled changes. The recorded background without a change must reproduce the explorer index, a change
must move only what it declares, nothing may be invented, and the public build must not serve any of it. Needs the local MetroPT-3 recording."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.explorer import ROOT, Explorer, build_index
from backend.app.lab import Lab, SensorFault
from backend.syrus_app.settings import load_settings
from backend.web import create_web

RAW = ROOT / "data" / "raw" / "MetroPT3(AirCompressor).csv"
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="local MetroPT-3 recording not present")


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    out = tmp_path_factory.mktemp("lab")
    build_index(out)
    return Lab(Explorer(out))


@pytest.fixture(scope="module")
def bg(lab):
    return lab.backgrounds()[0]


def fault(bg, **kw):
    return SensorFault(**{"start": bg["start"], "hours": 12, "fault": "offset", "channel": "tp3", "onset_hours": 4, "severity": 3, **kw})


def test_backgrounds_are_recorded_quiet_windows_inside_the_test_period(lab):
    bgs = lab.backgrounds()
    assert len(bgs) >= 4
    s = lab.ex.detector()
    for b in bgs:
        a = pd.Timestamp(b["start"])
        w, warm = s.loc[a:a + pd.Timedelta(hours=12)], s.loc[a - pd.Timedelta(hours=4):a]
        assert (warm.split == "test").all() and not warm.alert.any() and not w.threshold_hit.any() and len(w) > 0.95 * 144


def test_the_unchanged_background_reproduces_the_frozen_series(lab, bg):
    r = lab.sensor_fault(fault(bg, severity=0))
    idx = pd.DatetimeIndex(pd.to_datetime(r["t"]))
    ref = lab.ex.detector().reindex(idx)
    for name, key, tol in (("raw_score", "raw_score", 1e-5), ("smoothed_score", "smoothed_score", 1e-3)):
        mine = pd.Series(r["baseline"][name], index=idx)
        ok = mine.notna() & ref[key].notna()
        assert ok.all() and (((mine[ok] - ref.loc[ok, key]).abs() / ref.loc[ok, key].abs().clip(lower=1e-9)) < tol).all(), name
    assert r["perturbed"] == r["baseline"]  # severity 0 changes nothing, so the run with it is the baseline


def test_a_change_moves_nothing_before_its_onset(lab, bg):
    r = lab.sensor_fault(fault(bg, severity=10))
    onset = pd.Timestamp(r["injected"]["onset"])
    t = pd.to_datetime(pd.Series(r["t"]))
    before = (t <= onset).to_numpy()
    for k in ("raw_score", "smoothed_score"):
        a, b = np.array(r["baseline"][k], dtype=float)[before], np.array(r["perturbed"][k], dtype=float)[before]
        assert np.allclose(a, b, equal_nan=True)
    assert r["channel"]["perturbed"][-1] != r["channel"]["baseline"][-1]
    assert r["evidence_class"] == "SIMULATED" and r["detector_identifies_fault"] is False and r["injected"]["onset"] == "2020-03-01 13:00:00"


def test_runs_are_deterministic_and_the_seed_matters_only_where_it_is_used(lab, bg):
    a, b = lab.sensor_fault(fault(bg, fault="noise", seed=7)), lab.sensor_fault(fault(bg, fault="noise", seed=7))
    c, d = lab.sensor_fault(fault(bg, fault="noise", seed=8)), lab.sensor_fault(fault(bg, seed=8))
    assert a == b and a["perturbed"] != c["perturbed"] and d["perturbed"] == lab.sensor_fault(fault(bg, seed=9))["perturbed"]


def test_a_large_shift_alerts_and_a_small_one_may_not_and_the_lab_says_so_either_way(lab, bg):
    big, small = lab.sensor_fault(fault(bg, severity=10)), lab.sensor_fault(fault(bg, severity=0.5, channel="oil_temperature"))
    assert big["first_alert"]["perturbed"] is not None and big["first_alert"]["baseline"] is None
    assert small["first_alert"]["perturbed"] is None and small["detector_identifies_fault"] is False


def test_dropout_removes_bins_and_leaves_gaps_instead_of_filling_them(lab, bg):
    r = lab.sensor_fault(fault(bg, fault="dropout", channel=None, severity=0.6))
    assert r["bins"]["perturbed_valid"] < r["bins"]["baseline_valid"] == r["bins"]["expected"]
    gone = [i for i, v in enumerate(r["perturbed"]["raw_score"]) if v is None]
    assert gone and all(r["perturbed"]["smoothed_score"][i] is None and r["perturbed"]["alert"][i] is None for i in gone)


def test_digital_faults_and_channel_kinds_are_checked(lab, bg):
    assert lab.sensor_fault(fault(bg, fault="stuck", channel="comp", stuck_value=0))["injected"]["stuck_value"] == 0
    with pytest.raises(ValueError):
        fault(bg, fault="stuck", channel="tp3")
    with pytest.raises(ValueError):
        fault(bg, fault="offset", channel="comp")
    with pytest.raises(ValueError):
        fault(bg, onset_hours=12)


def test_windows_outside_the_test_period_are_refused(lab):
    with pytest.raises(ValueError):
        lab.sensor_fault(SensorFault(start="2020-02-10T00:00", hours=12, fault="offset", channel="tp3", onset_hours=4))
    with pytest.raises(ValueError):
        lab.sensor_fault(SensorFault(start="2020-03-01T01:00", hours=12, fault="offset", channel="tp3", onset_hours=4))  # no recorded warm-up before it


def test_the_api_serves_the_lab_locally_and_the_public_build_does_not(lab):
    from backend.app.routes import lab as routes
    routes.lab = lab
    local = TestClient(create_web(load_settings({"SYRUS_KILL_SWITCH": "1"}), dist=lab.ex.root / "none", public=False))
    public = TestClient(create_web(load_settings({"SYRUS_KILL_SWITCH": "1"}), dist=lab.ex.root / "none", public=True))
    start = local.get("/lab/backgrounds").json()["backgrounds"][0]["start"]
    body = {"start": start, "hours": 12, "fault": "offset", "channel": "tp3", "onset_hours": 4, "severity": 2}
    assert local.post("/lab/sensor-fault", json=body).json()["model"] == "recorded_background_sensor_fault"
    assert local.post("/lab/twin", json={}).json()["model"] == "generic_receiver_twin"
    assert local.post("/lab/sensor-fault", json={**body, "hours": 99}).status_code == 422
    assert local.post("/lab/sensor-fault", json={**body, "start": "2020-02-10T00:00"}).json()["detail"]["code"] == "window_invalid"
    for path, method in (("/lab/backgrounds", "get"), ("/lab/sensor-fault", "post"), ("/lab/twin", "post")):
        assert getattr(public, method)(path).status_code in (404, 405)
