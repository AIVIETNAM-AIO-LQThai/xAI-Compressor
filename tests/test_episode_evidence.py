"""Computed episode evidence: it must reproduce the frozen incident figures with the same explanation code, label itself as computed and
unreviewed, make no causal claim, and refuse anything that is not an alert episode. Needs the local MetroPT-3 recording."""
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.app.episode_evidence import EpisodeEvidence
from backend.app.explorer import ROOT, Explorer, build_index
from backend.syrus_app.settings import load_settings
from backend.web import create_web

RAW = ROOT / "data" / "raw" / "MetroPT3(AirCompressor).csv"
pytestmark = pytest.mark.skipif(not RAW.exists(), reason="local MetroPT-3 recording not present")
FROZEN = json.loads((ROOT / "docs" / "xai_verification_report.json").read_text("utf-8"))["incidents"]


@pytest.fixture(scope="module")
def ee(tmp_path_factory):
    out = tmp_path_factory.mktemp("episode_evidence")
    build_index(out)
    return EpisodeEvidence(Explorer(out))


@pytest.fixture(scope="module")
def api(ee):
    from backend.app.routes import explorer as routes
    routes.evidence = ee
    return TestClient(create_web(load_settings({"SYRUS_KILL_SWITCH": "1"}), dist=ee.ex.root / "none"))


@pytest.mark.parametrize("incident", FROZEN, ids=lambda i: f"incident_{i['id']}")
def test_it_reproduces_the_frozen_ranking_percentiles_and_window(ee, incident):
    got = ee.at(pd.Timestamp(incident["explanation_timestamp"]))
    assert [g["group"] for g in got["groups"]] == [g["group"] for g in incident["ranked_groups"]]
    for a, b in zip(got["groups"], incident["ranked_groups"]):
        assert a["smoothed_contribution"] == pytest.approx(b["smoothed_contribution"], rel=1e-6)
        assert a["share"] == pytest.approx(b["share"], rel=1e-6) and a["calibration_percentile"] == pytest.approx(b["calibration_percentile"], abs=1e-12)
    assert [b["time"] for b in got["window"]["bins"]] == [r["timestamp"] for r in incident["temporal_evidence"]["rows"]]
    assert got["smoothed_score"] == pytest.approx(incident["original_smoothed_score"], rel=1e-9)


def test_the_alert_rule_window_shows_how_three_of_four_was_met(ee):
    e = ee.at(pd.Timestamp("2020-05-29 21:20:00"))
    assert [r["hit"] for r in e["recent"]] == [False, True, True, True] and e["rule"] == {"hits": 3, "window": 4, "alpha": 0.2}
    assert e["carried_share"] == pytest.approx(0.9998785, abs=1e-6)  # the case file's 99.99 %


def test_the_recent_window_starts_again_after_a_smoothing_restart(ee):
    e = ee.at(pd.Timestamp("2020-04-18 00:40:00"))  # the case file: "persistence history restarted at 00:30 after a gap"
    assert [r["time"][11:16] for r in e["recent"]] == ["00:30", "00:35", "00:40"] and e["recent"][0]["restart"]
    for x in ee.ex.episodes():
        flags = [r["restart"] for r in ee.at(pd.Timestamp(x["start"]))["recent"]]
        assert not any(flags[1:])


def test_every_episode_gets_evidence_that_is_labelled_and_makes_no_causal_claim(ee):
    for x in ee.ex.episodes():
        r = ee.episode(pd.Timestamp(x["start"]))
        assert r["status"] == "computed_not_reviewed" and r["causal_claim"] is False
        assert {"counterfactual_repairs", "candidate_explanations", "documents"} <= set(r["not_computed"])
        assert r["evidence"]["groups"][0]["share"] == max(g["share"] for g in r["evidence"]["groups"])
        assert sum(g["share"] for g in r["evidence"]["groups"]) == pytest.approx(1.0)
        assert r["evidence"]["recent"][-1]["hit"]  # an alert bin is a bin at or above the threshold


def test_a_time_that_is_not_an_episode_start_is_refused(ee):
    with pytest.raises(KeyError):
        ee.episode(pd.Timestamp("2020-05-29 21:25:00"))


def test_the_endpoint_serves_it_and_refuses_bad_input(api):
    r = api.get("/explorer/episode", params={"start": "2020-05-29 21:20"})
    assert r.status_code == 200 and r.json()["linked_incident"] == 2 and r.json()["status"] == "computed_not_reviewed"
    assert api.get("/explorer/episode", params={"start": "2020-05-29 21:25"}).json()["detail"]["code"] == "episode_not_found"
    assert api.get("/explorer/episode", params={"start": "tomorrow"}).status_code == 400
