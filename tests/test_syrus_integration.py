"""Integration groundwork: shared request pacing, the server-owned diagnosis refusal, sanitised retry verdicts, de-duplicated qualifications,
and the single-origin web app. Offline only: scripted model doubles, no key, no network, no paid call."""
import copy
import json

import pytest
from fastapi.testclient import TestClient

from backend.syrus_app.answer import qualifications
from backend.syrus_app.governor import Governor
from backend.syrus_app.settings import load_settings
from backend.web import create_web
from tests.test_syrus_context import (
    CASE,
    KEY,
    PACK,
    STORE,
    U1,
    U2,
    Fake,
    ask,
    client,
    reply,
    start,
    unit,
)
from tests.test_syrus_label import EN, IDENT_EN, IDENT_VI, LABEL_EN, LABEL_VI, VI

# --- governor ---------------------------------------------------------------------------------

class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_the_governor_admits_one_call_at_a_time_and_names_a_retry_time():
    g = Governor(8000, 0.9, 1, Clock())
    ticket, wait = g.admit(3000)
    assert ticket and wait == 0
    assert g.admit(100) == (None, 3.0)  # one is in flight
    g.release(ticket, 2000)
    assert g.admit(100)[0]


def test_the_governor_keeps_the_minute_window_and_reports_when_it_frees():
    clock = Clock()
    g = Governor(8000, 0.9, 1, clock)  # 7,200 tokens per minute
    first = g.admit(4000)[0]
    g.release(first, 4000)
    clock.t = 20
    second = g.admit(3000)[0]
    g.release(second, 3000)
    clock.t = 30
    ticket, wait = g.admit(1000)  # 4,000 + 3,000 + 1,000 > 7,200: the first call must age out
    assert ticket is None and wait == 30.0
    clock.t = 61
    assert g.admit(1000)[0]


def test_a_call_larger_than_the_whole_window_is_never_admitted_and_actual_use_replaces_the_estimate():
    clock = Clock()
    g = Governor(1000, 0.9, 1, clock)
    assert g.admit(2000) == (None, 60.0)
    t = g.admit(800)[0]
    g.release(t, 100)
    assert g.admit(700)[0]  # the 800 reserved was settled to 100


def test_a_visitor_who_cannot_be_admitted_gets_busy_and_nothing_is_sent_or_spent(tmp_path):
    fake = Fake(reply(U1, U2))
    c = client(tmp_path, fake, patch=lambda cfg: cfg.update(governor={"concurrent_calls": 1, "window_share": 0.9}), SYRUS_DIAGNOSTICS="1")
    svc = c.app.state.svc
    held = svc.governor.admit(1000)[0]  # another visitor's call is in flight
    r = ask(c, start(c), "Why did this alert fire?")
    body = r.json()
    assert r.status_code == 503 and body["reason"] == "busy" and body["retry_after"] == 3.0 and not fake.calls
    assert svc.budget.summary()["total_usd"] == 0 and svc.budget.pending == 0 and body["diagnostics"]["calls"][0]["error"] == "busy"
    svc.governor.release(held)
    assert ask(c, start(c), "Why did this alert fire?", depth="quick").json()["status"] == "ok"


# --- diagnosis refusal ----------------------------------------------------------------------------

@pytest.mark.parametrize(("message", "lang", "units"), [(EN, "en", (LABEL_EN, IDENT_EN)), (VI, "vi", (LABEL_VI, IDENT_VI))])
def test_a_leak_certainty_answer_carries_the_diagnosis_request_label_whatever_the_model_declared(tmp_path, message, lang, units):
    for declared, by in ((None, "server"), ("diagnosis_request", "model")):
        c = client(tmp_path, Fake(reply(*units, purposes=["evidence_and_sources"], refusal=declared, lang=lang)))
        body = ask(c, start(c), message, depth="investigation").json()
        assert body["status"] == "ok" and body["refusal"] == "diagnosis_request" and body["refusal_by"] == by
        assert body["provenance"]["validator"] == "passed"


def test_a_server_set_refusal_adds_no_qualification_the_claim_did_not_trigger(tmp_path):
    c = client(tmp_path, Fake(reply(LABEL_EN, IDENT_EN, purposes=["evidence_and_sources"], refusal=None)))
    blocks = ask(c, start(c), EN, depth="investigation").json()["blocks"]
    assert "score_not_diagnosis" not in {q["id"] for b in blocks for q in b["qualifications"]}


def test_ordinary_questions_are_not_labelled_as_refusals(tmp_path):
    c = client(tmp_path, Fake(reply(U1, U2)))
    body = ask(c, start(c), "Why did this alert fire?", depth="quick").json()
    assert body["refusal"] is None and body["refusal_by"] is None


def test_a_refusal_the_model_chose_is_kept_over_the_servers(tmp_path):
    c = client(tmp_path, Fake(reply(LABEL_EN, IDENT_EN, purposes=["evidence_and_sources"], refusal="causal_claim")))
    body = ask(c, start(c), EN, depth="investigation").json()
    assert body["refusal"] == "causal_claim" and body["refusal_by"] == "model"


# --- retry verdicts -------------------------------------------------------------------------------------

def test_a_rejected_first_reply_leaves_check_codes_and_finding_ids_never_text(tmp_path):
    bad = reply(unit("The cause of the alert was a leak PRIVATE-WORDS.", ["alert.smoothed"], []), purposes=["what_happened"])
    fake = Fake(bad, reply(U1, U2))
    c = client(tmp_path, fake, SYRUS_DIAGNOSTICS="1")
    body = ask(c, start(c), "Why did this alert fire?", depth="quick").json()
    first, second = body["diagnostics"]["calls"]
    assert first["verdict"] == "rejected" and "forbidden_phrase" in first["codes"] and second["verdict"] == "accepted" and second["added_findings"] == []
    assert "PRIVATE" not in json.dumps(body["diagnostics"]) and "leak" not in json.dumps(first["codes"])


def test_a_missing_required_finding_is_named_by_id_in_the_verdict(tmp_path):
    fake = Fake(reply(U1))
    c = client(tmp_path, fake, SYRUS_DIAGNOSTICS="1")
    body = ask(c, start(c), "Why did this alert fire?", depth="quick").json()
    first, second = body["diagnostics"]["calls"]
    assert first["verdict"] == "rejected" and first["codes"] == ["required_finding"] and first["missing_findings"] == ["F-EWMA-MEMORY"]
    assert second["verdict"] == "accepted" and second["added_findings"] == ["F-EWMA-MEMORY"]


def test_an_unparseable_reply_is_a_schema_verdict(tmp_path):
    c = client(tmp_path, Fake("not json at all PRIVATE", reply(U1, U2)), SYRUS_DIAGNOSTICS="1")
    first = ask(c, start(c), "Why did this alert fire?", depth="quick").json()["diagnostics"]["calls"][0]
    assert first["verdict"] == "rejected" and first["codes"] == ["schema"] and "PRIVATE" not in json.dumps(first)


# --- qualifications and wording ----------------------------------------------------------------------------

def quals(text, direct, lang="en"):
    return [q["id"] for q in qualifications(set(direct), text, [], set(), set(), lang)]


def test_a_unit_that_already_states_the_limit_does_not_get_the_same_limit_appended():
    said = "Leakage and higher downstream demand cannot be told apart with this data."
    assert "leak_demand" not in quals(said, ["hyp.air_leakage", "F-IDENT"])
    assert "leak_demand" in quals("Air leakage is one open candidate.", ["hyp.air_leakage"])
    assert "leak_demand" not in quals("Rò rỉ và nhu cầu khí tăng không thể phân biệt được.", ["F-IDENT"], "vi")


def test_the_persistence_rule_is_not_repeated_beside_a_unit_that_states_it():
    assert "persistence_path" not in quals("An alert needs 3 of the last 4 smoothed scores at or above the threshold.", ["detector.persistence"])
    assert "persistence_path" in quals("The detector applies its persistence rule.", ["detector.persistence"])
    assert "persistence_path" in quals("The rule needs 13 of 14 scores.", ["detector.persistence"])


def test_a_later_unit_does_not_repeat_a_limit_an_earlier_unit_already_covered():
    shown = set()
    first = qualifications({"F-IDENT"}, "Leakage and demand cannot be told apart.", [], set(), shown, "en")
    second = qualifications({"hyp.air_leakage"}, "Air leakage is open.", [], set(), shown, "en")
    assert "leak_demand" not in [q["id"] for q in first + second]


def test_the_smoothing_memory_finding_no_longer_reads_as_persistence():
    text = PACK.findings["F-EWMA-MEMORY"]["text"]
    assert "accounts for 99.99% of the current smoothed score" in text and "came from earlier bins" not in text and "persist" not in text


# --- the single origin -----------------------------------------------------------------------------------------

def web(tmp_path, adapter=None, public=False, dist=None):
    s = load_settings({"SYRUS_LLM_API_KEY": KEY, "SYRUS_CAPS_CONFIRMED": "1", "SYRUS_SESSION_SECRET": "x", "SYRUS_BUDGET_FILE": str(tmp_path / "b.json")})
    cfg = copy.deepcopy(s.cfg)
    cfg["governor"] = {"concurrent_calls": 1, "window_share": 1000}
    s = type(s)(cfg, s.key, s.caps_confirmed, s.kill, s.secret, s.ip_salt, s.budget_file)
    return TestClient(create_web(s, adapter or Fake(reply(U1, U2)), STORE, dist=dist or tmp_path / "none", public=public))


def test_one_origin_serves_the_syrus_api_the_evidence_api_and_the_built_frontend(tmp_path):
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "index.html").write_text("<title>AeroXAI</title>", "utf-8")
    fake = Fake(reply(U1, U2))
    with web(tmp_path, fake, dist=tmp_path / "dist") as c:
        assert c.get("/").text == "<title>AeroXAI</title>"
        assert c.get("/health").json()["project"] == "AeroXAI" and c.get("/api/health").json()["status"] == "ok"
        assert c.get("/evidence/summary").status_code == 200 and len(c.get("/api/cases").json()["cases"]) == 4
        token = c.post("/api/session", json={"case_id": CASE, "mode": "retro"}).json()["token"]
        body = c.post("/api/syrus/turn", json={"message": "Why did this alert fire?", "depth": "quick"}, headers={"Authorization": f"Bearer {token}"}).json()
        assert body["status"] == "ok" and len(fake.calls) == 1
        assert c.post("/api/syrus/turn", json={"message": "x"}).json() == {"error": {"code": "token_missing"}}  # the Syrus error shape survives


def test_the_origin_has_one_syrus_service_one_budget_and_no_docs_routes(tmp_path):
    with web(tmp_path) as c:
        assert c.app.state.svc.budget is not None and c.app.state.svc.adapter is not None
        assert [c.get(p).status_code for p in ("/docs", "/openapi.json", "/redoc")] == [404, 404, 404]
        paths = {getattr(r, "path", "") for r in c.app.router.routes}
        assert {"/api/session", "/api/syrus/turn", "/api/health", "/health"} <= paths


def test_the_public_build_serves_no_simulation_route(tmp_path):
    with web(tmp_path, public=False) as local, web(tmp_path, public=True) as public:
        assert local.post("/twin/simulate", json={}).status_code == 200 and local.post("/optimization/recommend", json={}).status_code in (200, 422)
        assert public.post("/twin/simulate", json={}).status_code in (404, 405) and public.post("/optimization/recommend", json={}).status_code in (404, 405)
        assert public.get("/evidence/summary").status_code == 200 and public.get("/api/cases").status_code == 200


def test_the_origin_exposes_no_equipment_route_and_never_the_key(tmp_path):
    with web(tmp_path) as c:
        text = " ".join(getattr(r, "path", "") for r in c.app.router.routes).lower()
        assert not any(w in text for w in ("control", "write", "plc", "actuat", "setpoint"))
        assert KEY not in c.get("/api/health").text and KEY not in c.get("/health").text


def test_a_limit_stated_in_one_unit_is_not_appended_beside_another_unit_that_only_touches_the_topic(tmp_path):
    from tests.test_syrus_label import IDENT_EN, LABEL_EN
    c = client(tmp_path, Fake(reply(LABEL_EN, IDENT_EN, purposes=["evidence_and_sources"], refusal=None)))
    body = ask(c, start(c), EN, depth="investigation").json()
    assert body["status"] == "ok" and "leak_demand" not in {q["id"] for b in body["blocks"] for q in b["qualifications"]}
    assert body["answer"].count("cannot be told apart") == 1


def test_a_refusal_from_the_provider_frees_the_governor_but_a_dropped_connection_keeps_its_reservation(tmp_path):
    from backend.syrus_app.llm import LlmError
    from tests.test_syrus_guards import Failing
    cfg = lambda c: c.update(governor={"concurrent_calls": 1, "window_share": 0.9})
    refused = client(tmp_path, Failing(LlmError("rate_limited", 5.0, indeterminate=False)), patch=cfg)
    ask(refused, start(refused), "Why did this alert fire?")
    assert sum(t for _, t in refused.app.state.svc.governor.window) == 0 and refused.app.state.svc.governor.inflight == 0
    dropped = client(tmp_path, Failing(LlmError("provider_error", indeterminate=True)), patch=cfg)
    ask(dropped, start(dropped), "Why did this alert fire?")
    assert sum(t for _, t in dropped.app.state.svc.governor.window) > 0 and dropped.app.state.svc.governor.inflight == 0


def test_the_governor_learns_how_pessimistic_the_estimate_is_but_never_trusts_less_than_half():
    g = Governor(8000, 0.9, 1, Clock())
    g.observe(3000, 1800)
    assert g.ratio == pytest.approx(0.8)
    g.observe(3000, 1800)
    assert g.ratio == pytest.approx(0.7)
    for _ in range(20):
        g.observe(3000, 100)
    assert g.ratio == 0.5
    g.observe(3000, 9000)
    assert g.ratio == 1.0  # a call larger than estimated is trusted at once, up to the full estimate


def test_a_second_question_in_the_same_minute_is_admitted_once_the_real_usage_is_known(tmp_path):
    fake = Fake(reply(U1, U2), tokens=(1900, 300))
    c = client(tmp_path, fake, patch=lambda cfg: cfg.update(governor={"concurrent_calls": 1, "window_share": 0.9}))
    h = start(c)
    assert ask(c, h, "Why did this alert fire?", depth="quick").json()["status"] == "ok"
    second = ask(c, h, "Which signals contributed to the score?", depth="quick")
    assert second.status_code == 200 and second.json()["status"] in ("ok", "failed_checks")
