"""Question-relevant context: token measurement, deterministic selection, the schema built from the selected IDs, fail-closed rules,
and sanitised provider diagnostics. Offline only: scripted model doubles and mock HTTP transports, no key and no paid call."""
import copy
import dataclasses
import json
import logging
import subprocess
import sys
from collections import deque

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from backend.syrus_app.app import create_app
from backend.syrus_app.context import CaseStore
from backend.syrus_app.llm import Completion, GroqAdapter, LlmError
from backend.syrus_app.policy import FOCUS, LEAN, PURPOSES, REFUSALS, depth_after
from backend.syrus_app.prompt import baseline, build
from backend.syrus_app.records import ROOT
from backend.syrus_app.selection import (
    SelectionError,
    build_view,
    classify,
    plan,
    views,
)
from backend.syrus_app.sessions import Thread
from backend.syrus_app.settings import load_settings
from backend.syrus_app.tokens import estimate, estimate_request
from backend.syrus_app.validator import check, parse
from scripts.syrus_tokens import items, rows

STORE = CaseStore()
CASES = [c["case_id"] for c in STORE.manifest["cases"]]
CASE = next(c["case_id"] for c in STORE.manifest["cases"] if c["alert_time"].startswith("2020-05-29"))
PACK = STORE.pack(CASE)
SETTINGS = load_settings({})
CAP = SETTINGS.input_cap
KEY = "gsk-test-key-123"


def thread(case=CASE, focus=(), exchanges=()):
    t = Thread("sid", case, "retro", 1e12, exchanges=deque(exchanges, maxlen=2), focus=list(focus))
    t.turns = len(exchanges)
    return t


def view(message, depth="auto", pack=PACK, previous=()):
    return next(iter(views(pack, message, previous, depth)))


def unit(text, rec=(), fnd=(), kind="model_evidence"):
    return {"text": text, "kind": kind, "record_ids": list(rec), "finding_ids": list(fnd)}


U1 = unit("The smoothed score was 58,989 against a threshold of 52,942 (+11.4%).", ["alert.smoothed", "alert.threshold", "alert.margin"], ["F-MARGIN"])
U2 = unit("At 21:20 the bin's own raw score was 35.8, below the 73,727 carried from 21:15, so 99.99% of the smoothed score came from earlier bins.",
          ["alert.time", "alert.raw", "alert.prev_smoothed", "alert.memory"], ["F-EWMA-MEMORY"])


def reply(*units, **over):
    return {"lang": "en", "purposes": ["what_happened"], "units": list(units), "alternatives": [], "gaps": [], "next_test": None, "refusal": None,
            "operator_statement": False, **over}


class Fake:
    def __init__(self, *outputs, tokens=(3000, 300)):
        self.outputs, self.tokens, self.calls = list(outputs), tokens, []

    async def complete(self, messages, schema, max_tokens, timeout):
        self.calls.append({"messages": copy.deepcopy(messages), "schema": schema, "max_tokens": max_tokens})
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        return Completion(out if isinstance(out, str) else json.dumps(out), *self.tokens, {"remaining_tokens": "4000"})


def client(tmp_path, adapter, patch=None, **envover):
    env = {"SYRUS_LLM_API_KEY": KEY, "SYRUS_CAPS_CONFIRMED": "1", "SYRUS_SESSION_SECRET": "secret", "SYRUS_IP_SALT": "salt",
           "SYRUS_BUDGET_FILE": str(tmp_path / "budget.json"), **envover}
    s = load_settings(env)
    cfg = copy.deepcopy(s.cfg)
    cfg["governor"] = {"concurrent_calls": 1, "window_share": 1000}  # pacing has its own tests; these scripted token counts are not realistic
    if patch:
        patch(cfg)
    return TestClient(create_app(dataclasses.replace(s, cfg=cfg), adapter, STORE))


def start(c, case=CASE):
    return {"Authorization": "Bearer " + c.post("/api/session", json={"case_id": case, "mode": "retro"}).json()["token"]}


def ask(c, h, message, **kw):
    return c.post("/api/syrus/turn", json={"message": message, **kw}, headers=h)


# --- measurement ------------------------------------------------------------

def test_the_estimate_is_pessimistic_for_vietnamese_and_counts_the_schema():
    assert estimate("a" * 300) < estimate("ế" * 300)
    msgs = [{"role": "user", "content": "x"}]
    assert estimate_request(msgs, PACK.schema) > estimate_request(msgs, {"type": "object"}) + 500


def test_the_whole_ledger_baseline_reproduces_the_measured_overrun():
    """The live 429 reported an estimated input of 8,162 tokens for the first acceptance question."""
    b = baseline(PACK, thread(), "Why did this alert fire?", "quick")
    assert 8100 <= b.est <= 8250 and b.est + 900 > 8000


def test_parts_are_measured_separately_and_add_up():
    b = build(PACK, thread(), "Why did this alert fire?", "quick", CAP, 2)
    assert {"system", "scope", "records", "findings", "tests", "history", "request", "schema", "overhead"} == set(b.parts)
    assert sum(b.parts.values()) == b.est and b.parts["tests"] == 1 and b.parts["history"] == 0
    full = baseline(PACK, thread(), "Why did this alert fire?", "quick")
    assert full.parts["tests"] > 100 and full.parts["records"] > b.parts["records"] and full.parts["schema"] > b.parts["schema"]


# --- every acceptance question, on every case ---------------------------------

SWEEP = rows(STORE, SETTINGS, cases=CASES)


def test_the_sweep_covers_every_acceptance_question_on_every_case():
    assert len(SWEEP) == sum(len(i["turns"]) for i in items()) * len(CASES)
    assert {r["item"] for r in SWEEP} == {i["id"] for i in items()}


@pytest.mark.parametrize("r", SWEEP, ids=lambda r: f"{r['item']}-{r['turn']}-{r['case']}")
def test_each_acceptance_question_fits_keeps_its_evidence_and_is_much_smaller(r):
    assert r["after"] is not None, r.get("error")
    assert r["after"] <= CAP and r["after"] + SETTINGS.limits["max_output_tokens"] <= 8000 * 0.75
    assert r["after"] <= 0.67 * r["before"]
    assert not r["essentials_missing"] and not r["required_missing"]


@pytest.mark.parametrize("case", CASES)
def test_a_view_is_a_closed_subset_of_the_ledger(case):
    pack = STORE.pack(case)
    for item in items():
        for turn in item["turns"]:
            v = view(turn["text"], item["depth"], pack)
            assert all(pack.records[i] is r for i, r in v.records.items()) and all(pack.findings[i] is f for i, f in v.findings.items())
            for f in v.findings.values():
                needed = LEAN[f["id"]] if f["id"] in v.lean else f["inputs"]
                assert {i for i in needed if not i.startswith("group.")} <= set(v.records), f["id"]
                assert set(LEAN.get(f["id"], [])) <= set(f["inputs"])
            assert "F-GAP" in v.findings and {i for i in pack.records if i.startswith("unavailable:")} <= set(v.records)
            required = {f for p in v.purposes for f, d in PURPOSES[p]["require"] if depth_after(v.depth, d) and pack.findings[f]["fires"]}
            assert required <= set(v.findings)


def test_the_schema_offers_exactly_the_selected_ids():
    v = view("Why did this alert fire?", "quick")
    s, unit_props = v.schema["properties"], v.schema["properties"]["units"]["items"]["properties"]
    assert set(unit_props["record_ids"]["items"]["enum"]) == set(v.records) < set(PACK.records)
    assert set(unit_props["finding_ids"]["items"]["enum"]) == set(v.findings) < set(PACK.findings)
    assert set(s["purposes"]["items"]["enum"]) == set(v.purposes) and "possible_causes" not in v.purposes
    assert s["alternatives"]["items"]["enum"] == ["none"] and s["next_test"]["enum"] == [None]
    assert "cf.tp2" not in unit_props["record_ids"]["items"]["enum"] and "t_demand_record" not in json.dumps(v.schema)


# --- what each kind of question gets -------------------------------------------

CATEGORIES = [
    ("Why did this alert fire?", ["alert"]),
    ("Vì sao cảnh báo này xuất hiện?", ["alert"]),
    ("And what about the jump at 21:10?", ["timing"]),
    ("Bằng chứng tăng lúc nào trong cửa sổ?", ["timing"]),
    ("Was this definitely an air leak?", ["certainty"]),
    ("Đây có chắc chắn là rò rỉ khí không?", ["certainty"]),
    ("Did removing the motor_current contribution fix the compressor?", ["repair", "signals"]),
    ("Nếu thay thế tín hiệu tp2 thì sao?", ["repair", "signals"]),
    ("Did the model predict it 2 hours early?", ["lead"]),
    ("How much energy did this waste?", ["energy"]),
    ("Năng lượng bị lãng phí bao nhiêu?", ["energy"]),
    ("How many false alarms does the detector raise?", ["benchmark"]),
    ("Show the raw sensor trace.", ["sources"]),
    ("Cho tôi xem nhật ký bảo trì của máy này.", ["sources"]),
    ("Where does 73,727 come from?", ["method"]),
    ("Was dv_pressure_std inside its training range?", ["signals", "range"]),
]


@pytest.mark.parametrize(("message", "expected"), CATEGORIES)
def test_questions_are_classified_into_the_topics_they_ask_about(message, expected):
    assert classify(message) == expected


def test_a_simple_alert_explanation_leaves_out_candidates_repairs_and_tests():
    v = view("Why did this alert fire?", "quick")
    kinds = {r["kind"] for r in v.records.values()}
    assert "hypothesis" not in kinds and not v.tests and not v.pairs
    assert not any(i.startswith(("cf.", "temporal.2", "hyp.", "bench.", "label.")) for i in v.records) and "temporal.window" in v.records  # F-GAP depends on it
    assert {"F-MARGIN", "F-EWMA-MEMORY", "F-GAP"} == set(v.findings)


def test_a_candidate_question_gets_candidates_tests_and_the_pair_limit():
    v = view("What are the candidate causes?", "investigation")
    assert {"F-CANDIDATES", "F-IDENT", "F-CF-NONUNIQUE", "F-LABEL", "F-TIMING"} <= set(v.findings)
    assert len([i for i in v.records if i.startswith("hyp.")]) == 8 and len(v.tests) == 7 and v.pairs
    assert "leakage" in v.text.lower() and "air_leakage / downstream_demand" in v.sections["tests"]
    assert set(v.schema["properties"]["next_test"]["enum"]) == {*v.tests, None}


def test_a_finding_present_only_as_a_requirement_or_summary_comes_with_citation_anchors_not_every_input():
    cause = view("Did DV pressure cause it?", "investigation")
    assert "F-CF-NONUNIQUE" in cause.findings and "F-CF-NONUNIQUE" in cause.lean and not any(i.startswith("cf.") for i in cause.records)
    repair = view("What did the tp2 repair show?", "investigation")
    assert "F-CF-NONUNIQUE" not in repair.lean and {"cf.tp2", "cf.dv_pressure", "cf.prewindow"} <= set(repair.records)
    deep = view("Why did this alert fire?", "investigation")
    assert "F-EVIDENCE-TIMING" in deep.lean and "temporal.2110" not in deep.records and "temporal.window" in deep.records
    timing = view("When did the evidence rise?", "investigation")
    assert "F-EVIDENCE-TIMING" not in timing.lean and "temporal.2110" in timing.records
    assert "35.08 to 100" in deep.sections["findings"]  # the finding text keeps its figures


def test_every_refusal_can_always_cite_a_loaded_record():
    for name in FOCUS:
        v = build_view(PACK, [name], [], "quick")
        for cat, r in REFUSALS.items():
            assert not r["cite"] or set(r["cite"]) & v.ids(), (name, cat)


def test_required_findings_follow_the_declared_depth():
    quick = view("Why did this alert fire?", "quick")
    deep = view("Why did this alert fire?", "investigation")
    assert "F-EVIDENCE-TIMING" not in quick.findings and "F-EVIDENCE-TIMING" in deep.findings
    assert "temporal.window" in deep.records


def test_a_question_that_matches_nothing_gets_the_default_and_a_bare_follow_up_keeps_the_last_topic():
    assert plan("hello") == ["alert"]
    assert plan("Tell me more", ["attribution", "label"]) == ["attribution", "label"]
    assert plan("Tell me more", ["nonsense"]) == ["alert"]
    assert plan("Show me the energy figure", ["attribution"]) == ["energy", "sources"]
    assert view("Tell me more", "auto", previous=["attribution"]).focus == ("attribution",)


def test_every_focus_resolves_on_every_case_and_names_real_findings():
    for case in CASES:
        pack = STORE.pack(case)
        for name in FOCUS:
            v = build_view(pack, [name], [], "technical")
            assert v.findings and v.records and set(v.focus) == {name}


# --- multi-turn --------------------------------------------------------------

def test_a_follow_up_gets_its_own_topic_and_earlier_exchanges_while_they_fit():
    th = thread(focus=["alert"], exchanges=[("Why did this alert fire?", "a" * 700)])
    b = build(PACK, th, "And what about the jump at 21:10?", "auto", CAP, 2)
    assert b.view.focus == ("timing",) and b.keep == 1 and "temporal.2110" in b.view.records
    assert any(m["role"] == "assistant" for m in b.msgs) and b.parts["history"] > 0


def test_the_candidate_test_table_is_dropped_before_any_topic():
    q = "What are the candidate causes?"
    full = build(PACK, thread(), q, "investigation", CAP, 2)
    lean = build(PACK, thread(), q, "investigation", full.est - 1, 2)
    assert full.view.tests and not lean.view.tests and lean.view.focus == full.view.focus and lean.view.dropped == ()
    assert set(lean.view.findings) == set(full.view.findings) and lean.view.schema["properties"]["next_test"]["enum"] == [None]
    assert lean.est < full.est and "possible_causes" in lean.view.purposes


def test_history_is_dropped_before_any_evidence():
    th = thread(focus=["alert"], exchanges=[("q1", "a" * 900), ("q2", "b" * 900)])
    full = build(PACK, th, "And what about the jump at 21:10?", "auto", CAP, 2)
    tight = build(PACK, th, "And what about the jump at 21:10?", "auto", full.est - 1, 2)
    assert full.keep == 2 and tight.keep < 2 and tight.view.focus == full.view.focus == ("timing",) and tight.view.dropped == ()
    assert set(tight.view.findings) == set(full.view.findings)


def test_a_bare_follow_up_in_vietnamese_keeps_the_previous_topic():
    th = thread(focus=["attribution"], exchanges=[("Đây có chắc chắn là rò rỉ khí không?", "x" * 500)])
    b = build(PACK, th, "Giải thích thêm giúp tôi.", "auto", CAP, 2)
    assert b.view.focus == ("attribution",) and len(b.view.tests) == 7


# --- fail closed -------------------------------------------------------------

def test_a_prompt_over_the_budget_is_refused_not_shortened():
    with pytest.raises(SelectionError) as e:
        build(PACK, thread(), "Why did this alert fire?", "quick", 1500, 2)
    assert e.value.reason == "context_over_budget" and "cap of 1500" in e.value.detail


BROAD = "Why did this alert fire, when did the evidence rise, what did the tp2 repair show and was it a leak?"


def test_a_too_broad_question_drops_whole_lower_priority_topics_and_says_so():
    b = build(PACK, thread(), BROAD, "auto", CAP, 2)
    assert b.view.focus == ("alert", "timing") and b.view.dropped == ("attribution", "repair")
    assert "Not loaded because the question is too broad" in b.view.sections["scope"] and "attribution, repair" in b.view.sections["scope"]
    assert "hyp.air_leakage" not in b.view.records and "hyp.air_leakage" not in json.dumps(b.view.schema)


@pytest.mark.parametrize("message", ["Why did this alert fire?", BROAD, "Was this definitely an air leak?",
                                     "Did DV pressure cause it, and how reliable is the detector?", "Explain the repair, the label and the timing"])
@pytest.mark.parametrize("depth", ["auto", "investigation", "technical"])
def test_no_cap_ever_returns_a_view_missing_a_required_or_mandatory_finding(message, depth):
    served = 0
    for cap in range(1800, 7000, 150):
        try:
            b = build(PACK, thread(), message, depth, cap, 2)
        except SelectionError as e:
            assert e.reason == "context_over_budget"
            continue
        served += 1
        v = b.view
        required = {f for p in v.purposes for f, d in PURPOSES[p]["require"] if depth_after(v.depth, d) and PACK.findings[f]["fires"]}
        assert b.est <= cap and "F-GAP" in v.findings and required <= set(v.findings)
        assert {i for f in v.findings.values() for i in (LEAN[f["id"]] if f["id"] in v.lean else f["inputs"]) if i in PACK.records} <= set(v.records)
    assert served


def test_a_missing_finding_or_dependency_stops_the_selection():
    gone = dataclasses.replace(PACK, findings={k: v for k, v in PACK.findings.items() if k != "F-MARGIN"})
    with pytest.raises(SelectionError, match="context_incomplete") as e:
        build_view(gone, ["alert"], [], "quick")
    assert "F-MARGIN" in e.value.detail
    no_input = dataclasses.replace(PACK, records={k: v for k, v in PACK.records.items() if k != "alert.prev_smoothed"})
    with pytest.raises(SelectionError, match="context_incomplete") as e:
        build_view(no_input, ["alert"], [], "quick")
    assert "alert.prev_smoothed" in e.value.detail
    no_core = dataclasses.replace(PACK, records={k: v for k, v in PACK.records.items() if k != "alert.margin"})
    with pytest.raises(SelectionError, match="context_incomplete"):
        build_view(no_core, ["sources"], [], "quick")


def test_a_required_finding_missing_from_the_ledger_is_incomplete_not_ignored():
    gone = dataclasses.replace(PACK, findings={k: v for k, v in PACK.findings.items() if k != "F-IDENT"})
    with pytest.raises(SelectionError, match="context_incomplete"):
        build_view(gone, ["attribution"], [], "investigation")


# --- the ledger still validates ------------------------------------------------

def test_a_reply_may_use_only_what_the_turn_offered_though_the_ledger_holds_more():
    v = view("Why did this alert fire?", "quick")
    ok, _ = parse(json.dumps(reply(U1, U2)))
    assert {f.code for f in check(ok, PACK, "Why?", "quick", v)} == set()
    outside = unit("The tp2 repair removes the alert at 21:20.", ["cf.tp2"], ["F-CF-NONUNIQUE"])
    r, _ = parse(json.dumps(reply(U1, U2, outside)))
    assert "not_offered" not in {f.code for f in check(r, PACK, "Why?", "quick")}  # the ledger alone accepts it
    assert "not_offered" in {f.code for f in check(r, PACK, "Why?", "quick", v)}
    wrong_purpose, _ = parse(json.dumps(reply(U1, U2, purposes=["possible_causes"])))
    assert "not_offered" in {f.code for f in check(wrong_purpose, PACK, "Why?", "quick", v)}


def test_numbers_are_still_checked_against_the_ledger_records_a_unit_cites():
    v = view("Why did this alert fire?", "quick")
    bad, _ = parse(json.dumps(reply(unit("The smoothed score was 61,000.", ["alert.smoothed"], ["F-MARGIN"]))))
    assert "number_not_cited" in {f.code for f in check(bad, PACK, "Why?", "quick", v)}


# --- through the service ----------------------------------------------------------

def test_the_model_receives_only_the_selected_slice_and_its_schema(tmp_path):
    fake = Fake(reply(U1, U2))
    c = client(tmp_path, fake)
    body = ask(c, start(c), "Why did this alert fire?", depth="quick").json()
    call = fake.calls[0]
    system = call["messages"][0]["content"]
    assert body["status"] == "ok" and "diagnostics" not in body
    assert "hyp.air_leakage" not in system and "cf.tp2" not in system and "t_demand_record" not in system and "SCOPE: evidence loaded" in system
    assert "cf.tp2" not in json.dumps(call["schema"]) and estimate_request(call["messages"], call["schema"]) <= CAP
    assert estimate_request(call["messages"], call["schema"]) < 3500


def test_diagnostics_are_returned_only_when_enabled_and_hold_no_text(tmp_path):
    fake = Fake(reply(U1, U2))
    c = client(tmp_path, fake, SYRUS_DIAGNOSTICS="1")
    body = ask(c, start(c), "Why did this alert fire? PRIVATE-QUESTION-TEXT", depth="quick").json()
    d = body["diagnostics"]
    assert d["focus"] == ["alert"] and d["dropped"] == [] and d["estimate"]["total"] <= d["cap"] == CAP
    assert d["calls"][0]["tokens_in"] == 3000 and d["calls"][0]["remaining_tokens"] == "4000" and d["calls"][0]["estimated_in"] > 0
    assert "PRIVATE" not in json.dumps(d) and KEY not in json.dumps(body)


def test_a_prompt_over_the_budget_makes_no_call_and_says_why(tmp_path):
    fake = Fake(reply(U1, U2))
    c = client(tmp_path, fake, patch=lambda cfg: cfg.update(tokens_per_minute=1500), SYRUS_DIAGNOSTICS="1")
    r = ask(c, start(c), "Why did this alert fire?")
    assert r.status_code == 503 and r.json()["reason"] == "context_over_budget" and not fake.calls
    assert r.json()["diagnostics"]["cap"] == 225 and "estimated tokens" in r.json()["diagnostics"]["detail"]


def test_a_narrowed_question_tells_the_user_in_either_language(tmp_path):
    c = client(tmp_path, Fake(reply(U1, U2)))
    body = ask(c, start(c), BROAD).json()
    assert body["narrowed"] and "candidate explanations" in body["narrowed"] and body["answer"].count(body["narrowed"]) == 1
    vi = ask(c, start(c), "Vì sao cảnh báo này xuất hiện, bằng chứng tăng lúc nào, sửa tp2 thì sao và có phải rò rỉ không?").json()
    assert "các giả thuyết giải thích" in vi["narrowed"]
    plain = ask(c, start(c), "Why did this alert fire?").json()
    assert plain["narrowed"] is None


def test_a_follow_up_uses_the_new_topic_with_compact_history(tmp_path):
    fake = Fake(reply(U1, U2))
    c = client(tmp_path, fake)
    h = start(c)
    assert ask(c, h, "Why did this alert fire?").json()["status"] == "ok"
    timing = unit("dv_pressure reached the 100th percentile at 21:10.", ["temporal.2110"])
    fake.outputs = [reply(timing, purposes=["evidence_and_sources"])]
    body = ask(c, h, "And what about the jump at 21:10?").json()
    second = fake.calls[1]
    assert body["status"] == "ok" and "temporal.2110" in json.dumps(second["schema"]) and "temporal.2110" not in json.dumps(fake.calls[0]["schema"])
    history = [m for m in second["messages"] if m["role"] == "assistant"]
    assert len(history) == 1 and history[0]["content"] == f"{U1['text']} {U2['text']}"


def test_a_retry_that_would_not_fit_the_budget_is_not_sent(tmp_path):
    first = build(PACK, thread(), "Why did this alert fire?", "quick", 10**6, 2).est
    tight = lambda cfg: cfg.update(tokens_per_minute=int((first + 900 + 60) / cfg["request_share"]))
    for patch, calls in ((None, 2), (tight, 1)):
        fake = Fake(reply(U1))  # misses a required finding, so the service retries once if it may
        c = client(tmp_path, fake, patch=patch)
        body = ask(c, start(c), "Why did this alert fire?", depth="quick").json()
        assert len(fake.calls) == calls and body["status"] == "ok" and body["also_relevant"]


# --- sanitised provider diagnostics --------------------------------------------

def groq(handler):
    return GroqAdapter(KEY, {"model": "qwen/qwen3.8-27b", "endpoint": "https://api.groq.com/openai/v1/chat/completions"},
                       httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def call(adapter):
    import asyncio
    return asyncio.run(adapter.complete([{"role": "user", "content": "PRIVATE-PROMPT"}], {"type": "object"}, 900, 5))


RATE = {"error": {"message": "Rate limit reached for model `qwen/qwen3.8-27b` in organization `org_secret123` on tokens per minute (TPM): Limit 8000, Used 5637, "
                             f"Requested 9062. Please try again in 37s. Your key {KEY}", "type": "tokens", "code": "rate_limit_exceeded", "failed_generation": "PRIVATE-OUTPUT"}}
HEADERS = {"retry-after": "37", "x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "2363", "x-ratelimit-reset-tokens": "2m59.56s",
           "x-ratelimit-remaining-requests": "999", "x-request-id": "req_01abc", "set-cookie": "session=secret"}


def test_a_rate_limit_error_carries_numbers_and_headers_but_no_text():
    with pytest.raises(LlmError) as e:
        call(groq(lambda r: httpx.Response(429, headers=HEADERS, json=RATE)))
    d = e.value.diag
    assert e.value.reason == "rate_limited" and e.value.retry_after == 37.0
    assert (d["http"], d["limit_tokens"], d["remaining_tokens"], d["reset_tokens"], d["remaining_requests"]) == (429, "8000", "2363", "2m59.56s", "999")
    assert (d["limit"], d["used"], d["requested"], d["error_type"], d["error_code"], d["request_id"]) == (8000, 5637, 9062, "tokens", "rate_limit_exceeded", "req_01abc")
    blob = json.dumps(d)
    assert KEY not in blob and "PRIVATE" not in blob and "org_secret123" not in blob and "secret" not in blob and "cookie" not in blob


def test_unsafe_header_and_body_values_are_dropped():
    bad = {"retry-after": "3; drop table", "x-ratelimit-remaining-tokens": "5" * 80, "x-request-id": "req id with spaces"}
    with pytest.raises(LlmError) as e:
        call(groq(lambda r: httpx.Response(400, headers=bad, json={"error": {"type": "invalid request", "code": "x" * 100, "message": "PRIVATE"}})))
    assert {k: v for k, v in e.value.diag.items() if k != "elapsed_ms"} == {"http": 400, "response_received": True} and e.value.reason == "provider_error"


def test_non_json_provider_errors_still_map_without_leaking(caplog):
    with caplog.at_level(logging.WARNING, logger="syrus.provider"), pytest.raises(LlmError) as e:
        call(groq(lambda r: httpx.Response(502, text=f"<html>PRIVATE {KEY}</html>")))
    assert e.value.reason == "provider_error" and e.value.diag["http"] == 502 and e.value.indeterminate
    assert "PRIVATE" not in caplog.text and KEY not in caplog.text and '"http": 502' in caplog.text


def test_a_success_reports_the_model_and_remaining_budget_only():
    ok = {"model": "qwen/qwen3.8-27b", "choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 2100, "completion_tokens": 310}}
    done = call(groq(lambda r: httpx.Response(200, headers=HEADERS, json=ok)))
    assert (done.tokens_in, done.tokens_out) == (2100, 310)
    assert done.diag["model"] == "qwen/qwen3.8-27b" and done.diag["remaining_tokens"] == "2363" and "PRIVATE" not in json.dumps(done.diag)


def test_the_service_returns_a_rate_limit_diagnosis_to_a_developer_only(tmp_path):
    err = LlmError("rate_limited", 37.0, {"http": 429, "limit": 8000, "used": 5637, "requested": 9062})

    class Failing:
        async def complete(self, *a):
            raise err

    off = client(tmp_path, Failing())
    on = client(tmp_path, Failing(), SYRUS_DIAGNOSTICS="1")
    quiet = ask(off, start(off), "Why did this alert fire?").json()
    loud = ask(on, start(on), "Why did this alert fire?").json()
    assert quiet == {"status": "offline", "reason": "rate_limited", "retry_after": 37.0}
    assert loud["diagnostics"]["calls"][0]["requested"] == 9062 and loud["diagnostics"]["estimate"]["total"] < CAP


# --- the smoke script's cost ----------------------------------------------------

def test_the_smoke_cost_comes_from_each_prompt_not_a_fixed_estimate():
    src = (ROOT / "scripts" / "syrus_smoke.py").read_text("utf-8")
    assert "6000" not in src
    out = subprocess.run([sys.executable, "-m", "scripts.syrus_smoke"], capture_output=True, text=True, cwd=ROOT, check=True,
                         env={**__import__("os").environ, "SYRUS_LLM_API_KEY": "", "SYRUS_CAPS_CONFIRMED": ""}).stdout
    tokens = [int(line.split("~")[1].split()[0]) for line in out.splitlines() if "input tokens" in line]
    assert len(tokens) == 6 and len(set(tokens)) > 3 and max(tokens) <= CAP and "worst case" in out


def test_the_service_imports_stay_slim():
    code = "import sys, backend.syrus_app.app, backend.syrus_app.selection; print([m for m in ('torch','numpy','pandas','scipy','sklearn') if m in sys.modules])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip()
    assert out == "[]"


def test_the_fixture_still_loads():
    assert yaml.safe_load((ROOT / "tests" / "fixtures" / "syrus_acceptance.yaml").read_text("utf-8"))["version"] == 1
