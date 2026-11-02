"""The documented incident label: what a "was it definitely a leak?" question must convey, that a matching number is not conveyance, the
acceptance fixture agreeing with the completeness rules, and the label staying out of as-of mode. Offline only."""
import json

import pytest

from backend.syrus_app.policy import CONVEY
from backend.syrus_app.validator import conveyed, missing, parse, tokens
from scripts.syrus_tokens import items
from tests.test_syrus_context import (
    CASE,
    PACK,
    STORE,
    Fake,
    ask,
    client,
    reply,
    start,
    unit,
    view,
)

LABEL_EN = unit("The repository's transcription of the provider failure list labels 2020-05-29 23:30 to 2020-05-30 06:00 as \"Air leak, high stress\"; "
                "this is not an independent diagnosis, and AeroXAI's evidence does not establish it.", ["label.incident"], ["F-LABEL"], kind="dataset_label")
IDENT_EN = unit("Leakage and higher downstream demand cannot be told apart with the data recorded for this asset.",
                ["hyp.air_leakage", "hyp.downstream_demand", "unavailable:demand_measurement"], ["F-IDENT"], kind="hypothesis")
LABEL_VI = unit("Bản phiên âm danh sách sự cố của nhà cung cấp trong kho mã gán nhãn 2020-05-29 23:30 đến 2020-05-30 06:00 là \"Air leak, high stress\"; "
                "bằng chứng của AeroXAI không xác lập nhãn này như một chẩn đoán độc lập.", ["label.incident"], ["F-LABEL"], kind="dataset_label")
IDENT_VI = unit("Rò rỉ và nhu cầu khí phía hạ nguồn tăng không thể phân biệt được bằng dữ liệu ghi nhận cho thiết bị này.",
                ["hyp.air_leakage", "hyp.downstream_demand", "unavailable:demand_measurement"], ["F-IDENT"], kind="hypothesis")
NUMBERS_ONLY = unit("The interval ran from 2020-05-29 23:30 to 2020-05-30 06:00.", ["label.incident"], ["F-LABEL"], kind="dataset_label")
EN, VI = "Was this definitely an air leak?", "Đây có chắc chắn là rò rỉ khí không?"


def parsed(*units, lang="en", **over):
    r, bad = parse(json.dumps(reply(*units, purposes=["evidence_and_sources"], refusal="diagnosis_request", lang=lang, **over)))
    assert r, bad
    return r


# --- what the question requires ---------------------------------------------------------------

@pytest.mark.parametrize("depth", ["quick", "investigation", "technical"])
@pytest.mark.parametrize("message", [EN, VI, "Is it really a leak?", "Có phải rò rỉ thật sự không?"])
def test_a_leak_certainty_question_requires_the_label_and_identifiability_only(message, depth):
    v = view(message, depth)
    assert v.required == ("F-LABEL", "F-IDENT") and "F-TIMING" not in v.required and "F-CF-NONUNIQUE" not in v.required
    assert {"F-LABEL", "F-IDENT"} <= set(v.findings) and "label.incident" in v.records


def test_a_generic_candidate_question_does_not_require_the_label():
    for message in ("What are the candidate causes?", "Is it the regulator or a sensor problem?", "Did DV pressure cause it?"):
        assert "F-LABEL" not in view(message, "investigation").required, message
    assert view("Was it definitely a leak, and which signal drove it?", "investigation").required == ("F-CANDIDATES", "F-IDENT", "F-CF-NONUNIQUE")


def test_timing_is_required_only_when_the_question_is_about_onset_prediction_or_lead_time():
    assert "F-TIMING" not in view(EN, "investigation").required
    for message in ("Did the model predict it 2 hours early?", "How long before the onset did the alert come?", "Dự đoán sớm được không?"):
        assert "F-TIMING" in view(message, "quick").required, message


# --- what counts as conveying it --------------------------------------------------------------

def test_repeating_a_timestamp_does_not_convey_the_label():
    r = parsed(NUMBERS_ONLY, IDENT_EN)
    assert tokens(NUMBERS_ONLY["text"]) & tokens(PACK.findings["F-LABEL"]["text"])  # the earlier figure-only rule would have accepted it
    assert conveyed(r, PACK) == {"F-IDENT"}
    assert missing(r, PACK, "quick", set(), view(EN, "quick")) == ["F-LABEL"]


@pytest.mark.parametrize(("text", "ok"), [
    ("The provider's list labels it \"Air leak, high stress\".", False),  # no statement that AeroXAI does not establish it
    ("AeroXAI's evidence does not establish an air leak.", False),  # no provenance
    ("The provider failure list says the interval is a leak.", False),  # no limit
    ("The provider failure list labels the interval \"Air leak, high stress\"; this is not an independent finding by AeroXAI.", True),
    ("The transcribed provider list shows air leak, high stress, which AeroXAI cannot confirm.", True),
    ("Danh sách của nhà cung cấp ghi nhãn rò rỉ, nhưng AeroXAI chưa xác nhận điều đó.", True),
    ("Danh sách của nhà cung cấp ghi nhãn rò rỉ.", False),
])
def test_the_label_is_conveyed_only_when_its_source_and_its_limit_are_both_stated(text, ok):
    r = parsed(unit(text, ["label.incident"], ["F-LABEL"], kind="dataset_label"))
    assert (conveyed(r, PACK) == {"F-LABEL"}) is ok


@pytest.mark.parametrize(("text", "ok"), [
    ("Leakage is one candidate and higher downstream demand is another.", False),
    ("Leakage and demand are both candidates and cannot be told apart.", True),
    ("The data cannot distinguish leakage from higher downstream demand.", True),
    ("Rò rỉ và nhu cầu là hai giả thuyết.", False),
    ("Không thể phân biệt rò rỉ với nhu cầu khí tăng.", True),
])
def test_identifiability_is_conveyed_only_when_both_candidates_and_the_limit_are_stated(text, ok):
    r = parsed(unit(text, ["hyp.air_leakage", "hyp.downstream_demand"], ["F-IDENT"], kind="hypothesis"))
    assert (conveyed(r, PACK) == {"F-IDENT"}) is ok


def test_the_reviewed_texts_convey_themselves_and_two_units_can_share_a_finding():
    for f in ("F-LABEL", "F-IDENT"):
        assert all(rx.search(PACK.findings[f]["text"]) for rx in CONVEY[f]), f
    half_a = unit("The provider failure list labels the interval \"Air leak, high stress\".", ["label.incident"], ["F-LABEL"], kind="dataset_label")
    half_b = unit("AeroXAI's evidence does not establish this label.", ["label.incident"], ["F-LABEL"], kind="dataset_label")
    assert conveyed(parsed(half_a, half_b), PACK) == {"F-LABEL"}


# --- through the service ---------------------------------------------------------------------------

@pytest.mark.parametrize(("message", "lang", "units"), [(EN, "en", (LABEL_EN, IDENT_EN)), (VI, "vi", (LABEL_VI, IDENT_VI))])
def test_a_complete_answer_to_the_leak_question_passes_without_added_findings(tmp_path, message, lang, units):
    fake = Fake(reply(*units, purposes=["evidence_and_sources"], refusal="diagnosis_request", lang=lang))
    c = client(tmp_path, fake)
    body = ask(c, start(c), message, depth="investigation").json()
    assert body["status"] == "ok", body.get("failed") or body
    assert body["provenance"]["validator"] == "passed" and body["also_relevant"] == [] and len(fake.calls) == 1 and body["lang"] == lang


@pytest.mark.parametrize(("message", "lang", "ident"), [(EN, "en", IDENT_EN), (VI, "vi", IDENT_VI)])
def test_an_answer_that_skips_the_label_gets_the_reviewed_label_text_added(tmp_path, message, lang, ident):
    fake = Fake(reply(NUMBERS_ONLY, ident, purposes=["evidence_and_sources"], refusal="diagnosis_request", lang=lang))
    c = client(tmp_path, fake)
    body = ask(c, start(c), message, depth="investigation").json()
    added = {a["finding_id"]: a["text"] for a in body["also_relevant"]}
    assert len(fake.calls) == 2 and "required_finding: F-LABEL" in fake.calls[1]["messages"][-1]["content"]
    assert list(added) == ["F-LABEL"] and "does not establish this label" in added["F-LABEL"]
    assert body["provenance"]["validator"] == "passed_with_added_findings"


def test_the_label_is_not_demanded_of_a_generic_candidate_question(tmp_path):
    cands = unit("All candidate explanations are open and Syrus does not rank them.", ["hyp.air_leakage"], ["F-CANDIDATES"], kind="hypothesis")
    fake = Fake(reply(cands, IDENT_EN, purposes=["possible_causes"]))
    c = client(tmp_path, fake)
    body = ask(c, start(c), "What are the candidate causes?", depth="quick").json()
    assert body["status"] == "ok" and body["also_relevant"] == [] and len(fake.calls) == 1


# --- acceptance fixture and as-of mode ----------------------------------------------------------------

def test_the_fixture_lists_every_finding_the_server_requires_for_each_turn():
    by_alert = {c["alert_time"][:10]: c["case_id"] for c in STORE.manifest["cases"]}
    for item in items():
        pack = STORE.pack(by_alert[item["alert"]])
        for turn in item["turns"]:
            required = set(view(turn["text"], item["depth"], pack).required)
            assert required <= set(turn["essentials"]), (item["id"], sorted(required - set(turn["essentials"])))
    leak = {i["id"]: i["turns"][0]["essentials"] for i in items() if i["id"].startswith("definitely_a_leak")}
    assert leak == {"definitely_a_leak": ["F-LABEL", "F-IDENT"], "definitely_a_leak_vi": ["F-LABEL", "F-IDENT"]}


RETRO_ONLY = ["high stress", "label.incident", "label.timing", "F-LABEL", "F-TIMING", "F-SELECTION", "Documented incident", "case.status"]


def test_the_documented_label_is_in_no_as_of_file_and_every_label_record_is_retrospective_only():
    for case in STORE.manifest["cases"]:
        asof, retro = STORE.get(case["case_id"], "asof"), STORE.get(case["case_id"], "retro")
        blob = json.dumps(asof, ensure_ascii=False)
        assert not [t for t in RETRO_ONLY if t in blob], case["case_id"]
        assert "high stress" in json.dumps(retro)
        assert {r["visibility"] for r in retro["records"] if r["id"].startswith(("label.", "case.", "bench."))} == {"retro"}
        assert {f["visibility"] for f in retro["findings"] if f["id"] in ("F-LABEL", "F-TIMING", "F-SELECTION")} == {"retro"}


def test_the_as_of_api_and_chat_never_carry_the_label(tmp_path):
    fake = Fake(reply(LABEL_EN, IDENT_EN, purposes=["evidence_and_sources"], refusal="diagnosis_request"))
    c = client(tmp_path, fake)
    for mode, has in (("asof", False), ("retro", True)):
        text = c.get(f"/api/cases/{CASE}", params={"mode": mode}).text
        assert ("high stress" in text) is has and ("F-LABEL" in text) is has
    assert c.get(f"/api/cases/{CASE}").text == c.get(f"/api/cases/{CASE}", params={"mode": "asof"}).text  # the default is as-of
    h = {"Authorization": "Bearer " + c.post("/api/session", json={"case_id": CASE, "mode": "asof"}).json()["token"]}
    for message in (EN, VI):
        r = ask(c, h, message)
        assert r.status_code == 403 and "high stress" not in r.text
    assert fake.calls == []


def test_the_model_pack_is_built_from_the_retrospective_file_only():
    assert PACK.sha256 == STORE.get(CASE, "retro")["sha256"] != STORE.get(CASE, "asof")["sha256"]
