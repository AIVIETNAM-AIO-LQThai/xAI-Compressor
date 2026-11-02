"""Document, label and missing-evidence questions in English and Vietnamese, including the Vietnamese two-part question that failed live
with `kind_mismatch`; and the difference between `busy` and `paced`. Offline only: scripted model doubles, no key, no paid call."""
import json

import pytest

from backend.syrus_app.validator import check, parse
from tests.test_syrus_context import PACK, Fake, ask, client, reply, start, unit, view

VI = "Có nhật ký bảo trì hoặc tài liệu kỹ thuật nào xác nhận sự cố này chắc chắn do rò rỉ khí không? Nếu chưa có, cần bổ sung bằng chứng gì?"
EN = "Is there a maintenance log or technical document confirming this incident was definitely an air leak? If not, what evidence is needed?"

DOCS = {
    "en": unit("No maintenance log or technical document for this asset is loaded.", ["unavailable:documents"], [], "unavailable"),
    "vi": unit("Không có nhật ký bảo trì hay tài liệu kỹ thuật nào cho thiết bị này được nạp.", ["unavailable:documents"], [], "unavailable"),
}
LABEL = {
    "en": unit("The provider failure list labels the interval as an air leak, high stress; this is not an independent diagnosis, and AeroXAI's evidence does "
               "not establish it.", ["label.incident"], ["F-LABEL"], "dataset_label"),
    "vi": unit("Danh sách sự cố của nhà cung cấp ghi nhãn khoảng này là rò rỉ khí, áp lực cao; đây không phải chẩn đoán độc lập và bằng chứng của AeroXAI "
               "chưa xác lập nhãn này.", ["label.incident"], ["F-LABEL"], "dataset_label"),
}
IDENT = {
    "en": unit("Leakage and higher downstream demand cannot be told apart with this data; an independent record of downstream air use would be needed.",
               ["hyp.air_leakage", "hyp.downstream_demand", "unavailable:demand_measurement"], ["F-IDENT"], "hypothesis"),
    "vi": unit("Rò rỉ và nhu cầu khí phía hạ nguồn tăng không thể phân biệt được bằng dữ liệu này; cần một bản ghi độc lập về lượng khí sử dụng phía hạ nguồn.",
               ["hyp.air_leakage", "hyp.downstream_demand", "unavailable:demand_measurement"], ["F-IDENT"], "hypothesis"),
}
QUESTION = {"en": EN, "vi": VI}


def answer(*units, lang="en", **over):
    return reply(*units, purposes=["evidence_and_sources"], refusal="diagnosis_request", lang=lang, **over)


def codes(r, lang, depth="investigation"):
    parsed, bad = parse(json.dumps(r))
    assert parsed, bad
    return {f.code for f in check(parsed, PACK, QUESTION[lang], depth, view(QUESTION[lang], depth))}


# --- what the question selects -----------------------------------------------------------------

@pytest.mark.parametrize("lang", ["en", "vi"])
def test_the_two_part_document_question_selects_the_label_and_the_identifiability_limit(lang):
    v = view(QUESTION[lang], "investigation")
    assert v.required == ("F-LABEL", "F-IDENT") and "possible_causes" not in v.purposes
    assert {"label.incident", "unavailable:documents", "unavailable:demand_measurement", "hyp.air_leakage", "hyp.downstream_demand"} <= set(v.records)


# --- the validator: the split answer passes, the merged one still fails ---------------------------------

@pytest.mark.parametrize("lang", ["en", "vi"])
def test_a_split_answer_passes_in_both_languages(lang):
    assert codes(answer(DOCS[lang], LABEL[lang], IDENT[lang], lang=lang), lang) == set()


def test_the_label_record_names_the_candidate_its_wording_uses():
    assert PACK.records["label.incident"]["value"]["names_candidates"] == ["air_leakage"]


@pytest.mark.parametrize("lang", ["en", "vi"])
def test_a_unit_mixing_the_label_with_the_candidates_still_fails_kind_mismatch(lang):
    merged = unit(LABEL[lang]["text"] + " " + IDENT[lang]["text"], ["label.incident", "hyp.air_leakage", "hyp.downstream_demand", "unavailable:demand_measurement"],
                  ["F-LABEL", "F-IDENT"], "dataset_label")
    assert "kind_mismatch" in codes(answer(DOCS[lang], merged, lang=lang), lang)


@pytest.mark.parametrize(("lang", "text"), [("en", "The provider list labels the interval an air leak; higher downstream air demand is not in the label."),
                                            ("vi", "Danh sách của nhà cung cấp ghi nhãn rò rỉ khí; nhu cầu khí phía hạ nguồn tăng không có trong nhãn.")])
def test_a_label_unit_may_name_only_the_candidate_the_label_names(lang, text):
    other = unit(text, ["label.incident"], ["F-LABEL"], "dataset_label")
    assert "entity_not_cited" in codes(answer(DOCS[lang], other, IDENT[lang], lang=lang), lang)


@pytest.mark.parametrize(("lang", "text"), [("en", "No maintenance log confirms the label."), ("vi", "Không có nhật ký bảo trì nào xác nhận nhãn này.")])
def test_naming_documents_without_citing_their_absence_still_fails(lang, text):
    bare = unit(text, ["label.incident"], [], "dataset_label")
    assert "source_not_cited" in codes(answer(bare, LABEL[lang], IDENT[lang], lang=lang), lang)


@pytest.mark.parametrize("lang", ["en", "vi"])
def test_missing_external_evidence_is_stated_by_citing_what_is_not_loaded(lang):
    need = IDENT[lang]
    assert "unavailable:demand_measurement" in need["record_ids"] and codes(answer(DOCS[lang], LABEL[lang], need, lang=lang), lang) == set()


# --- through the service ---------------------------------------------------------------------------------

@pytest.mark.parametrize("lang", ["en", "vi"])
def test_a_merged_first_reply_gets_a_concrete_repair_hint_and_the_split_retry_passes(tmp_path, lang):
    merged = unit(LABEL[lang]["text"] + " " + IDENT[lang]["text"], ["label.incident", "hyp.air_leakage"], ["F-LABEL", "F-IDENT"], "dataset_label")
    fake = Fake(answer(DOCS[lang], merged, lang=lang), answer(DOCS[lang], LABEL[lang], IDENT[lang], lang=lang))
    c = client(tmp_path, fake, SYRUS_DIAGNOSTICS="1")
    body = ask(c, start(c), QUESTION[lang], depth="investigation").json()
    correction = fake.calls[1]["messages"][-1]["content"]
    assert "kind_mismatch" in correction and "one per record kind" in correction
    assert body["status"] == "ok" and body["provenance"]["validator"] == "passed" and body["refusal"] == "diagnosis_request"
    first = body["diagnostics"]["calls"][0]
    assert first["verdict"] == "rejected" and "kind_mismatch" in first["codes"]
    assert first["units"][1] == {"kind": "dataset_label", "cited_kinds": ["dataset_label", "hypothesis"]}
    blob = json.dumps(first, ensure_ascii=False)
    assert "failure list" not in blob and "Danh sách" not in blob and "AeroXAI" not in blob  # kinds and codes only, never text


@pytest.mark.parametrize("lang", ["en", "vi"])
def test_a_follow_up_about_missing_evidence_does_not_demand_what_was_already_said(tmp_path, lang):
    follow = {"en": "If not, what evidence would be needed?", "vi": "Nếu chưa có, cần bổ sung bằng chứng gì?"}[lang]
    fake = Fake(answer(DOCS[lang], LABEL[lang], IDENT[lang], lang=lang))
    c = client(tmp_path, fake)
    h = start(c)
    assert ask(c, h, QUESTION[lang], depth="investigation").json()["provenance"]["validator"] == "passed"
    fake.outputs = [answer(IDENT[lang], lang=lang)]
    body = ask(c, h, follow, depth="investigation").json()
    assert body["status"] == "ok" and body["also_relevant"] == [] and len(fake.calls) == 2


def test_the_unit_kind_diagnostic_never_echoes_an_unknown_kind_string():
    from backend.syrus_app.service import KIND_NAMES
    assert "dataset_label" in KIND_NAMES and "background" in KIND_NAMES and "free text" not in KIND_NAMES


# --- busy and paced -------------------------------------------------------------------------------------

def paced_client(tmp_path, fake):
    return client(tmp_path, fake, patch=lambda cfg: cfg.update(governor={"concurrent_calls": 1, "window_share": 0.9}))


def test_a_full_minute_window_is_paced_not_busy_and_nothing_is_sent(tmp_path):
    fake = Fake(answer(DOCS["en"], LABEL["en"], IDENT["en"]))
    c = paced_client(tmp_path, fake)
    svc = c.app.state.svc
    t = svc.governor.admit(6500)[0]
    svc.governor.release(t, 6500)  # this minute's allowance is nearly used
    r = ask(c, start(c), EN, depth="investigation")
    assert r.status_code == 503 and r.json()["reason"] == "paced" and r.json()["retry_after"] > 1 and not fake.calls


def test_a_call_in_flight_is_busy(tmp_path):
    fake = Fake(answer(DOCS["en"], LABEL["en"], IDENT["en"]))
    c = paced_client(tmp_path, fake)
    held = c.app.state.svc.governor.admit(100)[0]
    r = ask(c, start(c), EN, depth="investigation")
    assert r.json()["reason"] == "busy" and not fake.calls
    c.app.state.svc.governor.release(held, 0)


def test_a_refused_turn_is_not_counted_and_the_same_question_can_be_sent_again(tmp_path):
    fake = Fake(answer(DOCS["en"], LABEL["en"], IDENT["en"]))
    c = paced_client(tmp_path, fake)
    svc, h = c.app.state.svc, start(c)
    held = svc.governor.admit(100)[0]
    assert ask(c, h, EN, depth="investigation").json()["reason"] == "busy"
    svc.governor.release(held, 0)
    thread_after = svc.sessions.threads[next(iter(svc.sessions.threads))]
    assert thread_after.turns == 0 and not thread_after.exchanges
    assert ask(c, h, EN, depth="investigation").json()["status"] == "ok" and len(fake.calls) == 1
