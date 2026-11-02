"""Assembling what the user sees: the model's units, the server's qualifications inline after the claims that trigger them,
and the server-rendered additions. A qualification is part of its claim's block, never a footnote."""
from backend.syrus_app.context import Pack
from backend.syrus_app.policy import (
    QUALS,
    QUESTION_PURPOSES,
    REFUSALS,
    TEXT,
    VIETNAMESE,
    triggers,
)
from backend.syrus_app.sessions import Thread
from backend.syrus_app.validator import Reply


def pick_lang(reply_lang: str, message: str) -> str:
    return "vi" if VIETNAMESE.search(message) else reply_lang


def qualifications(direct: set[str], text: str, forced: list[str], thread_quals: set[str], shown: set[str], lang: str) -> list[dict]:
    """Qualifications for one unit, in table order, each at most once per answer. Critical ones repeat across answers in short form."""
    out = []
    for q in QUALS:
        on = q["id"] in forced or triggers(q, direct, text)
        if not on or q["id"] in shown or (q["severity"] == "context" and q["id"] in thread_quals):
            continue
        short = q["severity"] == "critical" and q["id"] in thread_quals
        out.append({"id": q["id"], "severity": q["severity"], "short": short, "text": q[f"short_{lang}" if short else lang]})
        shown.add(q["id"])
    return out


def question(r: Reply, pack: Pack, thread: Thread, turn_no: int, lang: str) -> dict | None:
    """The follow-up comes from the test table, only for candidate questions, never repeated, at most every other turn."""
    t = pack.tests.get(r.next_test or "")
    if not t or r.refusal or not set(r.purposes) & QUESTION_PURPOSES or t["result"] != "not_run" or t["id"] in thread.asked:
        return None
    if thread.last_question and turn_no - thread.last_question < 2:
        return None
    return {"test_id": t["id"], "text": t["follow_up"][lang]}


def assemble(r: Reply, pack: Pack, thread: Thread, depth: str, lang: str, added: list[str], model: str) -> tuple[dict, set[str]]:
    turn_no = thread.turns + 1
    forced = REFUSALS.get(r.refusal or "", {}).get("quals", [])
    shown, blocks = set(), []
    for n, u in enumerate(r.units):
        quals = qualifications(set(u.record_ids) | set(u.finding_ids), u.text, forced if n == 0 else [], thread.quals, shown, lang)
        blocks.append({"text": u.text, "kind": u.kind, "records": u.record_ids, "findings": u.finding_ids, "qualifications": quals,
                       "display": " ".join([u.text, *(q["text"] for q in quals)])})
    also = [{"finding_id": f, "text": pack.findings[f]["text"]} for f in added]
    op = TEXT["operator_note"][lang] if r.operator_statement else None
    nxt = question(r, pack, thread, turn_no, lang)
    paras = [b["display"] for b in blocks]
    if also:
        paras.append(f"{TEXT['also_relevant'][lang]}: " + " ".join(a["text"] for a in also))
    paras += [x for x in (op, nxt and nxt["text"]) if x]
    cited = {i for u in r.units for i in u.record_ids}
    out = {"status": "ok", "lang": lang, "depth": depth, "purposes": r.purposes, "refusal": r.refusal, "blocks": blocks, "also_relevant": also,
            "operator_note": op, "next_question": nxt, "alternatives": r.alternatives, "gaps": r.gaps, "answer": "\n\n".join(paras),
            "provenance": {"case_id": pack.case_id, "case_sha256": pack.sha256, "document_index_sha256": None, "model": model, "cited_records": len(cited),
                           "validator": "passed_with_added_findings" if added else "passed"}}
    return out, shown


def remember(thread: Thread, reply: Reply, out: dict, shown: set[str], message: str, limit: int) -> None:
    """Update the thread after a validated answer; prior answers are history for the model, never evidence."""
    n = thread.turns + 1
    for f in {f for u in reply.units for f in u.finding_ids} | {a["finding_id"] for a in out["also_relevant"]}:
        thread.stated[f] = n
    thread.quals |= shown
    if out["next_question"]:
        thread.asked.add(out["next_question"]["test_id"])
        thread.last_question = n
    if reply.operator_statement:
        thread.operator = [*thread.operator, message[:200]][-limit:]
    thread.exchanges.append((message, out["answer"]))
    thread.turns = n
