"""Deterministic checks on a model reply. The model writes cited units; nothing is shown that fails one of these.

Lexical checks supplement the structural ones (every unit that names a case entity must cite a record carrying it); they
cannot catch every rewording, so the acceptance set and human review remain part of release.
"""
import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from backend.syrus_app.context import Pack
from backend.syrus_app.policy import (
    CASE_REFERENCE,
    CONVEY,
    DEPTH,
    DEPTHS,
    FORBIDDEN,
    FORMULA,
    HYPOTHESIS_VERBS,
    NEGATION,
    PURPOSES,
    REFUSALS,
    TOPICS,
    depth_after,
)

TOKEN = re.compile(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}|(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
SIGNALS = ["tp2", "tp3", "h1", "dv_pressure", "reservoirs", "oil_temperature", "motor_current", "comp", "dv_electric", "towers", "mpg", "lps",
           "pressure_switch", "oil_level", "caudal_impulses"]
SIGNAL = {s: re.compile(r"\b(?:" + "|".join({re.escape(s), re.escape(s.replace("_", " "))}) + r")\b", re.IGNORECASE) for s in SIGNALS}
FORBIDDEN_RE = [(name, re.compile(rx, re.IGNORECASE), exempt) for name, rx, exempt in FORBIDDEN]
TOPIC_RE = [(name, re.compile(rx, re.IGNORECASE), need) for name, rx, need in TOPICS]


class Unit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    kind: str
    record_ids: list[str]
    finding_ids: list[str]


class Reply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lang: Literal["en", "vi"]
    purposes: list[str]
    units: list[Unit]
    alternatives: list[str]
    gaps: list[str]
    next_test: str | None
    refusal: str | None
    operator_statement: bool


@dataclass(frozen=True)
class Fail:
    code: str
    detail: str = ""

    def __str__(self):
        return f"{self.code}: {self.detail}" if self.detail else self.code


def parse(raw: str) -> tuple[Reply | None, list[Fail]]:
    try:
        return Reply.model_validate_json(raw), []
    except (ValidationError, ValueError) as e:
        return None, [Fail("schema", str(e).splitlines()[0][:120])]


def choose_depth(pref: str, purposes: list[str]) -> str:
    """The user's choice, or the deepest default among the declared purposes."""
    if pref in DEPTH:
        return pref
    return max((PURPOSES[p]["depth"] for p in purposes if p in PURPOSES), key=DEPTHS.index, default="quick")


def tokens(text: str) -> set[str]:
    return set(TOKEN.findall(text))


def carried(unit: Unit, pack: Pack, view=None) -> set[str]:
    """Record IDs a unit stands on: those it cites plus the inputs of the findings it cites. With a View, only inputs the turn
    offered count, so citing a finding that was shown with an anchor record does not vouch for records the model never saw."""
    ids = set(unit.record_ids)
    for f in unit.finding_ids:
        inputs = set(pack.findings[f]["inputs"]) & set(pack.records)
        ids |= inputs & set(view.records) if view else inputs
    return ids


def _blob(ids: set[str], pack: Pack) -> str:
    return " ".join(f"{i} {json.dumps(pack.records[i]['value'], ensure_ascii=False)} {json.dumps(pack.records[i]['display'], ensure_ascii=False)}" for i in ids).lower()


def _hits(rx: re.Pattern, text: str, exempt: bool) -> bool:
    return any(not (exempt and NEGATION.search(text[max(0, m.start() - 40):m.start()])) for m in rx.finditer(text))


def not_offered(r: Reply, view) -> set[str]:
    """IDs and purposes a reply uses that this turn's prompt did not offer. The ledger may hold them; the model was never shown them."""
    used = {*r.alternatives, *r.gaps, *(i for u in r.units for i in (*u.record_ids, *u.finding_ids))} | ({r.next_test} if r.next_test else set())
    return (used - view.ids()) | (set(r.purposes) - set(view.purposes))


def check(r: Reply, pack: Pack, message: str, depth: str, view=None) -> list[Fail]:
    """All failed checks for a parsed reply against the whole ledger and, if given, the View this turn offered; empty means it passes."""
    out = []
    if view and (extra := not_offered(r, view)):
        out.append(Fail("not_offered", ", ".join(sorted(extra))))
    if not 1 <= len(r.purposes) <= 2 or set(r.purposes) - set(PURPOSES):
        out.append(Fail("purposes", "declare one or two known purposes"))
    hyps = {i for i, x in pack.records.items() if x["kind"] == "hypothesis"}
    gaps = {i for i, x in pack.records.items() if x["kind"] == "unavailable"}
    if r.refusal and r.refusal not in REFUSALS:
        out.append(Fail("refusal", r.refusal))
    if set(r.alternatives) - hyps or set(r.gaps) - gaps or (r.next_test and r.next_test not in pack.tests):
        out.append(Fail("ids", "unknown alternative, gap or test"))
    if not r.units:
        return [*out, Fail("no_units")]
    if len(r.units) > DEPTH[depth]["units"]:
        out.append(Fail("depth_budget", f"{len(r.units)} units; {depth} allows {DEPTH[depth]['units']}"))
    for n, u in enumerate(r.units, 1):
        out += [Fail(f.code, f"unit {n}: {f.detail}") for f in check_unit(u, r, pack, message, depth, view)]
    cites = set().union(*(set(u.record_ids) | set(u.finding_ids) for u in r.units))
    need = REFUSALS.get(r.refusal or "", {}).get("cite")
    if need and not cites & set(need):
        out.append(Fail("refusal_citation", f"{r.refusal} must cite one of {', '.join(need)}"))
    return out


def check_unit(u: Unit, r: Reply, pack: Pack, message: str, depth: str, view=None) -> list[Fail]:
    out, text = [], u.text
    if set(u.record_ids) - set(pack.records) or set(u.finding_ids) - set(pack.findings):
        return [Fail("ids", "unknown record or finding")]
    if FORMULA.search(text) and not DEPTH[depth]["formulas"]:
        out.append(Fail("formula", f"formulas need technical depth, not {depth}"))
    have = carried(u, pack, view)
    stated = " ".join(pack.findings[f]["text"] for f in u.finding_ids)  # reviewed finding text the model was shown: its figures and signals are citable
    blob = _blob(have, pack) + " " + stated.lower()
    for name, rx, exempt in FORBIDDEN_RE:
        if _hits(rx, text, exempt):
            out.append(Fail("forbidden_phrase", name))
    if u.kind == "background":
        if u.record_ids or u.finding_ids:
            out.append(Fail("background_citation", "a background unit cites nothing"))
        if (TOKEN.search(text) or any(rx.search(text) for rx in SIGNAL.values()) or re.search(CASE_REFERENCE, text, re.IGNORECASE)
                or any(rx.search(text) for _, rx, _ in TOPIC_RE)):
            out.append(Fail("background_content", "no numbers, signals, case references or sources"))
        return out
    if u.kind not in {x["kind"] for x in pack.records.values()}:
        return [*out, Fail("kind", u.kind)]
    if not u.record_ids:
        out.append(Fail("uncited", "a claim unit cites at least one record"))
    wrong = {i for i in u.record_ids if pack.records[i]["kind"] not in (u.kind, "unavailable")}
    if wrong:
        out.append(Fail("kind_mismatch", f"{u.kind} unit cites {', '.join(sorted(wrong))}"))
    allowed = set().union(tokens(stated), *(tokens(json.dumps(pack.records[i]["display"], ensure_ascii=False)) for i in have))
    if r.refusal:
        allowed |= tokens(message)
    if extra := tokens(text) - allowed:
        out.append(Fail("number_not_cited", ", ".join(sorted(extra))))
    for s, rx in SIGNAL.items():
        if rx.search(text) and s not in blob:
            out.append(Fail("entity_not_cited", s))
    for hid, rxs in pack.hyp_patterns.items():
        if any(rx.search(text) for rx in rxs) and not (f"hyp.{hid}" in have or hid in blob):
            out.append(Fail("entity_not_cited", f"candidate {hid}"))
    for name, rx, need in TOPIC_RE:
        if rx.search(text) and need not in have:
            out.append(Fail("source_not_cited", f"{name} needs {need}"))
    if u.kind == "hypothesis" and re.search(HYPOTHESIS_VERBS, text, re.IGNORECASE):
        out.append(Fail("forbidden_phrase", "causal verb in a candidate claim"))
    return out


def conveyed(r: Reply, pack: Pack) -> set[str]:
    """Findings the answer actually states. A finding with phrase groups (CONVEY) needs every group in the units that cite it; any other
    finding needs a unit that cites it and, if it states figures, repeats at least one."""
    out = set()
    for f in {f for u in r.units for f in u.finding_ids}:
        mine = [u for u in r.units if f in u.finding_ids]
        if f in CONVEY:
            said = " ".join(u.text for u in mine)
            if all(rx.search(said) for rx in CONVEY[f]):
                out.add(f)
        elif not (figures := tokens(pack.findings[f]["text"])) or any(figures & tokens(u.text) for u in mine):
            out.add(f)
    return out


def missing(r: Reply, pack: Pack, depth: str, stated: set[str], view=None) -> list[str]:
    """Findings the answer must convey that it does not and the last turns have not already stated. With a View these are the ones the
    server set from the question's topics and depth, plus any the declared purposes add; a generic declared purpose removes none."""
    need = [f for p in r.purposes if p in PURPOSES for f, d in PURPOSES[p]["require"] if depth_after(depth, d) and pack.findings.get(f, {}).get("fires")]
    done = conveyed(r, pack)
    return [f for f in dict.fromkeys([*(view.required if view else ()), *need]) if f not in done and f not in stated]
