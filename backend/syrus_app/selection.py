"""Question-relevant context for one turn.

The complete case ledger (Pack) stays authoritative for validation. A View is the slice of it that one turn's prompt and response
schema may use. It is chosen deterministically from the question (or the previous turn's focus for a bare follow-up), closed over
what the chosen findings depend on, and extended with every finding the allowed purposes require. A slice that cannot be built
completely raises SelectionError instead of shrinking silently; prompt.build drops whole weaker focuses, never part of a finding,
when a slice does not fit the token budget.
"""
from dataclasses import dataclass

from backend.syrus_app.context import (
    HIDDEN,
    Pack,
    make_schema,
    render_findings,
    render_records,
    render_tests,
)
from backend.syrus_app.policy import (
    CORE_FINDINGS,
    CORE_PURPOSES,
    CORE_RECORDS,
    DEFAULT_FOCUS,
    DEPTH,
    DEPTHS,
    FOCUS,
    LEAN,
    PURPOSES,
    depth_after,
)


class SelectionError(Exception):
    """The context for a question cannot be built completely or does not fit; `reason` is shown, `detail` is for the log."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason, self.detail = reason, detail


@dataclass(frozen=True)
class View:
    focus: tuple      # focuses applied, in priority order
    dropped: tuple    # matched focuses left out to fit the budget
    purposes: tuple   # purposes an answer may declare
    depth: str        # deepest depth the required findings were computed for
    lean: tuple       # findings present with citation anchors only, not all their input records
    records: dict     # id -> record the model may cite this turn
    findings: dict    # id -> finding
    tests: dict       # id -> test (only when candidates are compared)
    pairs: list
    schema: dict
    sections: dict    # rendered text by part: head, scope, records, findings, tests

    @property
    def text(self) -> str:
        return "\n\n".join(x for x in self.sections.values() if x)

    def ids(self) -> set:
        return set(self.records) | set(self.findings) | set(self.tests)


def classify(message: str) -> list[str]:
    """Focuses a question touches, in priority order."""
    return [name for name, f in FOCUS.items() if f["pattern"].search(message)]


def plan(message: str, previous: list[str] | tuple = ()) -> list[str]:
    """The focuses for a turn: the question's own; else those of the turn before; else the default."""
    return classify(message) or [f for f in FOCUS if f in previous] or list(DEFAULT_FOCUS)


def _match(pack: Pack, want: list[str]) -> list[str]:
    """Record IDs the patterns name: exact IDs must exist; a prefix (ending with "." or ":") may match none."""
    out = []
    for w in want:
        if w.endswith((".", ":")):
            out += [i for i in pack.records if i.startswith(w)]
        elif w in pack.records:
            out.append(w)
        else:
            raise SelectionError("context_incomplete", f"record {w}")
    return out


def build_view(pack: Pack, focus: list[str], dropped: list[str], depth_pref: str, with_tests: bool = True) -> View:
    f = [FOCUS[x] for x in focus]
    purposes = [p for p in PURPOSES if p in CORE_PURPOSES or any(p in x["purposes"] for x in f)]
    depth = depth_pref if depth_pref in DEPTH else max((PURPOSES[p]["depth"] for p in purposes), key=DEPTHS.index)
    required = []
    for p in purposes:
        for fid, minimum in PURPOSES[p]["require"]:
            if fid not in pack.findings:
                raise SelectionError("context_incomplete", f"finding {fid}")
            if depth_after(depth, minimum) and pack.findings[fid]["fires"]:
                required.append(fid)
    wanted = list(dict.fromkeys([*CORE_FINDINGS, *(x for s in f for x in (*s["find"], *s.get("brief", []))), *required]))
    if absent := [x for x in wanted if x not in pack.findings]:
        raise SelectionError("context_incomplete", f"finding {', '.join(absent)}")
    findings = {i: x for i, x in pack.findings.items() if i in wanted}
    ids = set(_match(pack, [*CORE_RECORDS, *(r for s in f for r in s["rec"])]))
    asked = {*CORE_FINDINGS, *(x for s in f for x in s["find"])}
    lean = tuple(i for i in findings if i in LEAN and i not in asked)  # in the turn only because a purpose requires them
    for x in findings.values():
        for i in LEAN[x["id"]] if x["id"] in lean else x["inputs"]:
            if i in pack.records:
                ids.add(i)
            elif not i.startswith(HIDDEN):
                raise SelectionError("context_incomplete", f"{x['id']} depends on {i}")
    records = {i: r for i, r in pack.records.items() if i in ids}
    hyps = {i.removeprefix("hyp.") for i, r in records.items() if r["kind"] == "hypothesis"}
    pairs = [p for p in pack.pairs if set(p["pair"]) <= hyps]
    tests = dict(pack.tests) if with_tests and "possible_causes" in purposes else {}
    scope = f"SCOPE: evidence loaded for this question: {', '.join(focus)}. Evidence on other topics is not loaded for this turn."
    if dropped:
        scope += f" Not loaded because the question is too broad for one answer: {', '.join(dropped)}."
    sections = {"head": pack.head, "scope": scope, "records": render_records(records), "findings": render_findings(findings.values()),
                "tests": render_tests(tests, pairs)}
    return View(tuple(focus), tuple(dropped), tuple(purposes), depth, lean, records, findings, tests, pairs,
                make_schema(records, findings, tests, purposes), sections)


def views(pack: Pack, message: str, previous: list[str] | tuple, depth_pref: str):
    """The question's slice, then leaner ones. Each step first drops the candidate-test table, which only serves the optional follow-up
    question, and only then the lowest-priority focus, whole."""
    focus = plan(message, previous)
    for n in range(len(focus), 0, -1):
        full = build_view(pack, focus[:n], focus[n:], depth_pref)
        yield full
        if full.tests:
            yield build_view(pack, focus[:n], focus[n:], depth_pref, with_tests=False)
