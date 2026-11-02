"""Case files as the service loads them, the compact context pack sent with each turn, and the response schema.

Case files are integrity-checked at load. Chat is retrospective only (owner decision D8), so packs are built from retro
files; as-of files are served to the Brief but never to the model.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from backend.syrus_app.policy import PURPOSES
from backend.syrus_app.records import KINDS, ROOT, check_case

CASES_DIR = ROOT / "docs" / "syrus" / "cases"
HIDDEN = ("group.",)  # feature lists carry no evidence a claim may cite; definitions are not loaded
HYPOTHESES = ROOT / "configs" / "syrus_hypotheses.yaml"
PURPOSE_IDS = list(PURPOSES)
REFUSAL_IDS = ["diagnosis_request", "causal_claim", "prediction", "energy_or_cost", "simulated_transfer", "missing_document", "missing_data",
               "operator_claim", "instruction_override", "out_of_scope"]


@dataclass(frozen=True)
class Pack:
    case_id: str
    sha256: str
    records: dict       # id -> record, the ones a claim may cite
    findings: dict      # id -> finding
    tests: dict         # id -> test
    hyp_patterns: dict  # candidate id -> patterns for its English and Vietnamese labels
    text: str           # rendered context
    schema: dict        # JSON schema for the model's structured output


def label_patterns(hyp_cfg: dict) -> dict:
    """A candidate's label in either language, with spaces and hyphens interchangeable."""
    def rx(label):
        return re.compile(r"[\s-]+".join(re.escape(w) for w in re.split(r"[\s-]+", label.lower())), re.IGNORECASE)
    return {h["id"]: [rx(h["label"]["en"]), rx(h["label"]["vi"])] for h in hyp_cfg["hypotheses"]}


def _dump(x) -> str:
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def _display(d) -> str:
    return d if isinstance(d, str) else _dump(d)


def _span(ids: list[str]) -> str:
    """Name a run of IDs by prefix: cf.a, cf.b -> cf.*"""
    prefixes = {i.split(".")[0] for i in ids}
    return ", ".join(f"{p}.*" if sum(i.split(".")[0] == p for i in ids) > 1 else next(i for i in ids if i.split(".")[0] == p) for p in sorted(prefixes))


def render(case: dict, records: dict) -> str:
    """Record caveats appear once each in a legend instead of on every row."""
    legend = {}
    for r in records.values():
        if r["caveat"]:
            legend.setdefault(r["caveat"], []).append(r["id"])
    notes = "\n".join(f"- {c} (applies to {_span(ids)})" for c, ids in legend.items())
    rec = "\n".join(f"{r['id']} | {r['kind']} | {_display(r['display'])}" for r in records.values())
    fnd = "\n".join(f"{f['id']} | fires={str(f['fires']).lower()} | {f['text']}" for f in case["findings"])
    tst = "\n".join(f"{t['id']} | compares {_dump(t['compares'])} | {t['maturity']} | {t['result']} | needs: "
                    + "; ".join(f"{o['availability']}: {o['text']['en']}" for o in t["observations"]) for t in case["tests"])
    lim = "\n".join(f"{' / '.join(p['pair'])}: {p['text']['en']}" for p in case["pair_limits"])
    return (f"CASE {case['case_id']} (retrospective review). Alert {case['alert_time']} dataset local time.\n\nRECORDS (id | kind | display)\n{rec}\n\n"
            f"NOTES ON RECORDS\n{notes}\n\nFINDINGS (id | fires | text)\n{fnd}\n\nTESTS (id | compares | maturity | result | needs)\n{tst}\n\n"
            f"PAIRS THAT CANNOT BE TOLD APART WITH THIS DATA\n{lim}")


def make_schema(records: dict, findings: dict, tests: dict) -> dict:
    """Strict structured-output schema whose enums hold only this case's IDs."""
    ids = lambda xs: {"type": "array", "items": {"type": "string", "enum": list(xs)}}  # noqa: E731
    hyps = [i for i, r in records.items() if r["kind"] == "hypothesis"]
    gaps = [i for i, r in records.items() if r["kind"] == "unavailable"]
    unit = {"type": "object", "additionalProperties": False, "required": ["text", "kind", "record_ids", "finding_ids"],
            "properties": {"text": {"type": "string"}, "kind": {"type": "string", "enum": [*sorted(KINDS - {"simulated"}), "background"]},
                           "record_ids": ids(records), "finding_ids": ids(findings)}}
    props = {"lang": {"type": "string", "enum": ["en", "vi"]}, "purposes": ids(PURPOSE_IDS), "units": {"type": "array", "items": unit},
             "alternatives": ids(hyps), "gaps": ids(gaps), "next_test": {"type": ["string", "null"], "enum": [*tests, None]},
             "refusal": {"type": ["string", "null"], "enum": [*REFUSAL_IDS, None]}, "operator_statement": {"type": "boolean"}}
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


class CaseStore:
    """Every case's files, checked at load; a tampered or inconsistent file stops the service from starting."""

    def __init__(self, root: Path = CASES_DIR):
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text("utf-8"))
        self.files = {}
        for c in self.manifest["cases"]:
            for mode, f in c["files"].items():
                case = json.loads((root / f["path"]).read_text("utf-8"))
                check_case(case)
                if case["sha256"] != f["sha256"] or case["case_id"] != c["case_id"] or case["mode"] != mode:
                    raise ValueError(f"{f['path']} does not match the manifest")
                self.files[c["case_id"], mode] = case
        self.patterns = label_patterns(yaml.safe_load(HYPOTHESES.read_text("utf-8")))
        self.packs = {}

    def get(self, case_id: str, mode: str) -> dict | None:
        return self.files.get((case_id, mode))

    def pack(self, case_id: str) -> Pack:
        if case_id not in self.packs:
            case = self.files[case_id, "retro"]
            recs = {r["id"]: r for r in case["records"] if not r["id"].startswith(HIDDEN)}
            finds = {f["id"]: f for f in case["findings"]}
            tests = {t["id"]: t for t in case["tests"]}
            self.packs[case_id] = Pack(case_id, case["sha256"], recs, finds, tests, self.patterns, render(case, recs), make_schema(recs, finds, tests))
        return self.packs[case_id]
