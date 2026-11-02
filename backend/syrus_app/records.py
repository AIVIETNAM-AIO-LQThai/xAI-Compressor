"""Evidence records for Syrus case files: source pointers, content hashes, display formats and schema checks.

A source pointer is {"file", "path", "sha256"}. `path` walks parsed JSON/YAML with `/key` segments and
`[field=value,...]` selectors that must match exactly one list item, so pointers do not depend on list order.
Code sources use {"file", "symbol", "sha256"}.
"""
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "syrus.case.v2"
KINDS = {"measured", "model_evidence", "hypothesis", "dataset_label", "benchmark_stat", "case_status",
         "bridge_status", "simulated", "unavailable"}
RETRO_KINDS = {"dataset_label", "benchmark_stat", "case_status"}
RECORD_KEYS = {"id", "kind", "visibility", "value", "display", "source", "caveat"}
SEGMENT = re.compile(r"^(?P<key>[^\[]*)(?:\[(?P<sel>[^\]]+)\])?$")


def canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(obj) -> str:
    return hashlib.sha256(canon(obj)).hexdigest()


def file_sha(path: Path) -> str:
    """Content hash: canonical JSON for .json, LF-normalised bytes otherwise, so Git line endings do not matter."""
    data = path.read_bytes()
    if path.suffix == ".json":
        return sha(json.loads(data))
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _matches(item, key, want):
    got = item.get(key)
    return got == [want] if isinstance(got, list) else str(got) == want


def resolve(doc, path: str):
    node = doc
    for part in path.strip("/").split("/"):
        m = SEGMENT.match(part)
        if m["key"]:
            node = node[int(m["key"])] if isinstance(node, list) else node[m["key"]]
        if m["sel"]:
            conds = [c.split("=", 1) for c in m["sel"].split(",")]
            hits = [x for x in node if all(_matches(x, k, v) for k, v in conds)]
            if len(hits) != 1:
                raise KeyError(f"{path}: selector [{m['sel']}] matched {len(hits)} items")
            node = hits[0]
    return node


class Sources:
    """Loads each source file once and hands out pointers that carry its content hash."""

    def __init__(self, root: Path = ROOT):
        self.root, self.docs, self.hashes = root, {}, {}

    def load(self, rel: str):
        if rel not in self.docs:
            p = self.root / rel
            text = p.read_text("utf-8")
            self.docs[rel] = json.loads(text) if p.suffix == ".json" else yaml.safe_load(text) if p.suffix == ".yaml" else None
            self.hashes[rel] = file_sha(p)
        return self.docs[rel]

    def get(self, rel: str, path: str):
        """Return (value, pointer)."""
        return resolve(self.load(rel), path), {"file": rel, "path": path, "sha256": self.hashes[rel]}

    def code(self, rel: str, symbol: str):
        self.load(rel)
        return {"file": rel, "symbol": symbol, "sha256": self.hashes[rel]}


def record(id, kind, value, display, source, vis="both", caveat=None, **extra):
    return {"id": id, "kind": kind, "visibility": vis, "value": value, "display": display,
            "source": source, "caveat": caveat, **extra}


def derived(id, value, display, inputs, formula, recs, caveat=None, kind="model_evidence"):
    """A record computed from other records; it inherits their sources and keeps the formula."""
    src = [p for i in inputs for p in recs[i]["source"]]
    vis = "retro" if any(recs[i]["visibility"] == "retro" for i in inputs) else "both"
    return record(id, kind, value, display, src, vis, caveat, derivation={"formula": formula, "inputs": inputs})


# Display formats. Numbers in Syrus answers must trace to these strings or the raw values.
def score(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.1f}"


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f}%"


def signed_pct(x: float) -> str:
    return f"{100 * x:+.1f}%"


def pctile(p: float) -> str:
    return "100" if p == 1 else f"{100 * p:.2f}"


def num(x: float) -> str:
    return f"{x:.3g}"


def minutes(a: str, b: str) -> int:
    return int((datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() // 60)


def hhmm(ts: str) -> str:
    return ts[11:16]


def minute(ts: str) -> str:
    return ts[:16]


def count(n: int) -> str:
    return f"{n:,}"


def listing(items: list[str], word: str = "and") -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + f" {word} " + items[-1]


def check_case(case: dict) -> None:
    """Integrity checks for a case file at the load boundary. Raises ValueError on the first problem."""
    def fail(msg):
        raise ValueError(f"{case.get('case_id')}/{case.get('mode')}: {msg}")

    if case.get("schema") != SCHEMA:
        fail(f"schema {case.get('schema')!r}")
    body = {k: v for k, v in case.items() if k != "sha256"}
    if sha(body) != case.get("sha256"):
        fail("sha256 does not match content")
    mode, recs = case["mode"], case["records"]
    ids = [r["id"] for r in recs]
    if len(ids) != len(set(ids)):
        fail("duplicate record ids")
    known = set(ids)
    for r in recs:
        if set(r) - {"derivation", "definition"} != RECORD_KEYS:
            fail(f"{r['id']}: fields {sorted(r)}")
        if r["kind"] not in KINDS:
            fail(f"{r['id']}: kind {r['kind']}")
        if r["kind"] in RETRO_KINDS and r["visibility"] != "retro":
            fail(f"{r['id']}: {r['kind']} must be retrospective")
        if mode == "asof" and r["visibility"] != "both":
            fail(f"{r['id']}: retrospective record in as-of file")
        for p in r["source"]:
            if not {"file", "sha256"} <= set(p) or (mode == "retro" and not ({"path", "symbol"} & set(p))):
                fail(f"{r['id']}: incomplete source pointer")
        for i in r.get("derivation", {}).get("inputs", []) + ([r["definition"]] if "definition" in r else []):
            if i not in known:
                fail(f"{r['id']}: unknown input {i}")
    for f in case["findings"]:
        if mode == "asof" and f["visibility"] != "both":
            fail(f"{f['id']}: retrospective finding in as-of file")
        if missing := set(f["inputs"]) - known:
            fail(f"{f['id']}: unknown inputs {sorted(missing)}")
    hyps = {r["id"].removeprefix("hyp."): r for r in recs if r["kind"] == "hypothesis"}
    for h, r in hyps.items():
        if r["value"]["status"] != "open":
            fail(f"hyp.{h}: status {r['value']['status']!r}; candidates stay open, results belong to tests")
    for t in case["tests"]:
        if any(set(pair) - set(hyps) for pair in t["compares"]):
            fail(f"{t['id']}: compares an unknown candidate")
        loaded = all(o["availability"] == "loaded" for o in t["observations"])
        if t["result"] != "not_run" and not (t["maturity"] == "reviewed" and t["rule"] and loaded):
            fail(f"{t['id']}: result {t['result']!r} without a reviewed rule over loaded observations")
    for lim in case["pair_limits"]:
        if set(lim["pair"]) - set(hyps):
            fail(f"pair limit {lim['pair']}: unknown candidate")
