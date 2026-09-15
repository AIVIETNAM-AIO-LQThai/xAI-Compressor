from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from scripts.lock_cobra_source import _checksum_hex, _hash_file


def _config() -> dict:
    return yaml.safe_load(Path("configs/cobra_source_lock.yaml").read_text(encoding="utf-8"))

def test_source_lock_is_outcome_blind() -> None:
    c=_config(); assert c["study"]["outcome_metrics_allowed"] is False; assert not any(bool(v) for v in c["guardrails"].values())

def test_primary_archive_has_23_days_plus_overview() -> None:
    c=_config(); names=set(c["primary_files"]); days={n for n in names if n.startswith("CoBra20") and n.endswith(".zip")}; assert len(days)==23; assert "overview_experiments.csv" in names; assert len(names)==24

def test_expected_published_inventory_has_30_files() -> None:
    c=_config(); assert len(set(c["primary_files"]) | set(c["published_auxiliary_files"])) == 30

def test_checksum_parser() -> None:
    assert _checksum_hex("md5:78598e4ed16dd48f64cf5670110f66f7") == "78598e4ed16dd48f64cf5670110f66f7"

def test_hash_file(tmp_path: Path) -> None:
    p=tmp_path/"x.bin"; p.write_bytes(b"cobra-source-lock"); md5,sha,size=_hash_file(p,chunk_bytes=4); assert md5==hashlib.md5(b"cobra-source-lock").hexdigest(); assert sha==hashlib.sha256(b"cobra-source-lock").hexdigest(); assert size==len(b"cobra-source-lock")
