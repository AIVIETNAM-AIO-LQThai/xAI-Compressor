from __future__ import annotations

import hashlib
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/cobra_source_lock.yaml"

def _load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise TypeError("CoBra source-lock config must be a mapping.")
    return value

def _fetch_record(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent":"AeroXAI-CoBra-source-lock/1.0"})
    with urllib.request.urlopen(req, timeout=60) as response: payload = json.load(response)
    if not isinstance(payload, dict): raise TypeError("Zenodo record response is not a JSON object.")
    return payload

def _checksum_hex(checksum: str) -> str:
    prefix, sep, value = checksum.partition(":")
    if sep != ":" or prefix.lower() != "md5": raise ValueError(f"Expected Zenodo md5 checksum, got: {checksum}")
    return value.lower()

def _hash_file(path: Path, *, chunk_bytes: int) -> tuple[str, str, int]:
    md5, sha256, size = hashlib.md5(), hashlib.sha256(), 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk: break
            md5.update(chunk); sha256.update(chunk); size += len(chunk)
    return md5.hexdigest(), sha256.hexdigest(), size

def _download(url: str, destination: Path, *, chunk_bytes: int) -> None:
    temp = destination.with_suffix(destination.suffix + ".part")
    if temp.exists(): temp.unlink()
    req = urllib.request.Request(url, headers={"User-Agent":"AeroXAI-CoBra-source-lock/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as response, temp.open("wb") as output:
            while True:
                chunk = response.read(chunk_bytes)
                if not chunk: break
                output.write(chunk)
        temp.replace(destination)
    except Exception:
        if temp.exists(): temp.unlink()
        raise

def main() -> None:
    config = _load_config(); study = config["study"]
    if subprocess.run(["git","merge-base","--is-ancestor",str(study["selected_dataset_commit"]),"HEAD"], cwd=ROOT, check=False).returncode != 0:
        raise RuntimeError("Selected CoBra dataset commit is not an ancestor of HEAD.")
    if any(bool(v) for v in config["guardrails"].values()): raise RuntimeError("All source-lock guardrails must remain false.")
    z = config["zenodo"]; record = _fetch_record(str(z["api_url"]))
    if int(record["id"]) != int(z["record_id"]): raise RuntimeError("Zenodo record id differs.")
    metadata = record.get("metadata", {})
    doi = str(record.get("doi") or metadata.get("doi"))
    version = str(metadata.get("version"))
    if doi != str(z["doi"]): raise RuntimeError(f"Zenodo DOI changed: {doi}")
    if version != str(z["version"]): raise RuntimeError(f"Zenodo version changed: {version}")
    entries = record.get("files")
    if not isinstance(entries, list): raise TypeError("Zenodo files is not a list.")
    remote = {str(e["key"]): e for e in entries}
    primary = {str(k):str(v).lower() for k,v in config["primary_files"].items()}
    aux = {str(k):str(v).lower() for k,v in config["published_auxiliary_files"].items()}
    expected = primary | aux
    if set(remote) != set(expected):
        raise RuntimeError(f"Published inventory differs. missing={sorted(set(expected)-set(remote))}, extra={sorted(set(remote)-set(expected))}")
    inventory = {}
    for name, expected_md5 in expected.items():
        observed = _checksum_hex(str(remote[name]["checksum"]))
        if observed != expected_md5: raise RuntimeError(f"Publisher MD5 changed for {name}")
        inventory[name] = {"publisher_md5": observed, "remote_size_bytes": int(remote[name]["size"]), "downloaded_primary_source": name in primary}
    destination = ROOT / config["download"]["destination"]; destination.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / config["download"]["manifest_output"]
    if manifest.exists(): raise RuntimeError(f"Manifest already exists: {manifest}")
    chunk_bytes = int(config["download"]["chunk_bytes"]); local = {}
    for i, (name, expected_md5) in enumerate(primary.items(), start=1):
        path = destination / name; entry = remote[name]
        if path.exists():
            md5, sha256, size = _hash_file(path, chunk_bytes=chunk_bytes); action = "verified_existing"
            if md5 != expected_md5: raise RuntimeError(f"Existing local file has wrong MD5: {name}")
        else:
            links = entry.get("links", {}); url = (links.get("content") or links.get("self")) if isinstance(links, dict) else None
            if not url:
                url = f"https://zenodo.org/records/{z['record_id']}/files/{urllib.parse.quote(name)}?download=1"
            print(f"[{i}/{len(primary)}] downloading {name}", flush=True)
            _download(str(url), path, chunk_bytes=chunk_bytes)
            md5, sha256, size = _hash_file(path, chunk_bytes=chunk_bytes); action = "downloaded"
            if md5 != expected_md5:
                path.unlink(missing_ok=True); raise RuntimeError(f"Downloaded file failed MD5: {name}")
        if size != int(entry["size"]): raise RuntimeError(f"Byte size mismatch for {name}")
        local[name] = {"action":action,"path":str(path.relative_to(ROOT)),"size_bytes":size,"md5":md5,"sha256":sha256}
        print(f"[{i}/{len(primary)}] locked {name} sha256={sha256}", flush=True)
    payload = {"schema_version":"aeroxai.cobra_source_lock.v1","study":study["name"],"evidence_role":study["evidence_role"],"selected_dataset_commit":study["selected_dataset_commit"],"zenodo":{"record_id":int(z["record_id"]),"doi":doi,"version":version,"api_url":z["api_url"]},"source_lock_only":True,"source_archives_unzipped":False,"sensor_values_opened":False,"outcome_metrics_computed":False,"published_inventory":inventory,"primary_local_files":local,"guardrails":config["guardrails"]}
    manifest.parent.mkdir(parents=True, exist_ok=True); manifest.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status":"COBRA_SOURCE_LOCK_COMPLETE","record_id":int(z["record_id"]),"doi":doi,"version":version,"primary_files_locked":len(local),"published_files_verified":len(inventory),"source_archives_unzipped":False,"sensor_values_opened":False,"outcome_metrics_computed":False,"manifest":str(manifest.relative_to(ROOT))}, indent=2))

if __name__ == "__main__": main()
