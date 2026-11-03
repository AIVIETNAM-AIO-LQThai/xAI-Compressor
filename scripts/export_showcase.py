"""Export the public showcase: a static site in which every read-only view is precomputed. No server, no key, no simulation at view time.

    cd frontend; npm run build:showcase; cd ..
    python -m scripts.export_showcase [--out outputs/showcase]

Every answer is fetched through the real application (local mode, Syrus without a key), so the files are what the server would return.
Recorded days are written one file per day without gap rows; the showcase client joins days and inserts gaps with the server's rule. The
lab is exported as a few fixed, labelled replays. The export stops if any written file contains the server's key, a credential name, or
anything from `.env`.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from backend.access import resolve
from backend.app.explorer import ROOT, Explorer
from backend.app.lab import Lab, SensorFault, TwinRun
from backend.syrus_app.settings import load_settings
from backend.web import create_web

SHOWCASE_BUILD = ROOT / "frontend" / "dist-showcase"
FEATURES = {"syrus_chat": False, "lab": "replay", "twin": False, "optimization": False}
HEADERS = """/*
  X-Content-Type-Options: nosniff
  Referrer-Policy: no-referrer
  X-Frame-Options: DENY
  Permissions-Policy: camera=(), microphone=(), geolocation=()
  Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'
/data/*
  Cache-Control: public, max-age=3600
"""


def safe(t: str) -> str:
    return t.replace(" ", "T").replace(":", "-")[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description="Export the static AeroXAI showcase.")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "showcase")
    args = ap.parse_args()
    if not (SHOWCASE_BUILD / "index.html").exists():
        raise SystemExit("Build the showcase frontend first: cd frontend; npm run build:showcase")
    out, data = args.out, args.out / "data"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(SHOWCASE_BUILD, out)

    def write(rel: str, obj) -> None:
        p = data / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False), "utf-8")

    # The real application, local mode, Syrus without a key or an adapter: no provider can be reached from here.
    app = create_web(load_settings({}), None, dist=out / "none", access=resolve({"AEROXAI_MODE": "local"}))
    c = TestClient(app)

    def get(path: str):
        r = c.get(path)
        r.raise_for_status()
        return r.json()

    cases = get("/api/cases")
    write("api/cases.json", cases)
    for case in cases["cases"]:
        for mode in ("asof", "retro"):
            write(f"api/cases/{case['case_id']}.{mode}.json", get(f"/api/cases/{case['case_id']}?mode={mode}"))
    write("api/health.json", {"status": "offline", "reason": "guests_off", "model": None, "mock": False, "cases": len(cases["cases"]), "manifest_sha256": cases["sha256"]})
    for name in ("summary", "detection", "proof-manifest", "proof-bundle"):
        write(f"evidence/{name}.json", get(f"/evidence/{name}"))
    verify = c.post("/evidence/proof-bundle/verify", json=get("/evidence/proof-bundle"))
    verify.raise_for_status()
    write("evidence/proof-bundle/verify.json", verify.json())

    write("explorer/overview.json", get("/explorer/overview"))
    episodes = get("/explorer/episodes")
    write("explorer/episodes.json", episodes)
    for e in episodes["episodes"]:
        write(f"explorer/episode/{safe(e['start'])}.json", get(f"/explorer/episode?start={e['start'][:16]}"))

    ex = Explorer()
    detector = ex.detector()
    for f in sorted((ex.root / "raw").glob("*.parquet")):
        day = pd.Timestamp(f.stem)
        write(f"explorer/range/{f.stem}.json", ex.range(day, day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1), gaps=False))
        write(f"explorer/raw/{f.stem}.json", ex.raw(day, day + pd.Timedelta(days=1), gaps=False))
        bins = detector.loc[day:day + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
        write(f"explorer/bins/{f.stem}.json", {str(t): ex.bin(t) for t in bins.index})

    lab = Lab(ex)
    bg = lab.backgrounds()[0]["start"]
    fault = {"start": bg, "hours": 12, "onset_hours": 4}
    replays = [
        ("tp3-offset-12", "fault", "Offset of 12 training SDs on tp3", "A pressure channel shifted from 4 h into a quiet recorded window", [lab.sensor_fault(SensorFault(**fault, fault="offset", channel="tp3", severity=12))]),
        ("dv-noise-5", "fault", "Added noise, 5 training SDs, on dv_pressure", "Seeded Gaussian noise from 4 h, seed 0", [lab.sensor_fault(SensorFault(**fault, fault="noise", channel="dv_pressure", severity=5))]),
        ("oil-drift-8", "fault", "Drift of 8 training SDs on oil_temperature", "A slow ramp the detector does not alert on", [lab.sensor_fault(SensorFault(**fault, fault="drift", channel="oil_temperature", severity=8))]),
        ("dropout-60", "fault", "Dropout of 60% of samples", "Bins lose coverage and leave the detector's view", [lab.sensor_fault(SensorFault(**fault, fault="dropout", severity=0.6))]),
        ("leak-vs-demand", "twin", "Leak and demand step of the same size", "Generic plant: 0.02 kg/s from minute 20, both causes", [lab.twin(TwinRun(scenario="leak_onset")), lab.twin(TwinRun(scenario="demand_step"))]),
        ("twin-baseline", "twin", "Generic plant without a change", "Baseline cycles under the controller", [lab.twin(TwinRun(scenario="none"))]),
    ]
    write("lab/replays.json", [{"id": i, "kind": k, "title": t, "note": n} for i, k, t, n, _ in replays])
    for i, k, t, n, runs in replays:
        write(f"lab/replays/{i}.json", {"id": i, "kind": k, "title": t, "note": n, "runs": runs})

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    write("config.json", {"mode": "showcase", "features": FEATURES, "exported_at": datetime.now(UTC).isoformat(timespec="seconds"), "source_commit": commit})
    (out / "_headers").write_text(HEADERS, "utf-8")

    # Nothing secret may leave: the server's key (if this shell has one), credential names, or any value from .env.
    needles = [b"SYRUS_LLM_API_KEY", b"SYRUS_SESSION_SECRET", b"AEROXAI_PREVIEW", b"gsk_"]
    needles += [v.encode() for k, v in os.environ.items() if k.startswith(("SYRUS_", "AEROXAI_", "GROQ")) and len(v) >= 12]
    files = [p for p in out.rglob("*") if p.is_file()]
    leaks = [str(p.relative_to(out)) for p in files if any(n in p.read_bytes() for n in needles)]
    if leaks:
        shutil.rmtree(out)
        raise SystemExit(f"Export removed: secret-looking content in {leaks[:5]}")
    manifest = {"files": len(files), "bytes": sum(p.stat().st_size for p in files),
                "sha256": {str(p.relative_to(out)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (out / "showcase-manifest.json").write_text(json.dumps(manifest, indent=1), "utf-8")
    print(f"Showcase written to {out}: {manifest['files']} files, {manifest['bytes'] / 1e6:.1f} MB. Nothing was uploaded.")


if __name__ == "__main__":
    main()
