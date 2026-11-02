"""Build Syrus case files from frozen artifacts into docs/syrus/cases/.

Writes {case_id}.asof.json, {case_id}.retro.json and manifest.json. Run from the repository root:
    python -m scripts.build_syrus_cases          # write
    python -m scripts.build_syrus_cases --check  # exit 1 if files on disk differ from a fresh build
"""
import argparse
import json
import sys

from backend.syrus_app.cases import build_all
from backend.syrus_app.records import ROOT, Sources, check_case

OUT = ROOT / "docs" / "syrus" / "cases"


def build_files(src: Sources) -> dict[str, dict]:
    built, manifest = build_all(src)
    files = {"manifest.json": manifest}
    for pair in built:
        for case in pair:
            check_case(case)
            files[f"{case['case_id']}.{case['mode']}.json"] = case
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Syrus case files from frozen artifacts.")
    ap.add_argument("--check", action="store_true", help="compare files on disk with a fresh build; write nothing")
    args = ap.parse_args()
    files = build_files(Sources())
    if args.check:
        stale = [n for n, c in files.items() if not (OUT / n).exists() or json.loads((OUT / n).read_text("utf-8")) != c]
        extra = sorted(p.name for p in OUT.glob("*.json") if p.name not in files)
        for n in stale + extra:
            print(f"out of date: {n}")
        sys.exit(1 if stale or extra else 0)
    OUT.mkdir(parents=True, exist_ok=True)
    for n, c in files.items():
        (OUT / n).write_text(json.dumps(c, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(files)} files to {OUT.relative_to(ROOT)}; manifest {files['manifest.json']['sha256'][:12]}")


if __name__ == "__main__":
    main()
