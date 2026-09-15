from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cobra_schema_chronology.yaml"
PROTOCOL_COMMIT = "d4a4e794a35a5a7716c50ed2a540931985a45b2a"


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return value


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return value


def main() -> None:
    config = _yaml(CONFIG_PATH)

    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            PROTOCOL_COMMIT,
            "HEAD",
        ],
        cwd=ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            "CoBra schema/chronology protocol is not an ancestor of HEAD."
        )

    if config["study"]["model_outcome_metrics_allowed"] is not False:
        raise RuntimeError("Model outcome metrics must remain forbidden.")
    if config["study"]["evaluation_sensor_exploration_allowed"] is not False:
        raise RuntimeError("EVALUATION sensor exploration must remain forbidden.")
    if not all(bool(value) for value in config["forbidden"].values()):
        raise RuntimeError("A forbidden-operation guard was disabled.")

    source_manifest = _json(
        ROOT / config["source"]["source_lock_manifest"]
    )
    if source_manifest["source_archives_unzipped"] is not False:
        raise RuntimeError("Source-lock provenance says archives were unzipped.")
    if source_manifest["sensor_values_opened"] is not False:
        raise RuntimeError("Source-lock provenance says sensor values were opened.")
    if source_manifest["outcome_metrics_computed"] is not False:
        raise RuntimeError("Source-lock provenance says outcomes were computed.")

    split = config["split"]
    train = list(split["train_days"])
    calibration = list(split["calibration_days"])
    evaluation = list(split["evaluation_days"])

    if len(train) != 14 or len(calibration) != 4 or len(evaluation) != 5:
        raise RuntimeError("Frozen whole-day split cardinalities changed.")

    all_days = train + calibration + evaluation
    if len(set(all_days)) != 23:
        raise RuntimeError("Frozen split does not contain 23 unique test days.")
    if all_days != sorted(all_days):
        raise RuntimeError("Frozen split is not chronological.")
    if split["may_change_after_schema_inspection"] is not False:
        raise RuntimeError("Split mutability guard changed.")
    if split["may_change_after_model_results"] is not False:
        raise RuntimeError("Split mutability guard changed.")

    output = ROOT / config["output"]["manifest"]
    if output.exists():
        raise RuntimeError(
            f"Schema/chronology manifest already exists: {output}"
        )

    print(
        json.dumps(
            {
                "protocol_commit": PROTOCOL_COMMIT,
                "source_lock_commit": config["study"][
                    "parent_source_lock_commit"
                ],
                "train_days": len(train),
                "calibration_days": len(calibration),
                "evaluation_days": len(evaluation),
                "evaluation_days_frozen": evaluation,
                "model_outcome_metrics_allowed": False,
                "evaluation_sensor_exploration_allowed": False,
                "schema_manifest_absent": True,
                "ready_to_freeze_implementation": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
