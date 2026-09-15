from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rcsd1yd_router_failure_analysis.yaml"

PREREGISTRATION_COMMIT = "19aaad1611a0725501765fe4c39573e096f24c6a"


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
            PREREGISTRATION_COMMIT,
            "HEAD",
        ],
        cwd=ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            "Failure-analysis preregistration is not an ancestor of HEAD."
        )

    if config["study"]["evidence_class"] != "POST_HOC_EXPLORATORY":
        raise RuntimeError("Failure-analysis evidence class changed.")
    if config["study"]["causal_claim"] is not False:
        raise RuntimeError("Failure analysis must retain causal_claim=false.")
    if config["study"]["closed_result_may_change"] is not False:
        raise RuntimeError("Closed external result may not change.")

    frozen = config["frozen_inputs"]
    external = _json(ROOT / frozen["external_result"])
    one_shot = _json(ROOT / frozen["one_shot_result"])
    adaptation = _json(ROOT / frozen["adaptation_freeze"])

    if external["final_external_status"] != "EXTERNAL_TRANSPORT_FAIL":
        raise RuntimeError("Closed external classification changed.")
    if (
        one_shot["evidence_stage_status"]
        != "EVIDENCE_COVERAGE_FAIL"
    ):
        raise RuntimeError("Closed evidence-stage classification changed.")
    if adaptation["evaluation_metrics_computed"] is not False:
        raise RuntimeError(
            "Adaptation freeze provenance changed unexpectedly."
        )

    for key in (
        "evaluation_parquet",
        "calibration_parquet",
        "tcn_checkpoint",
    ):
        if not (ROOT / frozen[key]).exists():
            raise FileNotFoundError(
                f"Required frozen local artifact is missing: {frozen[key]}"
            )

    outputs = config["outputs"]
    if (ROOT / outputs["json"]).exists():
        raise RuntimeError(
            "Failure-analysis JSON already exists; refusing overwrite."
        )
    if (ROOT / outputs["markdown"]).exists():
        raise RuntimeError(
            "Failure-analysis Markdown already exists; refusing overwrite."
        )

    forbidden = config["forbidden"]
    if not all(bool(value) for value in forbidden.values()):
        raise RuntimeError("A frozen forbidden-operation guard was disabled.")

    print(
        json.dumps(
            {
                "preregistration_commit": PREREGISTRATION_COMMIT,
                "evidence_class": config["study"]["evidence_class"],
                "causal_claim": config["study"]["causal_claim"],
                "closed_external_status": external[
                    "final_external_status"
                ],
                "closed_result_may_change": False,
                "new_model_fitting_allowed": False,
                "threshold_sweep_allowed": False,
                "new_feature_search_allowed": False,
                "diagnostic_outputs_absent": True,
                "ready_to_freeze_implementation": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
