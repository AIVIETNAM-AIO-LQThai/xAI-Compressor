from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ml.data.rcsd1yd import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rcsd1yd_external_evaluation.yaml"
MANIFEST_PATH = ROOT / "docs/research/rcsd1yd_ingestion_manifest.json"
FREEZE_PATH = ROOT / "docs/research/rcsd1yd_adaptation_freeze.json"
OUTPUT_PATH = ROOT / "docs/research/rcsd1yd_pre_evaluation_preflight.json"

PREREGISTRATION_COMMIT = "7dc7bd2d9a44758075dab286cc3d674949a876bd"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return payload


def main() -> None:
    config = _load_yaml(CONFIG_PATH)
    manifest = _load_json(MANIFEST_PATH)
    freeze = _load_json(FREEZE_PATH)

    if freeze["preregistration_commit"] != PREREGISTRATION_COMMIT:
        raise RuntimeError("Adaptation freeze preregistration identity mismatch.")

    data_use = freeze["data_use"]

    forbidden_true = [
        "evaluation_parquet_opened_by_adaptation",
        "evaluation_sensor_summary_computed",
        "evaluation_tcn_metrics_computed",
        "evaluation_router_metrics_computed",
        "evaluation_energy_metrics_computed",
    ]

    if any(bool(data_use[key]) for key in forbidden_true):
        raise RuntimeError("Pre-EVALUATION blinding guard failed.")

    if bool(freeze["evaluation_metrics_computed"]):
        raise RuntimeError("Evaluation metrics already appear in adaptation freeze.")

    expected_sensors = [str(value) for value in config["sensor_columns"]]

    if freeze["source"]["sensor_columns"] != expected_sensors:
        raise RuntimeError("Frozen RCSD-1YD sensor identity changed.")

    if int(freeze["source"]["sensor_count"]) != 25:
        raise RuntimeError("Frozen RCSD-1YD sensor count changed.")

    checkpoint_path = ROOT / freeze["tcn"]["checkpoint_path"]

    if not checkpoint_path.exists():
        raise FileNotFoundError("Frozen RCSD-1YD TCN checkpoint is missing.")

    observed_checkpoint_sha = sha256_file(checkpoint_path)

    if observed_checkpoint_sha != freeze["tcn"]["checkpoint_sha256"]:
        raise RuntimeError("RCSD-1YD TCN checkpoint hash mismatch.")

    for split_name in ("train", "calibration", "evaluation"):
        split_path = ROOT / manifest[split_name]["path"]
        observed = sha256_file(split_path)
        expected = manifest[split_name]["sha256"]
        if observed != expected:
            raise RuntimeError(
                f"{split_name} parquet changed after frozen ingestion."
            )

    high_threshold = float(
        freeze["tcn"]["calibration_high_evidence_threshold_q90"]
    )
    alert_threshold = float(
        freeze["tcn"]["calibration_alert_evidence_threshold_q995"]
    )

    if not np.isfinite(high_threshold) or not np.isfinite(alert_threshold):
        raise RuntimeError("Frozen CAL TCN threshold is non-finite.")

    if alert_threshold < high_threshold:
        raise RuntimeError("CAL q0.995 threshold is below q0.90 threshold.")

    frozen_router = freeze["frozen_router"]

    if float(frozen_router["C"]) != float(config["frozen_router"]["C"]):
        raise RuntimeError("Frozen router C mismatch.")
    if frozen_router["class_weight"] != config["frozen_router"]["class_weight"]:
        raise RuntimeError("Frozen router class_weight mismatch.")
    if float(frozen_router["probability_threshold"]) != float(
        config["frozen_router"]["probability_threshold"]
    ):
        raise RuntimeError("Frozen router probability threshold mismatch.")

    payload = {
        "schema_version": "aeroxai.rcsd1yd_pre_evaluation_preflight.v1",
        "study": config["study"]["name"],
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "adaptation_freeze_file": str(FREEZE_PATH.relative_to(ROOT)),
        "adaptation_freeze_sha256": sha256_file(FREEZE_PATH),
        "source_md5": manifest["source"]["md5"],
        "schema_validation_passed": True,
        "split_hashes_verified": True,
        "tcn_checkpoint_sha256": observed_checkpoint_sha,
        "calibration_high_evidence_threshold_q90": high_threshold,
        "calibration_alert_evidence_threshold_q995": alert_threshold,
        "frozen_router_identity_sha256": frozen_router["identity_sha256"],
        "evaluation": {
            "rows": freeze["splits"]["evaluation"]["rows"],
            "frozen_calendar_bounds": freeze["splits"]["evaluation"][
                "frozen_calendar_bounds"
            ],
            "metrics_computed_before_freeze": False,
            "sensor_summary_computed_before_freeze": False,
        },
        "energy_protocol": config["energy_measurement"],
        "success_criteria": config["success_criteria"],
        "ready_to_commit_adaptation_freeze": True,
        "ready_for_one_shot_evaluation_after_commit": True,
    }

    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "schema_validation_passed": True,
                "split_hashes_verified": True,
                "tcn_checkpoint_sha256": observed_checkpoint_sha,
                "calibration_q90": high_threshold,
                "calibration_q995": alert_threshold,
                "evaluation_rows": payload["evaluation"]["rows"],
                "evaluation_metrics_computed_before_freeze": False,
                "ready_to_commit_adaptation_freeze": True,
                "ready_for_one_shot_evaluation_after_commit": True,
                "output": str(OUTPUT_PATH.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
