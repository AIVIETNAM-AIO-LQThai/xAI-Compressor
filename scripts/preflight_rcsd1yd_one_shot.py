from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from ml.data.rcsd1yd import sha256_file
from ml.energy.hierarchical_inference import seeded_condition_orders
from ml.energy.rcsd1yd_evaluation import ROUTE_NAMES

ROOT = Path(__file__).resolve().parents[1]
STUDY_CONFIG_PATH = ROOT / "configs/rcsd1yd_external_evaluation.yaml"
EXEC_CONFIG_PATH = ROOT / "configs/rcsd1yd_one_shot_execution.yaml"
OUTPUT_PATH = ROOT / "docs/research/rcsd1yd_one_shot_execution_preflight.json"


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
    study = _yaml(STUDY_CONFIG_PATH)
    execution = _yaml(EXEC_CONFIG_PATH)

    adaptation_commit = str(
        execution["study"]["adaptation_freeze_commit"]
    )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", adaptation_commit, "HEAD"],
        cwd=ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            "Frozen RCSD adaptation commit is not an ancestor of HEAD."
        )

    frozen = execution["frozen_artifacts"]
    freeze_path = ROOT / frozen["adaptation_freeze_file"]
    preflight_path = ROOT / frozen["pre_evaluation_preflight_file"]
    checkpoint_path = ROOT / frozen["tcn_checkpoint_file"]
    evaluation_path = ROOT / frozen["evaluation_parquet"]

    if sha256_file(freeze_path) != str(
        frozen["expected_adaptation_freeze_sha256"]
    ):
        raise RuntimeError("Adaptation-freeze file hash mismatch.")

    prior_preflight = _json(preflight_path)
    if prior_preflight["ready_for_one_shot_evaluation_after_commit"] is not True:
        raise RuntimeError("Frozen adaptation was not cleared for evaluation.")

    if sha256_file(checkpoint_path) != str(
        frozen["expected_tcn_checkpoint_sha256"]
    ):
        raise RuntimeError("Frozen RCSD TCN checkpoint hash mismatch.")

    if sha256_file(evaluation_path) != str(
        frozen["expected_evaluation_parquet_sha256"]
    ):
        raise RuntimeError("Frozen EVALUATION parquet hash mismatch.")

    if tuple(study["evaluation"]["systems"]) != ROUTE_NAMES:
        raise RuntimeError("Study systems differ from frozen route names.")

    energy = study["energy_measurement"]
    conditions = [
        str(value)
        for value in execution["energy_execution"]["conditions"]
    ]
    if tuple(conditions) != ROUTE_NAMES:
        raise RuntimeError("Energy conditions differ from frozen systems.")

    repetitions = int(energy["measured_repetitions"])
    condition_orders = seeded_condition_orders(
        conditions,
        repetitions=repetitions,
        seed=int(execution["energy_execution"]["condition_order_seed"]),
    )

    evaluation_result = ROOT / execution["evaluation"]["result_file"]
    final_result = ROOT / execution["energy_execution"]["final_result_file"]
    if evaluation_result.exists() or final_result.exists():
        raise RuntimeError(
            "One-shot result file already exists. "
            "Do not overwrite or rerun without an explicit new protocol."
        )

    payload = {
        "schema_version": "aeroxai.rcsd1yd_one_shot_execution_preflight.v1",
        "study": execution["study"]["name"],
        "adaptation_freeze_commit": adaptation_commit,
        "adaptation_freeze_sha256": sha256_file(freeze_path),
        "tcn_checkpoint_sha256": sha256_file(checkpoint_path),
        "evaluation_parquet_sha256": sha256_file(evaluation_path),
        "evaluation_parquet_opened": False,
        "evaluation_metrics_computed": False,
        "systems": conditions,
        "energy_protocol": {
            **energy,
            "gpu_device_index": int(
                execution["energy_execution"]["gpu_device_index"]
            ),
            "inference_batch_size": int(
                execution["energy_execution"]["inference_batch_size"]
            ),
            "cooldown_seconds": float(
                execution["energy_execution"]["cooldown_seconds"]
            ),
            "condition_order_seed": int(
                execution["energy_execution"]["condition_order_seed"]
            ),
            "condition_orders": condition_orders,
        },
        "zero_denominator_rule": study["evaluation_adequacy"],
        "ready_to_commit_execution_harness": True,
        "ready_to_open_evaluation_after_harness_commit": True,
    }

    OUTPUT_PATH.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "adaptation_freeze_sha256": payload[
                    "adaptation_freeze_sha256"
                ],
                "tcn_checkpoint_sha256": payload[
                    "tcn_checkpoint_sha256"
                ],
                "evaluation_parquet_sha256": payload[
                    "evaluation_parquet_sha256"
                ],
                "evaluation_parquet_opened": False,
                "evaluation_metrics_computed": False,
                "condition_orders": condition_orders,
                "ready_to_commit_execution_harness": True,
                "ready_to_open_evaluation_after_harness_commit": True,
                "output": str(OUTPUT_PATH.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
