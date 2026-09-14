from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from ml.data.rcsd1yd import sha256_file
from ml.energy.rcsd1yd_evaluation import (
    ROUTE_NAMES,
    build_frozen_routes,
    build_tcn_model,
    build_temporal_batch_from_checkpoint,
    evidence_stage_status,
    route_evidence_metrics,
    score_tcn_batch,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY_CONFIG_PATH = ROOT / "configs/rcsd1yd_external_evaluation.yaml"
EXEC_CONFIG_PATH = ROOT / "configs/rcsd1yd_one_shot_execution.yaml"
HARNESS_PREFLIGHT_PATH = (
    ROOT / "docs/research/rcsd1yd_one_shot_execution_preflight.json"
)


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
    harness_preflight = _json(HARNESS_PREFLIGHT_PATH)

    if harness_preflight["ready_to_open_evaluation_after_harness_commit"] is not True:
        raise RuntimeError("Execution harness was not cleared for EVALUATION.")

    frozen = execution["frozen_artifacts"]
    freeze_path = ROOT / frozen["adaptation_freeze_file"]
    checkpoint_path = ROOT / frozen["tcn_checkpoint_file"]
    evaluation_path = ROOT / frozen["evaluation_parquet"]
    output_path = ROOT / execution["evaluation"]["result_file"]

    if output_path.exists():
        raise RuntimeError(
            "One-shot evaluation result already exists; refusing to overwrite."
        )

    if sha256_file(freeze_path) != str(
        frozen["expected_adaptation_freeze_sha256"]
    ):
        raise RuntimeError("Adaptation freeze hash changed.")
    if sha256_file(checkpoint_path) != str(
        frozen["expected_tcn_checkpoint_sha256"]
    ):
        raise RuntimeError("TCN checkpoint hash changed.")
    if sha256_file(evaluation_path) != str(
        frozen["expected_evaluation_parquet_sha256"]
    ):
        raise RuntimeError("EVALUATION parquet hash changed.")

    freeze = _json(freeze_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    # This is the first model-metric stage that opens EVALUATION.
    evaluation = pd.read_parquet(evaluation_path)
    evaluation.index = pd.DatetimeIndex(
        evaluation.index,
        name="timestamp",
    )

    expected_rows = int(freeze["splits"]["evaluation"]["rows"])
    if len(evaluation) != expected_rows:
        raise RuntimeError("EVALUATION row count changed.")

    expected_bounds = freeze["splits"]["evaluation"]["frozen_calendar_bounds"]
    if evaluation.index[0] != pd.Timestamp(expected_bounds["start"]):
        raise RuntimeError("EVALUATION start boundary changed.")
    if evaluation.index[-1] != pd.Timestamp(expected_bounds["end"]):
        raise RuntimeError("EVALUATION end boundary changed.")

    if evaluation.columns.tolist() != freeze["source"]["sensor_columns"]:
        raise RuntimeError("EVALUATION sensor schema changed.")

    batch = build_temporal_batch_from_checkpoint(
        evaluation,
        checkpoint,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_tcn_model(checkpoint, device=device)

    scores = score_tcn_batch(
        model,
        batch,
        device=device,
        batch_size=int(execution["evaluation"]["tcn_inference_batch_size"]),
    )

    high_threshold = float(
        freeze["tcn"]["calibration_high_evidence_threshold_q90"]
    )
    alert_threshold = float(
        freeze["tcn"]["calibration_alert_evidence_threshold_q995"]
    )

    high = scores >= high_threshold
    alert = scores >= alert_threshold

    context = build_frozen_routes(
        frame=evaluation,
        temporal_batch=batch,
        adaptation_freeze=freeze,
    )

    systems = {}
    for name in ROUTE_NAMES:
        systems[name] = route_evidence_metrics(
            context.routes[name],
            high_evidence=high,
            alert_evidence=alert,
        )

    criteria = study["success_criteria"]
    evidence_status = evidence_stage_status(
        systems["frozen_evidence_aware_router"],
        minimum_high_coverage=float(
            criteria["minimum_high_tcn_evidence_bin_coverage"]
        ),
        minimum_alert_coverage=float(
            criteria["minimum_tcn_alert_bin_coverage"]
        ),
    )

    score_series = pd.Series(
        scores,
        index=batch.target_index,
        dtype=float,
    )

    payload = {
        "schema_version": "aeroxai.rcsd1yd_one_shot_evaluation.v1",
        "study": study["study"]["name"],
        "evidence_role": "genuinely_new_chronological_external_evaluation",
        "adaptation_freeze_commit": execution["study"][
            "adaptation_freeze_commit"
        ],
        "data_use": {
            "evaluation_parquet_opened": True,
            "evaluation_used_for_training": False,
            "evaluation_used_for_calibration": False,
            "evaluation_derived_tuning_performed": False,
        },
        "evaluation": {
            "source_rows": len(evaluation),
            "valid_tcn_rows": len(batch.inputs),
            "target_start": str(batch.target_index[0]),
            "target_end": str(batch.target_index[-1]),
            "tcn_score_min": float(score_series.min()),
            "tcn_score_median": float(score_series.median()),
            "tcn_score_p90": float(
                score_series.quantile(0.90, interpolation="linear")
            ),
            "tcn_score_p995": float(
                score_series.quantile(0.995, interpolation="linear")
            ),
            "tcn_score_max": float(score_series.max()),
        },
        "frozen_teacher": {
            "calibration_high_evidence_threshold_q90": high_threshold,
            "calibration_alert_evidence_threshold_q995": alert_threshold,
            "evaluation_high_evidence_bins": int(high.sum()),
            "evaluation_alert_evidence_bins": int(alert.sum()),
        },
        "systems": systems,
        "evidence_stage_status": evidence_status,
        "energy_stage_status": "PENDING",
        "final_external_status": "PENDING_ENERGY",
        "success_criteria": criteria,
        "zero_denominator_rule": study["evaluation_adequacy"],
        "interpretation_guardrails": study["interpretation"],
    }

    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "evaluation_source_rows": len(evaluation),
                "valid_tcn_rows": len(batch.inputs),
                "high_evidence_bins": int(high.sum()),
                "alert_evidence_bins": int(alert.sum()),
                "systems": systems,
                "evidence_stage_status": evidence_status,
                "energy_stage_status": "PENDING",
                "output": str(output_path.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
