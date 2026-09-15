from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cobra_adaptation_evaluation.yaml"

PROTOCOL_COMMIT = (
    "c18473e7a1ce80fafec04202ce0b61ad8378fe54"
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


def _candidate_envelope(
    schema_manifest: dict[str, Any],
) -> tuple[list[str], str]:
    archives = schema_manifest["archives"]

    if len(archives) != 23:
        raise RuntimeError(
            f"Expected 23 frozen CoBra archives, got {len(archives)}."
        )

    common_columns = set(archives[0]["columns"])

    for archive in archives[1:]:
        common_columns &= set(archive["columns"])

    timestamp_columns = {
        archive["timestamp_column"]
        for archive in archives
    }

    candidates = sorted(
        column
        for column in common_columns
        if column not in timestamp_columns
    )

    if not candidates:
        raise RuntimeError(
            "Metadata-only structural compatibility envelope is empty."
        )

    payload = "\n".join(candidates) + "\n"
    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()

    return candidates, digest


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
            "Frozen CoBra adaptation protocol is not an ancestor of HEAD."
        )

    study = config["study"]

    if (
        study[
            "cobra_model_outcomes_allowed_before_router_v2_freeze"
        ]
        is not False
    ):
        raise RuntimeError(
            "CoBra model outcomes must remain forbidden "
            "before Router V2 freeze."
        )

    if (
        study["zero_shot_external_validation_claim_allowed"]
        is not False
    ):
        raise RuntimeError(
            "Zero-shot external-validation claim guard changed."
        )

    alignment = config["schema_alignment"]
    if alignment["align_by_exact_name"] is not True:
        raise RuntimeError(
            "Exact-name schema alignment must remain enabled."
        )
    if alignment["positional_alignment_allowed"] is not False:
        raise RuntimeError(
            "Positional schema alignment must remain forbidden."
        )

    envelope = config["candidate_compatibility_envelope"]

    expected_envelope_rules = {
        "derivation_source": "committed_schema_metadata_only",
        "membership_rule": "intersection_all_23_archives",
        "exclude_timestamp": True,
        "coarse_schema_class_required": False,
        "ordering": "lexicographic_exact_name",
        "serialize_ordered_names": True,
        "sha256_required": True,
        "is_final_model_feature_set": False,
    }

    for key, expected in expected_envelope_rules.items():
        if envelope[key] != expected:
            raise RuntimeError(
                f"Candidate-envelope rule changed: "
                f"{key}={envelope[key]!r}, expected {expected!r}."
            )

    feature_contract = config["final_feature_contract"]

    if feature_contract["may_use_cobra_sensor_values"] is not False:
        raise RuntimeError(
            "Feature contract may not use CoBra sensor values."
        )
    if feature_contract["may_use_cobra_model_performance"] is not False:
        raise RuntimeError(
            "Feature contract may not use CoBra model performance."
        )
    if feature_contract["may_use_evaluation_behavior"] is not False:
        raise RuntimeError(
            "Feature contract may not use EVALUATION behavior."
        )

    temporal = config["temporal_contract"]

    if temporal["fixed_bin_width_defined_here"] is not False:
        raise RuntimeError(
            "Dataset protocol must not choose temporal bin width."
        )
    if temporal["fixed_history_length_defined_here"] is not False:
        raise RuntimeError(
            "Dataset protocol must not choose history length."
        )
    if (
        temporal["fixed_prediction_horizon_defined_here"]
        is not False
    ):
        raise RuntimeError(
            "Dataset protocol must not choose prediction horizon."
        )

    forbidden = config["forbidden_before_router_v2_freeze"]
    if not all(bool(value) for value in forbidden.values()):
        raise RuntimeError(
            "A pre-Router-V2 forbidden-operation guard was disabled."
        )

    schema_manifest = _json(
        ROOT / config["paths"]["schema_manifest"]
    )
    source_manifest = _json(
        ROOT / config["paths"]["source_lock_manifest"]
    )

    readiness = schema_manifest["structural_readiness"]

    if readiness["ready_for_adaptation_protocol"] is not True:
        raise RuntimeError(
            "Committed schema manifest is not adaptation-ready."
        )
    if readiness["all_timestamps_parse_cleanly"] is not True:
        raise RuntimeError(
            "Committed schema manifest has timestamp parse failures."
        )
    if (
        readiness["all_archives_monotonic_non_decreasing"]
        is not True
    ):
        raise RuntimeError(
            "Committed schema manifest has chronology failures."
        )
    if (
        readiness["all_row_field_counts_consistent"]
        is not True
    ):
        raise RuntimeError(
            "Committed schema manifest has field-count failures."
        )

    if (
        schema_manifest["source_lock_commit"]
        != study["source_lock_commit"]
    ):
        raise RuntimeError(
            "Schema manifest source-lock commit does not match protocol."
        )

    boundary = schema_manifest["inspection_boundary"]
    if any(bool(value) for value in boundary.values()):
        raise RuntimeError(
            "Schema-stage outcome boundary is no longer clean."
        )

    if source_manifest["source_archives_unzipped"] is not False:
        raise RuntimeError(
            "Source-lock provenance says archives were unzipped."
        )
    if source_manifest["sensor_values_opened"] is not False:
        raise RuntimeError(
            "Source-lock provenance says sensor values were opened."
        )
    if source_manifest["outcome_metrics_computed"] is not False:
        raise RuntimeError(
            "Source-lock provenance says outcomes were computed."
        )

    split = schema_manifest["frozen_split"]

    if len(split["train_days"]) != config["split"]["train_days"]:
        raise RuntimeError("TRAIN day count changed.")
    if (
        len(split["calibration_days"])
        != config["split"]["calibration_days"]
    ):
        raise RuntimeError("CALIBRATION day count changed.")
    if (
        len(split["evaluation_days"])
        != config["split"]["evaluation_days"]
    ):
        raise RuntimeError("EVALUATION day count changed.")

    candidates, digest = _candidate_envelope(schema_manifest)

    all_common = set(schema_manifest["archives"][0]["columns"])
    for archive in schema_manifest["archives"][1:]:
        all_common &= set(archive["columns"])

    timestamp_columns = sorted(
        {
            archive["timestamp_column"]
            for archive in schema_manifest["archives"]
        }
    )

    print(
        json.dumps(
            {
                "protocol_commit": PROTOCOL_COMMIT,
                "parent_schema_result_commit": study[
                    "parent_schema_result_commit"
                ],
                "source_lock_commit": study["source_lock_commit"],
                "archives": len(schema_manifest["archives"]),
                "train_days": len(split["train_days"]),
                "calibration_days": len(
                    split["calibration_days"]
                ),
                "evaluation_days": len(
                    split["evaluation_days"]
                ),
                "common_columns_including_timestamp": len(
                    all_common
                ),
                "timestamp_columns_excluded": timestamp_columns,
                "candidate_structural_field_count": len(candidates),
                "candidate_feature_sha256": digest,
                "hash_serialization": (
                    "UTF-8 exact feature names, "
                    "lexicographic order, LF-separated, final LF"
                ),
                "candidate_is_final_model_feature_set": False,
                "cobra_sensor_values_used": False,
                "cobra_model_outcomes_used": False,
                "evaluation_behavior_used": False,
                "ready_to_freeze_candidate_contract": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
