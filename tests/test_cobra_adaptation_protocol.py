from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from scripts.preflight_cobra_adaptation import (
    _candidate_envelope,
)


def _config() -> dict:
    return yaml.safe_load(
        Path("configs/cobra_adaptation_evaluation.yaml")
        .read_text(encoding="utf-8")
    )


def test_protocol_keeps_cobra_outcomes_closed() -> None:
    config = _config()

    assert (
        config["study"][
            "cobra_model_outcomes_allowed_before_router_v2_freeze"
        ]
        is False
    )
    assert (
        config["study"][
            "zero_shot_external_validation_claim_allowed"
        ]
        is False
    )

    assert all(
        bool(value)
        for value in config[
            "forbidden_before_router_v2_freeze"
        ].values()
    )


def test_candidate_envelope_is_not_final_feature_set() -> None:
    config = _config()
    envelope = config["candidate_compatibility_envelope"]

    assert envelope["membership_rule"] == (
        "intersection_all_23_archives"
    )
    assert envelope["exclude_timestamp"] is True
    assert envelope["coarse_schema_class_required"] is False
    assert envelope["ordering"] == "lexicographic_exact_name"
    assert envelope["is_final_model_feature_set"] is False


def test_temporal_choices_are_deferred() -> None:
    config = _config()
    temporal = config["temporal_contract"]

    assert temporal["fixed_bin_width_defined_here"] is False
    assert temporal["fixed_history_length_defined_here"] is False
    assert temporal["fixed_prediction_horizon_defined_here"] is False
    assert (
        temporal["fixed_temporal_model_architecture_defined_here"]
        is False
    )


def test_candidate_envelope_uses_only_common_structural_fields() -> None:
    manifest = {
        "archives": [
            {
                "columns": ["timestamp", "B", "A", "OnlyFirst"],
                "timestamp_column": "timestamp",
                "coarse_schema_classes": {
                    "timestamp": "timestamp",
                    "A": "text_or_mixed",
                    "B": "numeric",
                    "OnlyFirst": "numeric",
                },
            },
            {
                "columns": ["B", "timestamp", "A"],
                "timestamp_column": "timestamp",
                "coarse_schema_classes": {
                    "timestamp": "timestamp",
                    "A": "text_or_mixed",
                    "B": "text_or_mixed",
                },
            },
        ]
    }

    manifest["archives"] = (
        manifest["archives"][:1]
        + [manifest["archives"][1]] * 22
    )

    features, digest = _candidate_envelope(manifest)

    assert features == ["A", "B"]

    expected = hashlib.sha256(
        b"A\nB\n"
    ).hexdigest()

    assert digest == expected


def test_final_feature_contract_cannot_use_model_behavior() -> None:
    config = _config()
    contract = config["final_feature_contract"]

    assert contract["may_use_cobra_sensor_values"] is False
    assert contract["may_use_cobra_model_performance"] is False
    assert contract["may_use_evaluation_behavior"] is False


def test_evaluation_firewall_is_one_shot() -> None:
    config = _config()
    firewall = config["evaluation_firewall"]

    assert firewall["one_shot_future_evaluation"] is True
    assert firewall["post_evaluation_refit_allowed"] is False
    assert (
        firewall["post_evaluation_threshold_tuning_allowed"]
        is False
    )
    assert firewall["post_evaluation_feature_tuning_allowed"] is False
    assert (
        firewall["post_evaluation_rescue_as_primary_result_allowed"]
        is False
    )
