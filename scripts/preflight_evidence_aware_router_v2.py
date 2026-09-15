from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

PREREGISTRATION_COMMIT = (
    "c59659b8c131fbe28cef8e50b966acabb78cb54e"
)

CONFIG_PATH = ROOT / "configs/evidence_aware_router_v2.yaml"

V1_DEVELOPMENT_PATH = (
    ROOT
    / "docs/research/evidence_aware_router_development.json"
)

RCSD_FAILURE_PATH = (
    ROOT
    / "docs/research/rcsd1yd_router_failure_analysis.json"
)

RCSD_EXTERNAL_RESULT_PATH = (
    ROOT
    / "docs/research/rcsd1yd_external_evaluation_result.json"
)

COBRA_CONTRACT_PATH = (
    ROOT
    / "docs/research/cobra_structural_compatibility_contract.json"
)

OUTPUT_PATH = (
    ROOT
    / "docs/research/evidence_aware_router_v2_preflight.json"
)

EXPECTED_INPUTS = [
    "pca_ewma_percentile",
    "recent_q90_hit_fraction",
    "recent_q95_hit_fraction",
]

EXPECTED_THRESHOLDS = [
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]

EXPECTED_CONVEX_WEIGHTS = [
    [1.00, 0.00, 0.00],
    [0.75, 0.25, 0.00],
    [0.75, 0.00, 0.25],
    [0.50, 0.50, 0.00],
    [0.50, 0.25, 0.25],
    [0.50, 0.00, 0.50],
]

EXPECTED_SELECTION = [
    "minimize_worst_domain_invocation",
    "minimize_mean_domain_invocation",
    "maximize_worst_domain_high_evidence_coverage",
    "maximize_worst_applicable_alert_coverage",
    "prefer_family_order_max3_top2_mean_convex",
    "prefer_higher_threshold",
    "prefer_earlier_convex_weight_tuple",
]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON mapping in {path}.")
    return payload


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping in {path}.")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def enumerate_candidates(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    thresholds = [
        float(value)
        for value in config["thresholds"]
    ]

    families = config["candidate_families"]
    candidates: list[dict[str, Any]] = []

    if bool(families["max3"]["enabled"]):
        for threshold in thresholds:
            candidates.append(
                {
                    "family": "max3",
                    "threshold": threshold,
                    "weights": None,
                }
            )

    if bool(families["top2_mean"]["enabled"]):
        for threshold in thresholds:
            candidates.append(
                {
                    "family": "top2_mean",
                    "threshold": threshold,
                    "weights": None,
                }
            )

    if bool(families["convex"]["enabled"]):
        for weights in families["convex"]["weights"]:
            for threshold in thresholds:
                candidates.append(
                    {
                        "family": "convex",
                        "threshold": threshold,
                        "weights": [
                            float(value)
                            for value in weights
                        ],
                    }
                )

    return candidates


def validate_preregistered_candidate_space(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    if config["inputs"]["ordered"] != EXPECTED_INPUTS:
        raise RuntimeError(
            "Router V2 input identity/order changed."
        )

    if (
        bool(
            config["inputs"][
                "support_features_in_primary_family"
            ]
        )
        is not False
    ):
        raise RuntimeError(
            "Support features entered the primary V2 family."
        )

    if (
        bool(config["inputs"]["tcn_score_online_input"])
        is not False
    ):
        raise RuntimeError(
            "TCN score became an online router input."
        )

    if (
        bool(config["inputs"]["incident_label_input"])
        is not False
    ):
        raise RuntimeError(
            "Incident labels became router inputs."
        )

    observed_thresholds = [
        float(value)
        for value in config["thresholds"]
    ]

    if observed_thresholds != EXPECTED_THRESHOLDS:
        raise RuntimeError(
            "Router V2 threshold grid changed."
        )

    observed_weights = [
        [float(value) for value in weights]
        for weights in config[
            "candidate_families"
        ]["convex"]["weights"]
    ]

    if observed_weights != EXPECTED_CONVEX_WEIGHTS:
        raise RuntimeError(
            "Router V2 convex weight grid changed."
        )

    for weights in observed_weights:
        if any(value < 0.0 for value in weights):
            raise RuntimeError(
                "A Router V2 convex weight is negative."
            )

        if abs(sum(weights) - 1.0) > 1e-12:
            raise RuntimeError(
                "A Router V2 convex tuple does not sum to one."
            )

    if config["selection"] != EXPECTED_SELECTION:
        raise RuntimeError(
            "Router V2 candidate-selection order changed."
        )

    if bool(config["no_rescue_tuning"]) is not True:
        raise RuntimeError(
            "Router V2 no-rescue guard was disabled."
        )

    candidates = enumerate_candidates(config)

    if len(candidates) != 120:
        raise RuntimeError(
            f"Expected 120 candidates, got {len(candidates)}."
        )

    if len(candidates) != int(
        config["expected_candidate_count"]
    ):
        raise RuntimeError(
            "Enumerated candidate count disagrees with config."
        )

    canonical = [
        json.dumps(
            candidate,
            sort_keys=True,
            separators=(",", ":"),
        )
        for candidate in candidates
    ]

    if len(canonical) != len(set(canonical)):
        raise RuntimeError(
            "Router V2 candidate grid contains duplicates."
        )

    return candidates


def _verify_git_ancestry() -> None:
    result = subprocess.run(
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

    if result.returncode != 0:
        raise RuntimeError(
            "Router V2 preregistration is not an ancestor "
            "of HEAD."
        )


def _verify_v1_consumed_artifact(
    payload: dict[str, Any],
) -> None:
    if (
        payload["schema_version"]
        != "aeroxai.evidence_aware_router_development.v1"
    ):
        raise RuntimeError(
            "Unexpected Router V1 development schema."
        )

    data_use = payload["data_use"]

    for key in (
        "metropt3_test_opened",
        "metropt2_test_opened",
        "incident_labels_used",
        "new_evaluation_data_opened",
    ):
        if bool(data_use[key]):
            raise RuntimeError(
                f"Router V1 provenance guard failed: {key}."
            )

    old_features = payload["router_features"]

    missing = [
        feature
        for feature in EXPECTED_INPUTS
        if feature not in old_features
    ]

    if missing:
        raise RuntimeError(
            "Router V2 input(s) absent from consumed "
            f"Router V1 artifact: {missing}"
        )

    for domain in ("metropt3", "metropt2"):
        if domain not in payload["development_datasets"]:
            raise RuntimeError(
                f"Consumed development domain missing: {domain}."
            )

        teacher = payload[
            "development_datasets"
        ][domain]["teacher_thresholds"]

        for key in (
            "high_evidence_q90_fit",
            "alert_evidence_q995_fit",
        ):
            value = float(teacher[key])
            if not (value == value):
                raise RuntimeError(
                    f"{domain} teacher threshold is NaN."
                )


def _verify_rcsd_consumed_artifact(
    payload: dict[str, Any],
) -> None:
    if (
        payload["schema_version"]
        != "aeroxai.rcsd1yd_router_failure_analysis.v1"
    ):
        raise RuntimeError(
            "Unexpected RCSD failure-analysis schema."
        )

    if payload["closed_external_status"] != (
        "EXTERNAL_TRANSPORT_FAIL"
    ):
        raise RuntimeError(
            "RCSD closed external status changed."
        )

    if bool(payload["closed_result_changed"]):
        raise RuntimeError(
            "RCSD closed result was retrospectively changed."
        )

    if (
        bool(payload["reproduction_guard_passed"])
        is not True
    ):
        raise RuntimeError(
            "RCSD reproduction guard did not pass."
        )

    evaluation_features = payload[
        "cheap_feature_transport"
    ]["EVALUATION"]

    missing = [
        feature
        for feature in EXPECTED_INPUTS
        if feature not in evaluation_features
    ]

    if missing:
        raise RuntimeError(
            "Router V2 input(s) absent from consumed "
            f"RCSD evidence: {missing}"
        )

    teacher_shift = payload["teacher_shift"]

    cal = teacher_shift["CALIBRATION"]
    evaluation = teacher_shift["EVALUATION"]

    if float(cal["high_threshold_q90"]) != float(
        evaluation["high_threshold_q90"]
    ):
        raise RuntimeError(
            "RCSD q90 teacher threshold was recomputed "
            "on EVALUATION."
        )

    if float(cal["alert_threshold_q995"]) != float(
        evaluation["alert_threshold_q995"]
    ):
        raise RuntimeError(
            "RCSD q995 teacher threshold was recomputed "
            "on EVALUATION."
        )


def _verify_cobra_firewall(
    config: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    cobra = config["cobra"]

    for key, value in cobra.items():
        if bool(value):
            raise RuntimeError(
                "CoBra Router V2 development firewall "
                f"is open at {key}."
            )

    interpretation = contract["interpretation"]

    if (
        bool(
            interpretation[
                "is_structural_compatibility_envelope"
            ]
        )
        is not True
    ):
        raise RuntimeError(
            "Unexpected CoBra structural contract."
        )

    if (
        bool(
            interpretation["is_final_model_feature_set"]
        )
        is not False
    ):
        raise RuntimeError(
            "CoBra structural envelope was promoted "
            "to a final feature set."
        )

    for key in (
        "cobra_sensor_values_used",
        "cobra_model_outcomes_used",
        "evaluation_behavior_used",
    ):
        if bool(interpretation[key]):
            raise RuntimeError(
                f"CoBra firewall provenance failed: {key}."
            )


def main() -> None:
    _verify_git_ancestry()

    config = _load_yaml(CONFIG_PATH)

    if config["study"]["status"] != (
        "preregistered_no_v2_result"
    ):
        raise RuntimeError(
            "Router V2 preregistration status changed."
        )

    candidates = validate_preregistered_candidate_space(
        config
    )

    v1_development = _load_json(
        V1_DEVELOPMENT_PATH
    )
    rcsd_failure = _load_json(
        RCSD_FAILURE_PATH
    )
    rcsd_external = _load_json(
        RCSD_EXTERNAL_RESULT_PATH
    )
    cobra_contract = _load_json(
        COBRA_CONTRACT_PATH
    )

    _verify_v1_consumed_artifact(v1_development)
    _verify_rcsd_consumed_artifact(rcsd_failure)
    _verify_cobra_firewall(
        config,
        cobra_contract,
    )

    if not rcsd_external:
        raise RuntimeError(
            "RCSD external result artifact is empty."
        )

    source_paths = {
        "router_v1_development": V1_DEVELOPMENT_PATH,
        "rcsd_failure_analysis": RCSD_FAILURE_PATH,
        "rcsd_external_result": RCSD_EXTERNAL_RESULT_PATH,
        "cobra_structural_contract": COBRA_CONTRACT_PATH,
    }

    source_artifacts = {
        name: {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
        }
        for name, path in source_paths.items()
    }

    payload = {
        "schema_version": (
            "aeroxai.evidence_aware_router_v2_preflight.v1"
        ),
        "study": "evidence_aware_router_v2",
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "candidate_count": len(candidates),
        "router_inputs": EXPECTED_INPUTS,
        "development_domains": [
            "metropt3_consumed",
            "metropt2_consumed",
            "rcsd1yd_consumed",
        ],
        "source_artifacts": source_artifacts,
        "consumed_prior_outcome_artifacts_read": True,
        "router_v2_candidate_metrics_computed": False,
        "router_v2_policy_selected": False,
        "cobra_sensor_values_used": False,
        "cobra_model_outcomes_used": False,
        "cobra_evaluation_behavior_used": False,
        "ready_to_freeze_development_implementation": True,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "candidate_count": payload[
                    "candidate_count"
                ],
                "router_inputs": payload[
                    "router_inputs"
                ],
                "development_domains": payload[
                    "development_domains"
                ],
                "router_v2_candidate_metrics_computed": False,
                "router_v2_policy_selected": False,
                "cobra_model_outcomes_used": False,
                "ready_to_freeze_development_implementation": True,
                "output": str(
                    OUTPUT_PATH.relative_to(ROOT)
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
