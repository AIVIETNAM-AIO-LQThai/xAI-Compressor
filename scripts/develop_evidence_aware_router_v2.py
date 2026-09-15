from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ml.energy.evidence_aware_router_v2 import (
    V2_FEATURE_NAMES,
    candidate_grid,
    evaluate_candidate,
    select_candidate,
)
from ml.energy.rcsd1yd_evaluation import (
    build_frozen_routes,
    build_tcn_model,
    build_temporal_batch_from_checkpoint,
    score_tcn_batch,
)
from scripts.develop_evidence_aware_router import (
    _build_dataset,
)

ROOT = Path(__file__).resolve().parents[1]

V2_CONFIG_PATH = (
    ROOT / "configs/evidence_aware_router_v2.yaml"
)
V1_CONFIG_PATH = (
    ROOT / "configs/evidence_aware_compute_routing.yaml"
)
V1_RESULT_PATH = (
    ROOT
    / "docs/research/evidence_aware_router_development.json"
)
RCSD_CONFIG_PATH = (
    ROOT / "configs/rcsd1yd_router_failure_analysis.yaml"
)
PREFLIGHT_RESULT_PATH = (
    ROOT
    / "docs/research/evidence_aware_router_v2_preflight.json"
)

OUTPUT_PATH = (
    ROOT
    / "docs/research/evidence_aware_router_v2_development.json"
)

REQUIRED_PARENT_COMMIT = (
    "e2ef48e1df0f54ad07faea691fd72a6bfecde357"
)

IMPLEMENTATION_PATHS = [
    "ml/energy/evidence_aware_router_v2.py",
    "scripts/develop_evidence_aware_router_v2.py",
    "tests/test_evidence_aware_router_v2_development.py",
]


def _yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected mapping in {path}."
        )

    return payload


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected mapping in {path}."
        )

    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _verify_execution_boundary() -> None:
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            REQUIRED_PARENT_COMMIT,
            "HEAD",
        ],
        cwd=ROOT,
        check=False,
    )

    if ancestor.returncode != 0:
        raise RuntimeError(
            "Required Router V2 parent is not "
            "an ancestor of HEAD."
        )

    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            *IMPLEMENTATION_PATHS,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    if status.stdout.strip():
        raise RuntimeError(
            "Router V2 implementation files are "
            "not clean and committed. Freeze the "
            "implementation before development."
        )

    if OUTPUT_PATH.exists():
        raise RuntimeError(
            "Router V2 development output already "
            "exists; refusing to overwrite it."
        )


def _assert_close(
    observed: float,
    expected: float,
    *,
    label: str,
    tolerance: float = 1.0e-9,
) -> None:
    if not np.isclose(
        float(observed),
        float(expected),
        rtol=0.0,
        atol=tolerance,
    ):
        raise RuntimeError(
            f"{label} failed reproduction: "
            f"{observed} != {expected}."
        )


def _rebuild_metropt_domains() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    config = _yaml(V1_CONFIG_PATH)
    frozen_result = _json(V1_RESULT_PATH)

    development = config["development_data"]

    primary_reference = _json(
        ROOT
        / development["metropt3"][
            "feature_reference"
        ]
    )

    primary_features = [
        str(value)
        for value in primary_reference[
            "features"
        ]["names"]
    ]

    if len(primary_features) != 63:
        raise RuntimeError(
            "Frozen primary feature count changed."
        )

    domains: dict[str, dict[str, Any]] = {}
    provenance: dict[str, Any] = {}

    for name in ("metropt3", "metropt2"):
        bundle = _build_dataset(
            dataset_name=name,
            data_config=development[name],
            config=config,
            primary_features=primary_features,
        )

        expected = frozen_result[
            "development_datasets"
        ][name]

        if bundle["split"] != expected["split"]:
            raise RuntimeError(
                f"{name} chronological split changed."
            )

        if (
            bundle["teacher_counts"]
            != expected["teacher_counts"]
        ):
            raise RuntimeError(
                f"{name} teacher counts changed."
            )

        for key in (
            "high_evidence_q90_fit",
            "alert_evidence_q995_fit",
        ):
            _assert_close(
                bundle["teacher_thresholds"][key],
                expected["teacher_thresholds"][key],
                label=f"{name} {key}",
            )

        features = (
            bundle["validation_features"]
            .loc[:, V2_FEATURE_NAMES]
            .copy()
        )

        high = np.asarray(
            bundle["validation_high_target"],
            dtype=bool,
        )
        alert = np.asarray(
            bundle["validation_alert_target"],
            dtype=bool,
        )

        domain_name = f"{name}_consumed"

        domains[domain_name] = {
            "features": features,
            "high": high,
            "alert": alert,
        }

        provenance[domain_name] = {
            "partition": (
                "consumed_router_validation"
            ),
            "rows": len(features),
            "high_evidence_units": int(
                high.sum()
            ),
            "alert_evidence_units": int(
                alert.sum()
            ),
            "teacher_thresholds": (
                expected["teacher_thresholds"]
            ),
            "checkpoint_sha256": (
                expected["checkpoint_sha256"]
            ),
        }

    return domains, provenance


def _load_partition(
    path: Path,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)

    frame.index = pd.DatetimeIndex(
        frame.index,
        name="timestamp",
    )

    return frame.sort_index()


def _rebuild_rcsd_domain() -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    config = _yaml(RCSD_CONFIG_PATH)

    frozen = config["frozen_inputs"]

    adaptation = _json(
        ROOT / frozen["adaptation_freeze"]
    )

    checkpoint = torch.load(
        ROOT / frozen["tcn_checkpoint"],
        map_location="cpu",
        weights_only=False,
    )

    evaluation = _load_partition(
        ROOT / frozen["evaluation_parquet"]
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = build_tcn_model(
        checkpoint,
        device=device,
    )

    temporal_batch = (
        build_temporal_batch_from_checkpoint(
            evaluation,
            checkpoint,
        )
    )

    scores = score_tcn_batch(
        model,
        temporal_batch,
        device=device,
        batch_size=1024,
    )

    context = build_frozen_routes(
        frame=evaluation,
        temporal_batch=temporal_batch,
        adaptation_freeze=adaptation,
    )

    high_threshold = float(
        adaptation["tcn"][
            "calibration_high_evidence_threshold_q90"
        ]
    )

    alert_threshold = float(
        adaptation["tcn"][
            "calibration_alert_evidence_threshold_q995"
        ]
    )

    high = scores >= high_threshold
    alert = scores >= alert_threshold

    guard = config["reproduction_guard"]

    _assert_close(
        high_threshold,
        guard["high_threshold_q90"],
        label="RCSD frozen q90 teacher",
        tolerance=1.0e-12,
    )

    _assert_close(
        alert_threshold,
        guard["alert_threshold_q995"],
        label="RCSD frozen q995 teacher",
        tolerance=1.0e-12,
    )

    if len(scores) != int(
        guard["valid_tcn_rows"]
    ):
        raise RuntimeError(
            "RCSD valid TCN row count changed."
        )

    if int(high.sum()) != int(
        guard["evaluation_high_evidence_bins"]
    ):
        raise RuntimeError(
            "RCSD high-evidence count changed."
        )

    if int(alert.sum()) != int(
        guard["evaluation_alert_evidence_bins"]
    ):
        raise RuntimeError(
            "RCSD alert-evidence count changed."
        )

    old_route = context.routes[
        "frozen_evidence_aware_router"
    ]
    old_q90 = context.routes[
        "fixed_pca_route_q90"
    ]

    if int(old_route.sum()) != int(
        guard["evidence_aware_invocations"]
    ):
        raise RuntimeError(
            "RCSD frozen Router V1 reproduction "
            "changed."
        )

    if int(old_q90.sum()) != int(
        guard["fixed_q90_invocations"]
    ):
        raise RuntimeError(
            "RCSD frozen q90 reproduction changed."
        )

    features = (
        context.router_features
        .loc[:, V2_FEATURE_NAMES]
        .copy()
    )

    domain = {
        "features": features,
        "high": np.asarray(
            high,
            dtype=bool,
        ),
        "alert": np.asarray(
            alert,
            dtype=bool,
        ),
    }

    provenance = {
        "partition": "consumed_evaluation",
        "device": str(device),
        "rows": len(features),
        "high_evidence_units": int(
            high.sum()
        ),
        "alert_evidence_units": int(
            alert.sum()
        ),
        "teacher_thresholds": {
            "high_evidence_q90": high_threshold,
            "alert_evidence_q995": (
                alert_threshold
            ),
        },
    }

    return domain, provenance


def main() -> None:
    _verify_execution_boundary()

    v2_config = _yaml(V2_CONFIG_PATH)
    preflight = _json(
        PREFLIGHT_RESULT_PATH
    )

    if (
        preflight[
            "router_v2_candidate_metrics_computed"
        ]
        is not False
    ):
        raise RuntimeError(
            "Preflight says V2 candidate metrics "
            "were already computed."
        )

    if (
        preflight["router_v2_policy_selected"]
        is not False
    ):
        raise RuntimeError(
            "Preflight says a V2 policy was "
            "already selected."
        )

    if any(
        bool(preflight[key])
        for key in (
            "cobra_sensor_values_used",
            "cobra_model_outcomes_used",
            "cobra_evaluation_behavior_used",
        )
    ):
        raise RuntimeError(
            "CoBra firewall failed before V2 "
            "development."
        )

    candidates = candidate_grid(
        v2_config
    )

    metropt_domains, metropt_provenance = (
        _rebuild_metropt_domains()
    )

    rcsd_domain, rcsd_provenance = (
        _rebuild_rcsd_domain()
    )

    domains = {
        **metropt_domains,
        "rcsd1yd_consumed": rcsd_domain,
    }

    expected_domains = (
        v2_config["development_domains"]
    )

    if list(domains) != expected_domains:
        raise RuntimeError(
            "Router V2 development domain "
            "identity/order changed."
        )

    adequacy = v2_config["adequacy"]
    eligibility = v2_config["eligibility"]

    reports = [
        evaluate_candidate(
            candidate=candidate,
            domains=domains,
            minimum_high_units=int(
                adequacy[
                    "minimum_high_evidence_units"
                ]
            ),
            minimum_alert_units=int(
                adequacy[
                    "minimum_alert_evidence_units"
                ]
            ),
            minimum_high_coverage=float(
                eligibility[
                    "minimum_high_evidence_coverage"
                ]
            ),
            minimum_alert_coverage=float(
                eligibility[
                    "minimum_alert_evidence_coverage"
                ]
            ),
        )
        for candidate in candidates
    ]

    weight_order = [
        tuple(
            float(value)
            for value in row
        )
        for row in v2_config[
            "candidate_families"
        ]["convex"]["weights"]
    ]

    selected = select_candidate(
        reports,
        convex_weight_order=weight_order,
    )

    eligible_count = sum(
        bool(report["eligible"])
        for report in reports
    )

    final_status = (
        "ELIGIBLE_POLICY_SELECTED"
        if selected is not None
        else "NO_ELIGIBLE_CANDIDATE"
    )

    payload = {
        "schema_version": (
            "aeroxai.evidence_aware_router_v2_development.v1"
        ),
        "study": "evidence_aware_router_v2",
        "evidence_class": (
            "CONSUMED_DEVELOPMENT_EVIDENCE"
        ),
        "independent_validation": False,
        "implementation_commit": (
            _git_head()
        ),
        "candidate_count": len(candidates),
        "eligible_candidate_count": (
            eligible_count
        ),
        "development_domains": {
            **metropt_provenance,
            "rcsd1yd_consumed": (
                rcsd_provenance
            ),
        },
        "selection_status": final_status,
        "selected": selected,
        "candidate_reports": reports,
        "source_artifacts": {
            "v2_config": {
                "path": str(
                    V2_CONFIG_PATH.relative_to(
                        ROOT
                    )
                ),
                "sha256": _sha256(
                    V2_CONFIG_PATH
                ),
            },
            "v2_preflight_result": {
                "path": str(
                    PREFLIGHT_RESULT_PATH.relative_to(
                        ROOT
                    )
                ),
                "sha256": _sha256(
                    PREFLIGHT_RESULT_PATH
                ),
            },
            "router_v1_development": {
                "path": str(
                    V1_RESULT_PATH.relative_to(
                        ROOT
                    )
                ),
                "sha256": _sha256(
                    V1_RESULT_PATH
                ),
            },
        },
        "cobra_sensor_values_used": False,
        "cobra_model_outcomes_used": False,
        "cobra_evaluation_behavior_used": False,
        "no_rescue_tuning": bool(
            v2_config["no_rescue_tuning"]
        ),
    }

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
                "implementation_commit": (
                    payload[
                        "implementation_commit"
                    ]
                ),
                "candidate_count": (
                    len(candidates)
                ),
                "eligible_candidate_count": (
                    eligible_count
                ),
                "selection_status": (
                    final_status
                ),
                "selected": selected,
                "cobra_model_outcomes_used": (
                    False
                ),
                "output": str(
                    OUTPUT_PATH.relative_to(
                        ROOT
                    )
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
