from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.runtime import (
    REPO_ROOT,
    load_runtime_config,
)

DOCS_DIR = REPO_ROOT / "docs"


def load_json_report(
    filename: str,
) -> dict[str, Any]:
    path = DOCS_DIR / filename

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise TypeError(
            f"{filename} must contain "
            "a JSON object."
        )

    return data


def build_detection_evidence() -> dict[str, Any]:
    benchmark = load_json_report("detector_benchmark.json")
    xai = load_json_report("xai_report.json")
    verification = load_json_report("xai_verification_report.json")

    primary = benchmark[
        "details"
    ]["pca_robust"]

    return {
        "evidence_class": "REAL",
        "dataset": "MetroPT-3",
        "detector": primary["model"],
        "metrics": {
            "timely_incident_recall":
                primary[
                    "timely_incident_recall"
                ],
            "anytime_incident_recall":
                primary[
                    "anytime_incident_recall"
                ],
            "pre_onset_incident_recall":
                primary[
                    "pre_onset_incident_recall"
                ],
            "incident_overlap_recall":
                primary[
                    "incident_overlap_recall"
                ],
            "false_alerts_per_24h":
                primary[
                    "false_alerts_per_24h"
                ],
            "time_in_alert_fraction":
                primary[
                    "time_in_alert_fraction"
                ],
            "relevant_episode_precision":
                primary[
                    "relevant_episode_precision"
                ],
            "pr_auc":
                primary["pr_auc"],
            "episodes_total":
                primary["episodes_total"],
            "false_episodes":
                primary["false_episodes"],
        },
        "incidents": xai["incidents"],
        "explanation_basis": (
            xai["explanation"][
                "basis"
            ]
        ),
        "causal_claim": bool(
            xai["explanation"][
                "causal_claim"
            ]
        ),
        "verification": {
            "evidence_subtype":
                verification[
                    "evidence_subtype"
                ],
            "method":
                verification[
                    "method"
                ],
            "acceptance":
                verification[
                    "acceptance"
                ],
            "incidents":
                verification[
                    "incidents"
                ],
            "limitations":
                verification[
                    "limitations"
                ],
        },
        "limitations": [
            (
                "MetroPT evidence supports "
                "detection and explanation, "
                "not multi-compressor energy "
                "optimization claims."
            ),
            (
                "Benchmark-false episodes "
                "are not necessarily "
                "physically normal because "
                "incident labels are sparse."
            ),
            (
                "PCA contributions identify "
                "signals associated with "
                "anomaly evidence, not "
                "confirmed root cause."
            ),
            (
                "Contribution percentiles "
                "describe how unusual detector "
                "evidence is relative to the "
                "calibration partition; they "
                "are not fault probabilities."
            ),
            (
                "Counterfactual repair tests "
                "detector dependence in PCA "
                "feature space and does not "
                "represent a physical repair "
                "or causal intervention."
            ),
        ],
    }


def build_evidence_summary() -> dict[str, Any]:
    runtime = load_runtime_config()

    detection = (
        build_detection_evidence()
    )

    twin = load_json_report(
        "digital_twin_report.json"
    )

    baseline = load_json_report(
        "baseline_controller_report.json"
    )

    optimizer = load_json_report(
        "optimizer_report.json"
    )

    action = load_json_report(
        "action_explanations.json"
    )

    robustness = load_json_report(
        "robustness_report.json"
    )

    safety_config = runtime.project[
        "safety"
    ]

    optimizer_interval = float(
        optimizer["method"][
            "optimizer_interval_seconds"
        ]
    )

    open_loop_approved = bool(
        robustness["robustness"][
            "all_tested_perturbations_safe"
        ]
    )

    return {
        "project": "AeroXAI",
        "deployment": {
            "mode": safety_config[
                "deployment_mode"
            ],
            "override_equipment_ctrl":
                bool(
                    safety_config[
                        "override_equipment_ctrl"
                    ]
                ),
        },
        "evidence_classes": {
            "REAL": {
                "scope": (
                    "MetroPT-3 telemetry "
                    "detection and XAI"
                ),
                "detection": detection,
            },
            "SIMULATED": {
                "scope": (
                    "Digital twin, baseline "
                    "controller, optimization, "
                    "action explanations, and "
                    "robustness stress tests"
                ),
                "digital_twin": twin,
                "baseline": baseline,
                "optimizer": optimizer,
                "action_explanations": {
                    "summary":
                        action["summary"],
                    "causal_claim":
                        action[
                            "causal_claim"
                        ],
                    "acceptance":
                        action[
                            "acceptance"
                        ],
                },
                "robustness": robustness[
                    "robustness"
                ],
            },
            "LITERATURE": {
                "scope": (
                    "External references "
                    "support context and "
                    "industry framing only."
                ),
                "runtime_embedded": False,
            },
        },
        "safety": {
            "deployment_mode":
                safety_config[
                    "deployment_mode"
                ],
            "override_equipment_ctrl":
                bool(
                    safety_config[
                        "override_equipment_ctrl"
                    ]
                ),
            "recommendation_valid_for_seconds":
                optimizer_interval,
            "requires_reoptimization": True,
            "open_loop_schedule_approved":
                open_loop_approved,
            "robustness_status":
                robustness[
                    "robustness"
                ]["status"],
        },
    }


def write_evidence_summary(
    path: Path | None = None,
) -> Path:
    output_path = (
        path
        if path is not None
        else (
            DOCS_DIR
            / "evidence_summary.json"
        )
    )

    summary = build_evidence_summary()

    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return output_path


if __name__ == "__main__":
    written = write_evidence_summary()
    print(
        "Evidence summary written to:",
        written,
    )
