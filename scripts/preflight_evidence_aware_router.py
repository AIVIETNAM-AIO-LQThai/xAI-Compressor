from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ml.energy.evidence_aware_routing import (
    ROUTER_FEATURE_NAMES,
)
from ml.energy.hierarchical_inference import (
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "evidence_aware_compute_routing.yaml"
)


def _load_yaml(
    path: Path,
) -> dict[str, Any]:
    payload = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected mapping in {path}."
        )
    return payload


def _load_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected mapping in {path}."
        )
    return payload


def main() -> None:
    config = _load_yaml(
        CONFIG_PATH
    )

    development_path = (
        ROOT
        / config["outputs"][
            "development_calibration"
        ]
    )
    development = _load_json(
        development_path
    )

    data_use = development[
        "data_use"
    ]

    forbidden_flags = [
        "metropt3_test_opened",
        "metropt2_test_opened",
        "incident_labels_used",
        "new_evaluation_data_opened",
    ]

    if any(
        bool(data_use[key])
        for key in forbidden_flags
    ):
        raise RuntimeError(
            "Development provenance guard "
            "failed."
        )

    if int(
        development[
            "candidate_count"
        ]
    ) != int(
        config[
            "candidate_grid"
        ]["total_candidates"]
    ):
        raise RuntimeError(
            "Candidate count changed."
        )

    if (
        development[
            "router_features"
        ]
        != ROUTER_FEATURE_NAMES
    ):
        raise RuntimeError(
            "Router feature identity changed."
        )

    selected = development[
        "selection"
    ]["selected"]
    selected_model = development[
        "selection"
    ]["selected_model"]

    if selected is None:
        raise RuntimeError(
            "No evidence-aware policy was "
            "selected."
        )

    if selected_model is None:
        raise RuntimeError(
            "Selected model parameters are "
            "missing."
        )

    if not bool(
        selected["eligible"]
    ):
        raise RuntimeError(
            "Selected candidate is not "
            "eligible."
        )

    constraints = config[
        "selection"
    ]["constraints"]

    min_high = float(
        constraints[
            "minimum_high_evidence_coverage_each_validation_dataset"
        ]
    )
    min_alert = float(
        constraints[
            "minimum_tcn_alert_coverage_each_validation_dataset"
        ]
    )

    for (
        name,
        result,
    ) in selected["datasets"].items():
        if float(
            result[
                "high_evidence_coverage"
            ]
        ) < min_high:
            raise RuntimeError(
                f"{name} selected policy "
                "violates high-evidence "
                "coverage constraint."
            )

        if (
            bool(
                result[
                    "alert_constraint_applied"
                ]
            )
            and (
                result[
                    "tcn_alert_coverage"
                ]
                is None
                or float(
                    result[
                        "tcn_alert_coverage"
                    ]
                )
                < min_alert
            )
        ):
            raise RuntimeError(
                f"{name} selected policy "
                "violates alert-evidence "
                "coverage constraint."
            )

    development_cfg = config[
        "development_data"
    ]

    checkpoint_verification = {}

    for name in (
        "metropt3",
        "metropt2",
    ):
        item = development_cfg[name]
        observed = sha256_file(
            ROOT
            / item[
                "tcn_checkpoint"
            ]
        )
        expected = str(
            item[
                "tcn_checkpoint_sha256"
            ]
        )

        if observed != expected:
            raise RuntimeError(
                f"{name} checkpoint changed "
                "after development."
            )

        checkpoint_verification[
            name
        ] = {
            "path": (
                item[
                    "tcn_checkpoint"
                ]
            ),
            "sha256": observed,
        }

    if (
        config[
            "new_evaluation_data"
        ]["status"]
        != "not_yet_configured"
    ):
        raise RuntimeError(
            "New evaluation data was "
            "configured before policy freeze."
        )

    energy = config[
        "energy_measurement"
    ]

    payload = {
        "schema_version": (
            "aeroxai.evidence_aware_router_preflight.v1"
        ),
        "study": (
            config["study"]["name"]
        ),
        "preregistration_commit": (
            development[
                "preregistration_commit"
            ]
        ),
        "development_file": str(
            development_path.relative_to(
                ROOT
            )
        ),
        "selected_policy": selected,
        "selected_model": (
            selected_model
        ),
        "checkpoint_verification": (
            checkpoint_verification
        ),
        "data_use": data_use,
        "success_criteria": (
            config[
                "success_criteria"
            ]
        ),
        "energy_protocol": {
            "scope": energy["scope"],
            "nvml_sampling_interval_seconds": (
                float(
                    energy[
                        "nvml_sampling_interval_seconds"
                    ]
                )
            ),
            "warmup_runs_per_condition": (
                int(
                    energy[
                        "warmup_runs_per_condition"
                    ]
                )
            ),
            "measured_repetitions": (
                int(
                    energy[
                        "measured_repetitions"
                    ]
                )
            ),
            "matched_idle_seconds": (
                float(
                    energy[
                        "matched_idle_seconds"
                    ]
                )
            ),
            "minimum_metered_workload_seconds": (
                float(
                    energy[
                        "minimum_metered_workload_seconds"
                    ]
                )
            ),
            "router_cpu_energy": (
                energy[
                    "router_cpu_energy"
                ]
            ),
            "whole_system_energy": (
                energy[
                    "whole_system_energy"
                ]
            ),
        },
        "new_evaluation_data_status": (
            config[
                "new_evaluation_data"
            ]["status"]
        ),
        "ready_to_freeze_policy": True,
        "ready_for_simulated_mechanism_tests": (
            True
        ),
        "ready_for_primary_evaluation": (
            False
        ),
    }

    output_path = (
        ROOT
        / config["outputs"][
            "policy_preflight"
        ]
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
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
                "selected_policy": {
                    "C": selected["C"],
                    "class_weight": (
                        selected[
                            "class_weight"
                        ]
                    ),
                    "probability_threshold": (
                        selected[
                            "probability_threshold"
                        ]
                    ),
                    "mean_tcn_invocation_fraction": (
                        selected[
                            "mean_tcn_invocation_fraction"
                        ]
                    ),
                    "worst_dataset_high_evidence_coverage": (
                        selected[
                            "worst_dataset_high_evidence_coverage"
                        ]
                    ),
                    "worst_dataset_alert_coverage": (
                        selected[
                            "worst_dataset_alert_coverage"
                        ]
                    ),
                    "datasets": (
                        selected["datasets"]
                    ),
                },
                "ready_to_freeze_policy": True,
                "ready_for_simulated_mechanism_tests": (
                    True
                ),
                "ready_for_primary_evaluation": (
                    False
                ),
                "new_evaluation_data_status": (
                    payload[
                        "new_evaluation_data_status"
                    ]
                ),
                "output": str(
                    output_path.relative_to(
                        ROOT
                    )
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
