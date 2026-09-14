from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ml.energy.hierarchical_inference import (
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "support_aware_energy_routing.yaml"
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

    calibration_path = (
        ROOT
        / config["outputs"][
            "calibration_policy"
        ]
    )
    calibration = _load_json(
        calibration_path
    )

    if calibration[
        "data_use"
    ]["metropt3_test_opened"]:
        raise RuntimeError(
            "MetroPT-3 TEST was opened during "
            "policy calibration."
        )
    if calibration[
        "data_use"
    ]["metropt2_test_opened"]:
        raise RuntimeError(
            "MetroPT2 TEST was opened during "
            "policy calibration."
        )
    if calibration[
        "data_use"
    ][
        "incident_labels_used_for_selection"
    ]:
        raise RuntimeError(
            "Incident labels were used during "
            "policy calibration."
        )

    selected = calibration[
        "selection"
    ]["selected"]

    if selected is None:
        raise RuntimeError(
            "No support-aware policy was "
            "selected."
        )

    if not bool(
        selected["eligible"]
    ):
        raise RuntimeError(
            "Selected support-aware policy is "
            "not eligible."
        )

    constraints = config[
        "support_aware_policy"
    ]["eligibility_constraints"]

    min_high = float(
        constraints[
            "minimum_high_evidence_bin_coverage_each_development_dataset"
        ]
    )
    min_alert = float(
        constraints[
            "minimum_tcn_alert_bin_coverage_each_development_dataset"
        ]
    )
    min_alert_bins = int(
        constraints[
            "minimum_alert_bins_for_alert_constraint"
        ]
    )

    for (
        name,
        result,
    ) in selected["datasets"].items():
        if float(
            result[
                "high_evidence_bin_coverage"
            ]
        ) < min_high:
            raise RuntimeError(
                f"{name} violates the frozen "
                "high-evidence coverage "
                "constraint."
            )

        alert_bins = int(
            result["tcn_alert_bins"]
        )
        alert_coverage = result[
            "tcn_alert_bin_coverage"
        ]

        if (
            alert_bins >= min_alert_bins
            and (
                alert_coverage is None
                or float(alert_coverage)
                < min_alert
            )
        ):
            raise RuntimeError(
                f"{name} violates the frozen "
                "TCN-alert coverage "
                "constraint."
            )

    development = config[
        "development_data"
    ]

    checkpoint_verification = {}

    for name in (
        "metropt3",
        "metropt2",
    ):
        item = development[name]
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
                "after policy calibration."
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

    energy = config[
        "energy_measurement"
    ]

    if float(
        energy[
            "nvml_sampling_interval_seconds"
        ]
    ) != 0.10:
        raise RuntimeError(
            "Expected frozen 100 ms NVML "
            "sampling."
        )
    if int(
        energy[
            "warmup_runs_per_condition"
        ]
    ) != 2:
        raise RuntimeError(
            "Expected two warm-ups."
        )
    if int(
        energy[
            "measured_repetitions"
        ]
    ) != 5:
        raise RuntimeError(
            "Expected five energy "
            "repetitions."
        )

    new_data = config[
        "new_evaluation_data"
    ]

    payload = {
        "schema_version": (
            "aeroxai.support_aware_router_preflight.v1"
        ),
        "study": (
            config["study"]["name"]
        ),
        "preregistration_commit": (
            calibration[
                "preregistration_commit"
            ]
        ),
        "calibration_file": str(
            calibration_path.relative_to(ROOT)
        ),
        "selected_policy": selected,
        "checkpoint_verification": (
            checkpoint_verification
        ),
        "data_use": {
            "metropt3_test_opened": False,
            "metropt2_test_opened": False,
            "incident_labels_used_for_selection": (
                False
            ),
        },
        "support_definition": (
            calibration[
                "support_definition"
            ]
        ),
        "energy_protocol": {
            "scope": (
                energy["scope"]
            ),
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
            "cpu_router_energy": (
                energy[
                    "cpu_router_energy"
                ]
            ),
            "whole_system_energy": (
                energy[
                    "whole_system_energy"
                ]
            ),
        },
        "success_criteria": (
            config[
                "success_criteria"
            ]
        ),
        "new_evaluation_data_status": (
            new_data["status"]
        ),
        "ready_to_freeze_selected_policy": (
            True
        ),
        "ready_for_primary_evaluation": (
            new_data["status"]
            != "not_yet_configured"
        ),
    }

    output_path = (
        ROOT
        / config["outputs"][
            "preflight"
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
                    "mid_support_percentile": (
                        selected[
                            "mid_support_percentile"
                        ]
                    ),
                    "high_support_percentile": (
                        selected[
                            "high_support_percentile"
                        ]
                    ),
                    "mean_tcn_invocation_fraction": (
                        selected[
                            "mean_tcn_invocation_fraction"
                        ]
                    ),
                    "datasets": (
                        selected["datasets"]
                    ),
                },
                "ready_to_freeze_selected_policy": (
                    True
                ),
                "ready_for_primary_evaluation": (
                    payload[
                        "ready_for_primary_evaluation"
                    ]
                ),
                "new_evaluation_data_status": (
                    new_data["status"]
                ),
                "output": str(
                    output_path.relative_to(ROOT)
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
