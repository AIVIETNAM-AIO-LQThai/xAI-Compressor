from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pynvml
import torch
import yaml

from ml.energy.hierarchical_inference import (
    seeded_condition_orders,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY_CONFIG = (
    ROOT
    / "configs"
    / "energy_aware_hierarchical_intelligence.yaml"
)
BENCHMARK_CONFIG = (
    ROOT
    / "configs"
    / "energy_aware_inference_benchmark.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected mapping in {path}."
        )
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected mapping in {path}."
        )
    return payload


def _gpu_preflight(
    device_index: int,
) -> dict[str, object]:
    pynvml.nvmlInit()

    try:
        handle = (
            pynvml
            .nvmlDeviceGetHandleByIndex(
                device_index
            )
        )
        name = pynvml.nvmlDeviceGetName(
            handle
        )
        driver = (
            pynvml.nvmlSystemGetDriverVersion()
        )

        fields = (
            pynvml.nvmlDeviceGetFieldValues(
                handle,
                [
                    pynvml
                    .NVML_FI_DEV_POWER_INSTANT,
                    pynvml
                    .NVML_FI_DEV_POWER_AVERAGE,
                ],
            )
        )

        values: list[dict[str, object]] = []

        labels = [
            "NVML_FI_DEV_POWER_INSTANT",
            "NVML_FI_DEV_POWER_AVERAGE",
        ]

        for label, field in zip(
            labels,
            fields,
            strict=True,
        ):
            ok = (
                field.nvmlReturn
                == pynvml.NVML_SUCCESS
            )
            values.append(
                {
                    "field": label,
                    "nvml_return": int(
                        field.nvmlReturn
                    ),
                    "supported": bool(ok),
                    "power_w": (
                        float(
                            field.value.uiVal
                        )
                        / 1000.0
                        if ok
                        else None
                    ),
                }
            )

        return {
            "gpu_device_index": (
                device_index
            ),
            "gpu_name": str(name),
            "driver_version": str(driver),
            "power_fields": values,
        }

    finally:
        pynvml.nvmlShutdown()


def main() -> None:
    study = _load_yaml(STUDY_CONFIG)
    benchmark = _load_yaml(
        BENCHMARK_CONFIG
    )

    # Catch accidental loss of the original study guardrails.
    decision = study.get("decision", {})
    outputs = study.get("outputs")
    guardrails = study.get("guardrails")

    if (
        decision.get(
            "promotion_allowed"
        )
        is not False
    ):
        raise RuntimeError(
            "Study promotion_allowed:false "
            "guardrail is missing."
        )

    if not isinstance(outputs, dict):
        raise TypeError(
            "Study outputs section is missing."
        )

    if not isinstance(
        guardrails,
        list,
    ) or not guardrails:
        raise RuntimeError(
            "Study guardrails section is missing."
        )

    tcn = benchmark["tcn"]

    evidence = benchmark["tcn_evidence"]

    if evidence["score_smoothing"] != "none":
        raise RuntimeError("TCN evidence score smoothing must remain disabled.")
    if evidence["persistence"] != "none":
        raise RuntimeError("TCN evidence persistence must remain disabled.")
    if int(evidence["episode_merge_minutes"]) != 30:
        raise RuntimeError("TCN evidence episode merge must remain 30 minutes.")
    if int(evidence["episode_reset_gap_minutes"]) != 10:
        raise RuntimeError("TCN evidence reset gap must remain 10 minutes.")

    energy = benchmark[
        "energy_measurement"
    ]
    benchmark_outputs = benchmark[
        "outputs"
    ]

    checkpoint_path = (
        ROOT / tcn["checkpoint_file"]
    )
    observed_sha = sha256_file(
        checkpoint_path
    )
    expected_sha = str(
        tcn["checkpoint_sha256"]
    )

    if observed_sha != expected_sha:
        raise RuntimeError(
            "TCN checkpoint SHA256 mismatch. "
            f"Expected {expected_sha}, "
            f"got {observed_sha}."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        checkpoint.get("scaler_name")
        != "standard"
    ):
        raise RuntimeError(
            "Benchmark TCN checkpoint must "
            "use StandardScaler."
        )

    model_config = checkpoint[
        "model_config"
    ]

    if (
        int(model_config["hidden_dim"])
        != 32
        or list(
            model_config["dilations"]
        )
        != [1, 2, 4]
        or int(
            model_config["kernel_size"]
        )
        != 3
    ):
        raise RuntimeError(
            "TCN checkpoint architecture "
            "does not match the frozen study."
        )

    router_path = (
        ROOT
        / benchmark_outputs[
            "router_calibration_file"
        ]
    )
    router = _load_json(
        router_path
    )

    if (
        router["data_use"][
            "test_file_opened"
        ]
        is not False
    ):
        raise RuntimeError(
            "Router calibration unexpectedly "
            "opened TEST."
        )

    if (
        router["tcn_checkpoint"][
            "sha256"
        ]
        != expected_sha
    ):
        raise RuntimeError(
            "Router calibration was not "
            "frozen to this TCN checkpoint."
        )

    expected_points = {
        "route_q90",
        "route_q95",
        "route_q99",
    }
    observed_points = set(
        router["router"][
            "operating_points"
        ]
    )

    if observed_points != expected_points:
        raise RuntimeError(
            "Router operating points do not "
            "match q90/q95/q99."
        )

    if int(
        energy[
            "warmup_runs_per_condition"
        ]
    ) != 2:
        raise RuntimeError(
            "Expected two warm-up runs."
        )

    repetitions = int(
        energy[
            "measured_repetitions"
        ]
    )
    if repetitions != 5:
        raise RuntimeError(
            "Expected five measured repetitions."
        )

    if float(
        energy[
            "sampling_interval_seconds"
        ]
    ) != 0.10:
        raise RuntimeError(
            "Expected 100 ms NVML sampling."
        )

    if float(
        energy[
            "paired_measurement"
        ][
            "minimum_workload_seconds"
        ]
    ) != 20.0:
        raise RuntimeError(
            "Expected a minimum 20-second "
            "metered workload."
        )

    materiality = float(
        energy[
            "materiality"
        ][
            "minimum_mean_gpu_energy_reduction_fraction"
        ]
    )
    if materiality != 0.03:
        raise RuntimeError(
            "Expected frozen 3% GPU-energy "
            "materiality threshold."
        )

    condition_config = energy[
        "condition_order"
    ]
    conditions = [
        str(value)
        for value
        in condition_config[
            "conditions"
        ]
    ]

    orders = seeded_condition_orders(
        conditions,
        repetitions=repetitions,
        seed=int(
            condition_config["seed"]
        ),
    )

    gpu = _gpu_preflight(
        int(
            energy[
                "gpu_device_index"
            ]
        )
    )

    if not all(
        bool(field["supported"])
        for field
        in gpu["power_fields"]
    ):
        raise RuntimeError(
            "Required NVML power fields are "
            "not supported."
        )

    report = {
        "schema_version": (
            "aeroxai.energy_aware_pretest_preflight.v1"
        ),
        "study": study["study"]["name"],
        "purpose": (
            "Freeze and verify the hierarchical "
            "inference benchmark before primary TEST."
        ),
        "test_file_opened": False,
        "ready_for_primary_test": True,
        "tcn_checkpoint": {
            "file": tcn[
                "checkpoint_file"
            ],
            "sha256": observed_sha,
            "scaler": (
                checkpoint[
                    "scaler_name"
                ]
            ),
            "model_config": (
                model_config
            ),
        },
        "router_calibration_file": str(
            router_path.relative_to(ROOT)
        ),
        "router_operating_points": (
            router["router"][
                "operating_points"
            ]
        ),
        "measurement": {
            "scope": (
                energy["scope"]
            ),
            "gpu": gpu,
            "sampling_interval_seconds": (
                float(
                    energy[
                        "sampling_interval_seconds"
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
                repetitions
            ),
            "cooldown_seconds": float(
                energy[
                    "paired_measurement"
                ][
                    "cooldown_seconds"
                ]
            ),
            "matched_idle_seconds": float(
                energy[
                    "paired_measurement"
                ][
                    "matched_idle_seconds"
                ]
            ),
            "minimum_workload_seconds": float(
                energy[
                    "paired_measurement"
                ][
                    "minimum_workload_seconds"
                ]
            ),
            "condition_orders": orders,
            "materiality_fraction": (
                materiality
            ),
            "cpu_energy": "UNKNOWN",
            "whole_system_energy": (
                "UNKNOWN"
            ),
        },
        "study_guardrails_present": True,
        "tcn_evidence_definition": {
            "score_smoothing": evidence["score_smoothing"],
            "persistence": evidence["persistence"],
            "episode_merge_minutes": int(evidence["episode_merge_minutes"]),
            "episode_reset_gap_minutes": int(evidence["episode_reset_gap_minutes"]),
            "episode_coverage_rule": evidence["episode_coverage_rule"],
        },
    }

    output_path = (
        ROOT
        / benchmark_outputs[
            "pretest_preflight_file"
        ]
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "ready_for_primary_test": True,
                "test_file_opened": False,
                "tcn_checkpoint_sha256": (
                    observed_sha
                ),
                "router_operating_points": (
                    router["router"][
                        "operating_points"
                    ]
                ),
                "gpu_name": (
                    gpu["gpu_name"]
                ),
                "condition_orders": orders,
                "output": str(
                    output_path.relative_to(ROOT)
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
