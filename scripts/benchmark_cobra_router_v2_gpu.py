from __future__ import annotations

import json
import pickle
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ml.energy.hierarchical_inference import (
    energy_reduction_fraction,
    seeded_condition_orders,
    sha256_file,
)
from ml.temporal.detector import prepare_temporal_batch
from ml.temporal.tcn import (
    TCNForecaster,
    TemporalForecastConfig,
)
from scripts.benchmark_rcsd1yd_gpu import (
    _measure_condition,
    _measure_idle,
    _run_logical_pass,
)

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT / "configs/cobra_router_v2_adaptation.yaml"
)
FEATURE_PATH = (
    ROOT
    / "docs/research/cobra_router_v2_raw_feature_contract.json"
)
TRAIN_FIT_PATH = (
    ROOT
    / "docs/research/cobra_router_v2_train_fit.json"
)
EVALUATION_RESULT_PATH = (
    ROOT
    / "docs/research/cobra_router_v2_evaluation.json"
)

SKLEARN_ARTIFACT = (
    ROOT
    / "artifacts/research/cobra_router_v2_train_models.pkl"
)
TCN_ARTIFACT = (
    ROOT
    / "artifacts/research/cobra_router_v2_tcn.pt"
)

EVALUATION_DATA_PATH = (
    ROOT
    / "data/processed/cobra_router_v2/evaluation_5min.csv"
)
EVALUATION_SCORES_PATH = (
    ROOT
    / "data/processed/cobra_router_v2/evaluation_scores.csv"
)

FINAL_RESULT_PATH = (
    ROOT
    / "docs/research/cobra_router_v2_final_evaluation.json"
)

REQUIRED_EVALUATION_COMMIT = (
    "f7d646a14ca7b0928d6208134efe944e6ab3d325"
)

CONDITIONS = [
    "always_on_tcn",
    "router_v2",
]

CONDITION_ORDER_SEED = 20260915
INFERENCE_BATCH_SIZE = 1024
GPU_DEVICE_INDEX = 0
COOLDOWN_SECONDS = 5.0

IMPLEMENTATION_PATHS = [
    "scripts/benchmark_cobra_router_v2_gpu.py",
    "tests/test_cobra_router_v2_gpu.py",
]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected JSON mapping in {path}."
        )

    return value


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected YAML mapping in {path}."
        )

    return value


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def frozen_condition_orders(
    repetitions: int,
) -> list[list[str]]:
    return seeded_condition_orders(
        CONDITIONS,
        repetitions=repetitions,
        seed=CONDITION_ORDER_SEED,
    )


def energy_status(
    reductions: list[float],
    *,
    minimum_mean: float,
    positive_each_required: bool,
) -> tuple[str, float, bool]:
    if not reductions:
        raise ValueError(
            "Energy reductions cannot be empty."
        )

    mean_reduction = float(
        statistics.fmean(reductions)
    )

    positive_each = all(
        value > 0.0
        for value in reductions
    )

    passed = (
        mean_reduction >= minimum_mean
        and (
            not positive_each_required
            or positive_each
        )
    )

    return (
        "PASS" if passed else "FAIL",
        mean_reduction,
        positive_each,
    )


def overall_status(
    *,
    high_status: str,
    alert_status: str,
    gpu_energy_status: str,
) -> str:
    if (
        high_status == "FAIL"
        or alert_status == "FAIL"
        or gpu_energy_status == "FAIL"
    ):
        return "FAIL"

    if (
        high_status == "PASS"
        and alert_status == "PASS"
        and gpu_energy_status == "PASS"
    ):
        return "PASS"

    if (
        high_status == "PASS"
        and alert_status
        == "INSUFFICIENT_EVIDENCE"
        and gpu_energy_status == "PASS"
    ):
        return (
            "PARTIAL_PASS_ALERT_"
            "INSUFFICIENT_EVIDENCE"
        )

    return "INCONCLUSIVE"


def _verify_boundary() -> None:
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            REQUIRED_EVALUATION_COMMIT,
            "HEAD",
        ],
        cwd=ROOT,
        check=False,
    )

    if ancestor.returncode != 0:
        raise RuntimeError(
            "Frozen EVAL result is not "
            "an ancestor of HEAD."
        )

    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            *IMPLEMENTATION_PATHS,
            str(
                EVALUATION_RESULT_PATH.relative_to(
                    ROOT
                )
            ),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    if status.stdout.strip():
        raise RuntimeError(
            "Energy implementation or frozen "
            "EVAL result is dirty."
        )

    if FINAL_RESULT_PATH.exists():
        raise RuntimeError(
            "Final CoBra result already exists; "
            "refusing to overwrite."
        )


def _build_model(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
) -> TCNForecaster:
    cfg = checkpoint["model_config"]

    model = TCNForecaster(
        TemporalForecastConfig(
            input_dim=int(
                cfg["input_dim"]
            ),
            hidden_dim=int(
                cfg["hidden_dim"]
            ),
            kernel_size=int(
                cfg["kernel_size"]
            ),
            dilations=tuple(
                int(value)
                for value in cfg[
                    "dilations"
                ]
            ),
            dropout=float(
                cfg["dropout"]
            ),
        )
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    return model.to(device).eval()


def _route_array(
    scores: pd.DataFrame,
) -> np.ndarray:
    raw = scores[
        "route_tcn"
    ]

    if pd.api.types.is_bool_dtype(
        raw.dtype
    ):
        route = raw.to_numpy(
            dtype=bool
        )
    else:
        mapped = (
            raw.astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                }
            )
        )

        if mapped.isna().any():
            raise RuntimeError(
                "Could not parse route_tcn."
            )

        route = mapped.to_numpy(
            dtype=bool
        )

    return route


def main() -> None:
    _verify_boundary()

    config = _yaml(
        CONFIG_PATH
    )
    feature_contract = _json(
        FEATURE_PATH
    )
    train_fit = _json(
        TRAIN_FIT_PATH
    )
    evaluation_result = _json(
        EVALUATION_RESULT_PATH
    )

    if evaluation_result[
        "status"
    ] != "EVALUATION_FUNCTIONAL_COMPLETE":
        raise RuntimeError(
            "Frozen EVAL result status changed."
        )

    expected_data_sha = (
        evaluation_result[
            "representation"
        ][
            "file_sha256"
        ]
    )

    expected_scores_sha = (
        evaluation_result[
            "local_outputs"
        ][
            "scores_sha256"
        ]
    )

    if sha256_file(
        EVALUATION_DATA_PATH
    ) != expected_data_sha:
        raise RuntimeError(
            "Frozen EVALUATION data SHA mismatch."
        )

    if sha256_file(
        EVALUATION_SCORES_PATH
    ) != expected_scores_sha:
        raise RuntimeError(
            "Frozen EVALUATION scores SHA mismatch."
        )

    if sha256_file(
        SKLEARN_ARTIFACT
    ) != train_fit[
        "artifacts"
    ][
        "sklearn_sha256"
    ]:
        raise RuntimeError(
            "Frozen sklearn artifact SHA mismatch."
        )

    if sha256_file(
        TCN_ARTIFACT
    ) != train_fit[
        "artifacts"
    ][
        "tcn_sha256"
    ]:
        raise RuntimeError(
            "Frozen TCN artifact SHA mismatch."
        )

    features = [
        str(value)
        for value in feature_contract[
            "final_raw_features"
        ]
    ]

    with SKLEARN_ARTIFACT.open(
        "rb"
    ) as handle:
        trained = pickle.load(
            handle
        )

    if trained["features"] != features:
        raise RuntimeError(
            "Frozen model feature order changed."
        )

    tcn_scaler = trained[
        "tcn_scaler"
    ]

    checkpoint = torch.load(
        TCN_ARTIFACT,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint[
        "features"
    ] != features:
        raise RuntimeError(
            "Frozen TCN feature order changed."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Preregistered GPU energy "
            "measurement requires CUDA."
        )

    device = torch.device(
        "cuda"
    )

    model = _build_model(
        checkpoint,
        device=device,
    )

    evaluation = pd.read_csv(
        EVALUATION_DATA_PATH,
        parse_dates=["timestamp"],
        float_precision="round_trip",
    )

    indexed = (
        evaluation
        .sort_values("timestamp")
        .set_index("timestamp")
    )

    batch = prepare_temporal_batch(
        indexed,
        features=features,
        scaler=tcn_scaler,
        sequence_length=int(
            checkpoint[
                "history_bins"
            ]
        ),
        bin_minutes=int(
            checkpoint[
                "bin_minutes"
            ]
        ),
    )

    scores = pd.read_csv(
        EVALUATION_SCORES_PATH,
        parse_dates=["timestamp"],
    ).set_index(
        "timestamp"
    )

    if len(batch.inputs) != int(
        evaluation_result[
            "teacher_evidence"
        ][
            "eligible_targets"
        ]
    ):
        raise RuntimeError(
            "Frozen TCN target count changed."
        )

    if not batch.target_index.equals(
        scores.index
    ):
        raise RuntimeError(
            "Energy score timestamps differ "
            "from frozen TCN target timestamps."
        )

    route = _route_array(
        scores
    )

    expected_route_count = round(
        float(
            evaluation_result[
                "router_v2"
            ][
                "tcn_invocation_fraction"
            ]
        )
        * len(route)
    )

    observed_route_count = int(
        route.sum()
    )

    if observed_route_count != (
        expected_route_count
    ):
        raise RuntimeError(
            "Frozen route count changed: "
            f"{observed_route_count} != "
            f"{expected_route_count}"
        )

    if observed_route_count != 194:
        raise RuntimeError(
            "Expected frozen Router V2 "
            "invocation count of 194."
        )

    all_indices = np.arange(
        len(route),
        dtype=np.int64,
    )

    routed_indices = np.flatnonzero(
        route
    ).astype(
        np.int64,
        copy=False,
    )

    selected_by_condition = {
        "always_on_tcn": all_indices,
        "router_v2": routed_indices,
    }

    energy_cfg = config[
        "energy_measurement"
    ]
    success_cfg = config[
        "success_criteria"
    ][
        "gpu_energy"
    ]

    interval_s = float(
        energy_cfg[
            "sampling_interval_seconds"
        ]
    )
    warmups = int(
        energy_cfg[
            "warmup_runs_per_condition"
        ]
    )
    repetitions = int(
        energy_cfg[
            "measured_repetitions"
        ]
    )
    idle_s = float(
        energy_cfg[
            "matched_idle_seconds"
        ]
    )
    minimum_workload_s = float(
        energy_cfg[
            "minimum_metered_workload_seconds"
        ]
    )

    if interval_s != 0.10:
        raise RuntimeError(
            "Frozen NVML interval changed."
        )
    if warmups != 2:
        raise RuntimeError(
            "Frozen warmup count changed."
        )
    if repetitions != 5:
        raise RuntimeError(
            "Frozen repetition count changed."
        )
    if idle_s != 10.0:
        raise RuntimeError(
            "Frozen matched idle changed."
        )
    if minimum_workload_s != 20.0:
        raise RuntimeError(
            "Frozen workload minimum changed."
        )

    orders = frozen_condition_orders(
        repetitions
    )

    # Frozen two logical-pass warmups.
    for condition in CONDITIONS:
        selected = (
            selected_by_condition[
                condition
            ]
        )

        for _ in range(
            warmups
        ):
            _run_logical_pass(
                model=model,
                inputs=batch.inputs,
                targets=batch.targets,
                selected=selected,
                batch_size=INFERENCE_BATCH_SIZE,
                device=device,
            )

    records: list[
        dict[str, Any]
    ] = []

    print(
        "Measured CoBra GPU repetitions"
    )

    for repetition, order in enumerate(
        orders,
        start=1,
    ):
        print(
            f"  repetition "
            f"{repetition}/{repetitions}: "
            f"{order}"
        )

        for condition in order:
            selected = (
                selected_by_condition[
                    condition
                ]
            )

            time.sleep(
                COOLDOWN_SECONDS
            )

            idle = _measure_idle(
                device_index=GPU_DEVICE_INDEX,
                interval_s=interval_s,
                duration_s=idle_s,
            )

            workload = _measure_condition(
                model=model,
                inputs=batch.inputs,
                targets=batch.targets,
                selected=selected,
                batch_size=INFERENCE_BATCH_SIZE,
                device=device,
                device_index=GPU_DEVICE_INDEX,
                interval_s=interval_s,
                minimum_workload_s=(
                    minimum_workload_s
                ),
            )

            meter = workload[
                "meter"
            ]

            if meter is None:
                raise RuntimeError(
                    "Unexpected zero-invocation "
                    "condition."
                )

            passes = int(
                workload[
                    "logical_passes"
                ]
            )

            gross_j = float(
                meter[
                    "instant_energy_j"
                ]
            )
            duration_s = float(
                meter[
                    "duration_s"
                ]
            )
            idle_mean_w = float(
                idle[
                    "mean_instant_w"
                ]
            )

            idle_equivalent_j = (
                idle_mean_w
                * duration_s
            )

            incremental_j = (
                gross_j
                - idle_equivalent_j
            )

            accounting = {
                "gross_gpu_energy_j": (
                    gross_j
                ),
                "matched_idle_mean_power_w": (
                    idle_mean_w
                ),
                "matched_idle_equivalent_energy_j": (
                    idle_equivalent_j
                ),
                "incremental_gpu_energy_j": (
                    incremental_j
                ),
                "gross_gpu_energy_per_logical_pass_j": (
                    gross_j / passes
                ),
                "incremental_gpu_energy_per_logical_pass_j": (
                    incremental_j
                    / passes
                ),
            }

            records.append(
                {
                    "repetition": (
                        repetition
                    ),
                    "condition": (
                        condition
                    ),
                    "idle": idle,
                    "workload": (
                        workload
                    ),
                    "accounting": (
                        accounting
                    ),
                }
            )

            print(
                "    "
                f"{condition}: "
                f"inv/pass={selected.size}, "
                f"passes={passes}, "
                f"gross/pass="
                f"{accounting['gross_gpu_energy_per_logical_pass_j']:.6f} J"
            )

    by_rep: dict[
        int,
        dict[
            str,
            dict[str, Any],
        ],
    ] = {}

    for record in records:
        by_rep.setdefault(
            int(
                record[
                    "repetition"
                ]
            ),
            {},
        )[
            str(
                record[
                    "condition"
                ]
            )
        ] = record

    gross_reductions: list[
        float
    ] = []

    incremental_reductions: list[
        float | None
    ] = []

    for repetition in range(
        1,
        repetitions + 1,
    ):
        baseline = by_rep[
            repetition
        ][
            "always_on_tcn"
        ]

        candidate = by_rep[
            repetition
        ][
            "router_v2"
        ]

        baseline_gross = float(
            baseline[
                "accounting"
            ][
                "gross_gpu_energy_per_logical_pass_j"
            ]
        )

        candidate_gross = float(
            candidate[
                "accounting"
            ][
                "gross_gpu_energy_per_logical_pass_j"
            ]
        )

        gross_reductions.append(
            energy_reduction_fraction(
                baseline_gross,
                candidate_gross,
            )
        )

        baseline_incremental = float(
            baseline[
                "accounting"
            ][
                "incremental_gpu_energy_per_logical_pass_j"
            ]
        )

        candidate_incremental = float(
            candidate[
                "accounting"
            ][
                "incremental_gpu_energy_per_logical_pass_j"
            ]
        )

        incremental_reductions.append(
            energy_reduction_fraction(
                baseline_incremental,
                candidate_incremental,
            )
            if baseline_incremental > 0.0
            else None
        )

    (
        gpu_status,
        mean_gross_reduction,
        positive_each,
    ) = energy_status(
        gross_reductions,
        minimum_mean=float(
            success_cfg[
                "minimum_mean_gross_reduction"
            ]
        ),
        positive_each_required=bool(
            success_cfg[
                "positive_each_of_5_repetitions"
            ]
        ),
    )

    valid_incremental = [
        value
        for value in incremental_reductions
        if value is not None
    ]

    high_status = str(
        evaluation_result[
            "success_criteria"
        ][
            "high"
        ][
            "status"
        ]
    )

    alert_status = str(
        evaluation_result[
            "success_criteria"
        ][
            "alert"
        ][
            "status"
        ]
    )

    final_status = overall_status(
        high_status=high_status,
        alert_status=alert_status,
        gpu_energy_status=gpu_status,
    )

    payload = {
        "schema_version": (
            "aeroxai.cobra_router_v2_final_evaluation.v1"
        ),
        "study": (
            "cobra_router_v2_adaptation"
        ),
        "status": (
            "FINAL_EXTERNAL_EVALUATION_COMPLETE"
        ),
        "implementation_commit": (
            _git_head()
        ),
        "evaluation_commit": (
            REQUIRED_EVALUATION_COMMIT
        ),
        "functional_evaluation": (
            evaluation_result
        ),
        "gpu_energy": {
            "measurement_scope": (
                "gpu_device_tcn_scoring_only"
            ),
            "gpu_device": (
                torch.cuda.get_device_name(
                    GPU_DEVICE_INDEX
                )
            ),
            "cpu_router_energy": (
                "UNKNOWN"
            ),
            "whole_system_energy": (
                "UNKNOWN"
            ),
            "physical_compressor_energy": (
                "NOT_MEASURED"
            ),
            "protocol": {
                "nvml_sampling_interval_seconds": (
                    interval_s
                ),
                "warmup_runs_per_condition": (
                    warmups
                ),
                "measured_repetitions": (
                    repetitions
                ),
                "matched_idle_seconds": (
                    idle_s
                ),
                "minimum_metered_workload_seconds": (
                    minimum_workload_s
                ),
                "cooldown_seconds": (
                    COOLDOWN_SECONDS
                ),
                "condition_order_seed": (
                    CONDITION_ORDER_SEED
                ),
                "condition_orders": (
                    orders
                ),
                "inference_batch_size": (
                    INFERENCE_BATCH_SIZE
                ),
                "always_on_invocations_per_pass": (
                    int(
                        all_indices.size
                    )
                ),
                "router_v2_invocations_per_pass": (
                    int(
                        routed_indices.size
                    )
                ),
            },
            "records": records,
            "gross_reduction_by_repetition": (
                gross_reductions
            ),
            "mean_gross_reduction_fraction": (
                mean_gross_reduction
            ),
            "positive_each_repetition": (
                positive_each
            ),
            "incremental_reduction_by_repetition": (
                incremental_reductions
            ),
            "mean_incremental_reduction_fraction": (
                float(
                    statistics.fmean(
                        valid_incremental
                    )
                )
                if valid_incremental
                else None
            ),
            "success_criterion": {
                "minimum_mean_gross_reduction": (
                    float(
                        success_cfg[
                            "minimum_mean_gross_reduction"
                        ]
                    )
                ),
                "positive_each_of_5_repetitions": (
                    bool(
                        success_cfg[
                            "positive_each_of_5_repetitions"
                        ]
                    )
                ),
                "status": (
                    gpu_status
                ),
            },
        },
        "final_status": (
            final_status
        ),
        "claim_boundaries": {
            "causal_fault_diagnosis": (
                False
            ),
            "autonomous_control": (
                False
            ),
            "plant_energy_savings": (
                False
            ),
            "carbon_savings": (
                False
            ),
            "zero_shot_generalization": (
                False
            ),
            "cpu_energy_measured": (
                False
            ),
            "whole_system_energy_measured": (
                False
            ),
        },
        "post_evaluation_tuning_allowed": (
            False
        ),
    }

    FINAL_RESULT_PATH.write_text(
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
                "high_status": (
                    high_status
                ),
                "alert_status": (
                    alert_status
                ),
                "gross_reduction_by_repetition": (
                    gross_reductions
                ),
                "mean_gross_reduction": (
                    mean_gross_reduction
                ),
                "positive_each_repetition": (
                    positive_each
                ),
                "gpu_energy_status": (
                    gpu_status
                ),
                "final_status": (
                    final_status
                ),
                "output": str(
                    FINAL_RESULT_PATH.relative_to(
                        ROOT
                    )
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
