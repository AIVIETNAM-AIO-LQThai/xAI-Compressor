from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.data.features import model_feature_columns
from ml.detection.pca_research_benchmark import (
    AlertProtocol,
    benchmark_pca_variant,
)
from ml.detection.regime_pca_benchmark import (
    benchmark_regime_pca_variant,
)
from ml.detection.regime_smoothing_diagnostics import (
    bootstrap_smoothed_threshold,
)
from ml.detection.regime_transition_ewma import (
    benchmark_transition_reset_variant,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "regime_transition_ewma_benchmark.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return payload


def _protocol(config: dict[str, Any]) -> AlertProtocol:
    alerting = config["alerting"]
    evaluation = config["evaluation"]

    return AlertProtocol(
        ewma_alpha=float(alerting["ewma_alpha"]),
        threshold_quantile=float(
            alerting["threshold_quantile"]
        ),
        persistence_hits=int(
            alerting["persistence_hits"]
        ),
        persistence_window=int(
            alerting["persistence_window"]
        ),
        reset_gap_minutes=int(
            alerting["reset_gap_minutes"]
        ),
        merge_minutes=int(
            alerting["merge_minutes"]
        ),
        early_warning_hours=int(
            evaluation["early_warning_hours"]
        ),
        late_tolerance_hours=int(
            evaluation["late_tolerance_hours"]
        ),
        bin_minutes=int(evaluation["bin_minutes"]),
    )


def _assert_metrics(
    result: dict[str, Any],
    expected: dict[str, Any],
    *,
    tolerance: float,
    label: str,
) -> None:
    metrics = result["metrics"]
    failures = []

    for key, expected_value in expected.items():
        observed = float(metrics[key])
        difference = abs(
            observed - float(expected_value)
        )
        if difference > tolerance:
            failures.append(
                {
                    "metric": key,
                    "observed": observed,
                    "expected": float(expected_value),
                    "difference": difference,
                }
            )

    if failures:
        raise RuntimeError(
            f"{label} reproduction failed:\n"
            + json.dumps(failures, indent=2)
        )


def _assert_close(
    name: str,
    observed: float,
    expected: float,
    tolerance: float,
) -> None:
    if abs(observed - expected) > tolerance:
        raise RuntimeError(
            f"{name} reproduction failed: "
            f"observed={observed!r}, "
            f"expected={expected!r}"
        )


def main() -> None:
    config = _load_yaml(CONFIG_PATH)
    dataset = config["dataset"]

    train = pd.read_parquet(
        ROOT / dataset["train_file"]
    ).sort_index()
    calibration = pd.read_parquet(
        ROOT / dataset["calibration_file"]
    ).sort_index()
    test = pd.read_parquet(
        ROOT / dataset["test_file"]
    ).sort_index()

    metropt = _load_yaml(
        ROOT / dataset["incidents_config"]
    )
    incidents = list(
        metropt["dataset"]["reported_incidents"]
    )

    features = model_feature_columns(train.columns)
    protocol = _protocol(config)
    variance_retained = float(
        config["representation"]["variance_retained"]
    )
    router = config["regime_router"]
    reproduction = config["reproduction"]
    tolerance = float(reproduction["tolerance"])

    print("Study role:", config["study"]["role"])
    print(
        "Rows:",
        {
            "train": len(train),
            "calibration": len(calibration),
            "test": len(test),
        },
    )
    print("Features:", len(features))

    print("Reproducing frozen Robust-PCA...")
    robust = benchmark_pca_variant(
        train=train,
        calibration=calibration,
        test=test,
        features=features,
        incidents=incidents,
        protocol=protocol,
        variance_retained=variance_retained,
        scaler_name="robust",
    )
    _assert_metrics(
        robust,
        reproduction["frozen_robust_expected"],
        tolerance=tolerance,
        label="Frozen Robust-PCA",
    )
    print("Frozen Robust-PCA reproduction: PASS")

    print("Reproducing baseline regime Standard-PCA...")
    baseline, _ = benchmark_regime_pca_variant(
        train=train,
        calibration=calibration,
        test=test,
        features=features,
        incidents=incidents,
        protocol=protocol,
        variance_retained=variance_retained,
        current_feature=str(router["current_feature"]),
        pressure_feature=str(router["pressure_feature"]),
        high_quantile=float(router["high_quantile"]),
    )
    _assert_metrics(
        baseline,
        reproduction["baseline_regime_expected"],
        tolerance=tolerance,
        label="Baseline regime Standard-PCA",
    )
    print("Baseline regime Standard-PCA reproduction: PASS")

    print("Benchmarking transition-reset EWMA candidate...")
    (
        candidate,
        candidate_calibration_smoothed,
        candidate_calibration_regimes,
    ) = benchmark_transition_reset_variant(
        train=train,
        calibration=calibration,
        test=test,
        features=features,
        incidents=incidents,
        protocol=protocol,
        variance_retained=variance_retained,
        current_feature=str(router["current_feature"]),
        pressure_feature=str(router["pressure_feature"]),
        high_quantile=float(router["high_quantile"]),
    )

    expected_calibration = reproduction[
        "candidate_calibration_expected"
    ]
    _assert_close(
        "Candidate calibration threshold",
        float(candidate["model"]["threshold"]),
        float(expected_calibration["threshold"]),
        tolerance,
    )

    stability_config = config[
        "calibration_stability"
    ]
    stability, _ = bootstrap_smoothed_threshold(
        candidate_calibration_smoothed,
        candidate_calibration_regimes,
        threshold_quantile=protocol.threshold_quantile,
        replicates=int(
            stability_config["bootstrap_replicates"]
        ),
        block_lengths=[
            int(value)
            for value in stability_config[
                "circular_block_lengths"
            ]
        ],
        seed=int(stability_config["seed"]),
        confidence=float(
            stability_config["confidence"]
        ),
        variant="transition_reset_candidate",
    )

    _assert_close(
        "Candidate block-1 relative width",
        float(
            stability["circular_block_1"][
                "relative_95_width"
            ]
        ),
        float(
            expected_calibration[
                "relative_width_block_1"
            ]
        ),
        1.0e-9,
    )
    _assert_close(
        "Candidate block-12 relative width",
        float(
            stability["circular_block_12"][
                "relative_95_width"
            ]
        ),
        float(
            expected_calibration[
                "relative_width_block_12"
            ]
        ),
        1.0e-9,
    )
    print("Candidate calibration reproduction: PASS")

    metrics = candidate["metrics"]
    gate = config["decision"][
        "exploratory_candidate_gate"
    ]

    checks = {
        "timely_incident_recall": (
            float(metrics["timely_incident_recall"])
            >= float(
                gate[
                    "timely_incident_recall_minimum"
                ]
            )
        ),
        "pre_onset_incident_recall": (
            float(
                metrics[
                    "pre_onset_incident_recall"
                ]
            )
            >= float(
                gate[
                    "pre_onset_incident_recall_minimum"
                ]
            )
        ),
        "false_alerts_per_24h": (
            float(metrics["false_alerts_per_24h"])
            <= float(
                gate[
                    "false_alerts_per_24h_maximum"
                ]
            )
        ),
        "pr_auc": (
            float(metrics["pr_auc"])
            >= float(gate["pr_auc_minimum"])
        ),
        "calibration_relative_width_block12": (
            float(
                stability["circular_block_12"][
                    "relative_95_width"
                ]
            )
            <= float(
                gate[
                    "calibration_relative_width_block12_maximum"
                ]
            )
        ),
    }

    exploratory_gate_pass = all(checks.values())

    payload = {
        "study": config["study"],
        "data": {
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "test_rows": len(test),
            "feature_count": len(features),
            "incident_count": len(incidents),
        },
        "frozen_robust_reference": robust,
        "baseline_regime_reference": baseline,
        "transition_reset_candidate": {
            "result": candidate,
            "calibration_stability": stability,
        },
        "decision": {
            "checks": checks,
            "exploratory_candidate_gate_pass": (
                exploratory_gate_pass
            ),
            "independent_confirmation_required": True,
            "production_promotion_allowed": False,
        },
        "guardrails": {
            "only_candidate_change": (
                "reset EWMA state on operating-regime change"
            ),
            "persistence_reset_on_regime_change": False,
            "no_post_result_tuning": True,
            "test_set_reused": True,
        },
    }

    result_path = ROOT / config["outputs"]["result_json"]
    result_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    result_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Candidate timely recall:",
        metrics["timely_incident_recall"],
    )
    print(
        "Candidate pre-onset recall:",
        metrics["pre_onset_incident_recall"],
    )
    print(
        "Candidate false alerts/day:",
        metrics["false_alerts_per_24h"],
    )
    print("Candidate PR-AUC:", metrics["pr_auc"])
    print(
        "Candidate block-12 relative width:",
        stability["circular_block_12"][
            "relative_95_width"
        ],
    )
    print(
        "Exploratory candidate gate:",
        "PASS"
        if exploratory_gate_pass
        else "FAIL",
    )
    print("Result:", result_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
