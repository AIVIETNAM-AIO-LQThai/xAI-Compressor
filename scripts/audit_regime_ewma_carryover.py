from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.data.features import model_feature_columns
from ml.detection.alerts import causal_ewma
from ml.detection.regime_pca import (
    assign_regimes,
    fit_regime_pca_detector,
    score_regime_pca_detector,
)
from ml.detection.regime_smoothing_diagnostics import (
    bootstrap_smoothed_threshold,
    conditional_tail_summary,
    gap_aware_autocorrelation,
    tail_composition,
    transition_carryover_table,
    transition_flags,
    transition_pair_summary,
    transition_reset_ewma,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "regime_ewma_carryover_audit.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return payload


def _assert_close(
    name: str,
    observed: float,
    expected: float,
    tolerance: float,
) -> None:
    difference = abs(observed - expected)
    if difference > tolerance:
        raise RuntimeError(
            f"{name} mismatch: observed={observed!r}, "
            f"expected={expected!r}, difference={difference!r}"
        )


def main() -> None:
    config = _load_yaml(CONFIG_PATH)
    dataset = config["dataset"]

    forbidden = {"test_file", "incidents_config", "reported_incidents"}
    overlap = forbidden.intersection(dataset)
    if overlap:
        raise RuntimeError(
            f"Forbidden dataset inputs in audit config: {sorted(overlap)}"
        )

    train = pd.read_parquet(
        ROOT / dataset["train_file"]
    ).sort_index()
    calibration = pd.read_parquet(
        ROOT / dataset["calibration_file"]
    ).sort_index()

    print("Study role:", config["study"]["role"])
    print("Loading TRAIN:", len(train))
    print("Loading CALIBRATION:", len(calibration))
    print("TEST DATA: NOT LOADED")
    print("INCIDENT LABELS: NOT LOADED")

    features = model_feature_columns(train.columns)
    representation = config["representation"]
    router = config["regime_router"]
    smoothing = config["smoothing"]

    detector = fit_regime_pca_detector(
        train,
        features,
        current_feature=str(router["current_feature"]),
        pressure_feature=str(router["pressure_feature"]),
        high_quantile=float(router["high_quantile"]),
        variance_retained=float(representation["variance_retained"]),
    )

    raw_scores = score_regime_pca_detector(
        calibration,
        detector,
    )
    regimes = assign_regimes(
        calibration,
        current_feature=detector.current_feature,
        pressure_feature=detector.pressure_feature,
        current_threshold=detector.current_threshold,
        pressure_threshold=detector.pressure_threshold,
    )

    baseline = causal_ewma(
        raw_scores,
        alpha=float(smoothing["ewma_alpha"]),
        reset_gap_minutes=int(smoothing["reset_gap_minutes"]),
    )
    reset = transition_reset_ewma(
        raw_scores,
        regimes,
        alpha=float(smoothing["ewma_alpha"]),
        reset_gap_minutes=int(smoothing["reset_gap_minutes"]),
    )

    expected = config["candidate_reproduction"]["expected"]
    tolerance = float(config["candidate_reproduction"]["tolerance"])

    exact = {
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "feature_count": len(features),
        "component_count": detector.component_count,
    }
    for name, observed in exact.items():
        if int(observed) != int(expected[name]):
            raise RuntimeError(
                f"{name} mismatch: {observed} != {expected[name]}"
            )

    _assert_close(
        "current_threshold",
        float(detector.current_threshold),
        float(expected["current_threshold"]),
        tolerance,
    )
    _assert_close(
        "pressure_threshold",
        float(detector.pressure_threshold),
        float(expected["pressure_threshold"]),
        tolerance,
    )

    baseline_threshold = float(
        baseline.quantile(
            float(smoothing["threshold_quantile"]),
            interpolation="higher",
        )
    )
    _assert_close(
        "baseline_pooled_threshold",
        baseline_threshold,
        float(expected["baseline_pooled_threshold"]),
        tolerance,
    )

    flags = transition_flags(
        regimes,
        reset_gap_minutes=int(smoothing["reset_gap_minutes"]),
    )
    if int(flags.sum()) != int(expected["transition_count"]):
        raise RuntimeError(
            "transition_count mismatch: "
            f"{int(flags.sum())} != {int(expected['transition_count'])}"
        )

    print("Frozen candidate reproduction: PASS")
    print("Baseline threshold:", baseline_threshold)
    print("Transitions:", int(flags.sum()))

    carryover = transition_carryover_table(
        raw_scores,
        baseline,
        reset,
        regimes,
        reset_gap_minutes=int(smoothing["reset_gap_minutes"]),
    )

    pair_summary = transition_pair_summary(carryover)
    conditional = conditional_tail_summary(
        raw_scores,
        baseline,
        reset,
        regimes,
    )

    tail_quantiles = [
        float(value)
        for value in config["diagnostics"]["tail_quantiles"]
    ]
    tails = {
        "baseline": {
            str(q): tail_composition(
                baseline,
                regimes,
                quantile=q,
            )
            for q in tail_quantiles
        },
        "transition_reset": {
            str(q): tail_composition(
                reset,
                regimes,
                quantile=q,
            )
            for q in tail_quantiles
        },
    }

    lags = [
        int(value)
        for value in config["diagnostics"]["autocorrelation_lags"]
    ]
    autocorr = {
        "baseline": gap_aware_autocorrelation(
            baseline,
            lags=lags,
            reset_gap_minutes=int(smoothing["reset_gap_minutes"]),
        ),
        "transition_reset": gap_aware_autocorrelation(
            reset,
            lags=lags,
            reset_gap_minutes=int(smoothing["reset_gap_minutes"]),
        ),
    }

    bootstrap = config["bootstrap"]
    block_lengths = [
        int(value)
        for value in bootstrap["circular_block_lengths"]
    ]

    baseline_bootstrap, baseline_table = (
        bootstrap_smoothed_threshold(
            baseline,
            regimes,
            threshold_quantile=float(
                smoothing["threshold_quantile"]
            ),
            replicates=int(bootstrap["replicates"]),
            block_lengths=block_lengths,
            seed=int(bootstrap["seed"]),
            confidence=float(bootstrap["confidence"]),
            variant="baseline_gap_reset_only",
        )
    )
    reset_bootstrap, reset_table = (
        bootstrap_smoothed_threshold(
            reset,
            regimes,
            threshold_quantile=float(
                smoothing["threshold_quantile"]
            ),
            replicates=int(bootstrap["replicates"]),
            block_lengths=block_lengths,
            seed=int(bootstrap["seed"]),
            confidence=float(bootstrap["confidence"]),
            variant="diagnostic_reset_on_regime_change",
        )
    )

    result = {
        "study": {
            "role": config["study"]["role"],
            "evidence_class": config["study"]["evidence_class"],
            "posthoc": True,
            "test_loaded": False,
            "incidents_loaded": False,
        },
        "reproduction": {
            "status": "PASS",
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "feature_count": len(features),
            "component_count": int(detector.component_count),
            "current_threshold": float(detector.current_threshold),
            "pressure_threshold": float(detector.pressure_threshold),
            "baseline_threshold": baseline_threshold,
            "transition_count": int(flags.sum()),
        },
        "diagnostics": {
            "transition_pair_carryover": pair_summary,
            "conditional_tails": conditional,
            "tail_composition": tails,
            "autocorrelation": autocorr,
            "bootstrap_threshold_stability": {
                "baseline": baseline_bootstrap,
                "transition_reset": reset_bootstrap,
            },
        },
        "guardrails": {
            "no_test_loaded": True,
            "no_incidents_loaded": True,
            "detector_unchanged": True,
            "transition_reset_is_diagnostic_only": True,
        },
    }

    output_dir = ROOT / config["outputs"]["local_output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    carryover.to_parquet(
        output_dir / "transition_carryover.parquet"
    )
    pd.concat(
        [baseline_table, reset_table],
        ignore_index=True,
    ).to_parquet(
        output_dir / "bootstrap_thresholds.parquet",
        index=False,
    )

    result_path = ROOT / config["outputs"]["result_json"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Audit complete.")
    print("Result:", result_path.relative_to(ROOT))
    print("TEST DATA: NOT LOADED")
    print("INCIDENT LABELS: NOT LOADED")


if __name__ == "__main__":
    main()
