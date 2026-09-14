from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ml.data.features import model_feature_columns
from ml.detection.alerts import causal_ewma
from ml.detection.regime_calibration_diagnostics import (
    bootstrap_threshold_composition,
    conditional_thresholds,
    gap_aware_autocorrelation,
    occupancy_comparison,
    pooled_tail_composition,
    regime_run_summary,
    router_margin_frame,
    score_distribution_by_regime,
    tail_transition_proximity,
    transition_score_diagnostics,
)
from ml.detection.regime_pca import (
    assign_regimes,
    fit_regime_pca_detector,
    score_regime_pca_detector,
)
from ml.detection.regime_pca_benchmark import (
    calibration_threshold_stability,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "regime_calibration_audit.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return payload


def _assert_forbidden_inputs_absent(
    config: dict[str, Any],
) -> None:
    dataset = config["dataset"]
    forbidden = {
        "test_file",
        "incidents_config",
        "reported_incidents",
    }
    present = sorted(forbidden.intersection(dataset))
    if present:
        raise RuntimeError(
            "Calibration audit config contains forbidden "
            f"dataset keys: {present}"
        )


def _assert_close(
    name: str,
    observed: float,
    expected: float,
    tolerance: float,
) -> None:
    difference = abs(observed - expected)
    if difference > tolerance:
        raise RuntimeError(
            f"Reproduction mismatch for {name}: "
            f"observed={observed!r}, expected={expected!r}, "
            f"difference={difference!r}, tolerance={tolerance!r}"
        )


def _assert_candidate_reproduction(
    *,
    config: dict[str, Any],
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    features: list[str],
    detector: Any,
    observed_threshold: float,
) -> None:
    reproduction = config["candidate_reproduction"]
    expected = reproduction["expected"]
    tolerance = float(reproduction["tolerance"])

    exact_checks = {
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "feature_count": len(features),
        "component_count": detector.component_count,
    }

    for key, observed in exact_checks.items():
        expected_value = int(expected[key])
        if int(observed) != expected_value:
            raise RuntimeError(
                f"Reproduction mismatch for {key}: "
                f"observed={observed}, expected={expected_value}"
            )

    _assert_close(
        "global_explained_variance_ratio",
        float(detector.global_explained_variance_ratio),
        float(expected["global_explained_variance_ratio"]),
        tolerance,
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
    _assert_close(
        "pooled_calibration_threshold",
        observed_threshold,
        float(expected["pooled_calibration_threshold"]),
        tolerance,
    )

    expected_counts = {
        str(key): int(value)
        for key, value in expected["train_regime_counts"].items()
    }
    observed_counts = {
        str(key): int(value)
        for key, value in detector.train_regime_counts.items()
    }
    if observed_counts != expected_counts:
        raise RuntimeError(
            "Train regime-count reproduction mismatch:\n"
            f"observed={observed_counts}\n"
            f"expected={expected_counts}"
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def main() -> None:
    config = _load_yaml(CONFIG_PATH)
    _assert_forbidden_inputs_absent(config)

    dataset = config["dataset"]

    train_path = ROOT / dataset["train_file"]
    calibration_path = ROOT / dataset["calibration_file"]

    print("Study role:", config["study"]["role"])
    print("Evidence class:", config["study"]["evidence_class"])
    print()
    print("Loading TRAIN:", train_path.relative_to(ROOT))
    train = pd.read_parquet(train_path).sort_index()
    print("Rows:", len(train))
    print()
    print("Loading CALIBRATION:", calibration_path.relative_to(ROOT))
    calibration = pd.read_parquet(calibration_path).sort_index()
    print("Rows:", len(calibration))
    print()
    print("TEST DATA: NOT LOADED")
    print("INCIDENT LABELS: NOT LOADED")
    print()

    features = model_feature_columns(train.columns)
    print("Features:", len(features))

    representation = config["representation"]
    router = config["regime_router"]
    alerting = config["alerting"]
    stability_config = config["calibration_stability"]
    diagnostics_config = config["diagnostics"]

    print()
    print("Reconstructing frozen regime Standard-PCA candidate...")

    detector = fit_regime_pca_detector(
        train,
        features,
        current_feature=str(router["current_feature"]),
        pressure_feature=str(router["pressure_feature"]),
        high_quantile=float(router["high_quantile"]),
        variance_retained=float(representation["variance_retained"]),
    )

    calibration_scores = score_regime_pca_detector(
        calibration,
        detector,
    )

    smoothed_scores = causal_ewma(
        calibration_scores,
        alpha=float(alerting["ewma_alpha"]),
        reset_gap_minutes=int(alerting["reset_gap_minutes"]),
    )

    observed_threshold = float(
        smoothed_scores.quantile(
            float(alerting["threshold_quantile"]),
            interpolation="higher",
        )
    )

    print("Component count:", detector.component_count)
    print("Current threshold:", detector.current_threshold)
    print("Pressure threshold:", detector.pressure_threshold)

    _assert_candidate_reproduction(
        config=config,
        train=train,
        calibration=calibration,
        features=features,
        detector=detector,
        observed_threshold=observed_threshold,
    )
    print("Candidate reproduction: PASS")
    print()
    print("Observed pooled calibration threshold:", observed_threshold)

    stability = calibration_threshold_stability(
        calibration_scores,
        protocol=type(
            "AuditProtocol",
            (),
            {
                "ewma_alpha": float(alerting["ewma_alpha"]),
                "reset_gap_minutes": int(alerting["reset_gap_minutes"]),
                "threshold_quantile": float(
                    alerting["threshold_quantile"]
                ),
            },
        )(),
        bootstrap_replicates=int(
            stability_config["bootstrap_replicates"]
        ),
        block_lengths=[
            int(value)
            for value in stability_config["circular_block_lengths"]
        ],
        seed=int(stability_config["seed"]),
        confidence=float(stability_config["confidence"]),
    )

    expected_widths = config["candidate_reproduction"]["expected"][
        "calibration_relative_width"
    ]
    stability_tolerance = float(
        config["candidate_reproduction"]["stability_tolerance"]
    )

    for mode, expected_width in expected_widths.items():
        observed_width = stability["modes"][mode][
            "relative_95_width"
        ]
        _assert_close(
            f"{mode}.relative_95_width",
            float(observed_width),
            float(expected_width),
            stability_tolerance,
        )

    print("Pooled calibration stability reproduction: PASS")
    print()

    train_regimes = assign_regimes(
        train,
        current_feature=detector.current_feature,
        pressure_feature=detector.pressure_feature,
        current_threshold=detector.current_threshold,
        pressure_threshold=detector.pressure_threshold,
    )
    calibration_regimes = assign_regimes(
        calibration,
        current_feature=detector.current_feature,
        pressure_feature=detector.pressure_feature,
        current_threshold=detector.current_threshold,
        pressure_threshold=detector.pressure_threshold,
    )

    print("Analyzing regime occupancy...")
    occupancy = occupancy_comparison(
        train_regimes,
        calibration_regimes,
    )

    print("Analyzing regime score distributions...")
    raw_distributions = score_distribution_by_regime(
        calibration_scores,
        calibration_regimes,
    )
    ewma_distributions = score_distribution_by_regime(
        smoothed_scores,
        calibration_regimes,
    )

    tail_quantiles = [
        float(value)
        for value in diagnostics_config["tail_quantiles"]
    ]

    print("Analyzing upper-tail composition...")
    tail_composition = pooled_tail_composition(
        smoothed_scores,
        calibration_regimes,
        quantiles=tail_quantiles,
    )

    margins = router_margin_frame(
        calibration,
        train,
        detector,
    )

    print("Analyzing router transitions...")
    (
        transition_diagnostics,
        transition_windows,
        transitions,
        near1,
        near2,
    ) = transition_score_diagnostics(
        calibration_scores,
        smoothed_scores,
        calibration_regimes,
        reset_gap_minutes=int(alerting["reset_gap_minutes"]),
        window_rows=int(
            diagnostics_config["transition_window_rows"]
        ),
    )

    transition_tail = tail_transition_proximity(
        smoothed_scores,
        transitions,
        near1,
        near2,
        quantiles=tail_quantiles,
    )

    print("Analyzing regime runs and temporal dependence...")
    run_summary = regime_run_summary(
        calibration_regimes,
        reset_gap_minutes=int(alerting["reset_gap_minutes"]),
    )

    autocorrelation = gap_aware_autocorrelation(
        smoothed_scores,
        lags=[
            int(value)
            for value in diagnostics_config["autocorrelation_lags"]
        ],
        reset_gap_minutes=int(alerting["reset_gap_minutes"]),
        regimes=calibration_regimes,
    )

    if len(calibration.index) >= 2:
        observed_span_days = float(
            (calibration.index.max() - calibration.index.min())
            / pd.Timedelta(days=1)
        )
    else:
        observed_span_days = 0.0

    transition_diagnostics["observed_span_days"] = observed_span_days
    transition_diagnostics["transitions_per_observed_day"] = (
        float(transition_diagnostics["transition_count"] / observed_span_days)
        if observed_span_days > 0.0
        else None
    )

    print("Analyzing bootstrap composition...")
    (
        bootstrap_composition,
        bootstrap_table,
    ) = bootstrap_threshold_composition(
        smoothed_scores,
        calibration_regimes,
        threshold_quantile=float(alerting["threshold_quantile"]),
        bootstrap_replicates=int(
            stability_config["bootstrap_replicates"]
        ),
        block_lengths=[
            int(value)
            for value in stability_config["circular_block_lengths"]
        ],
        seed=int(stability_config["seed"]),
        confidence=float(stability_config["confidence"]),
    )

    print("Calculating diagnostic conditional thresholds...")
    conditional = conditional_thresholds(
        smoothed_scores,
        calibration_regimes,
        threshold_quantile=float(alerting["threshold_quantile"]),
    )

    q99_cutoff = float(
        smoothed_scores.quantile(0.99, interpolation="higher")
    )
    q995_cutoff = float(
        smoothed_scores.quantile(0.995, interpolation="higher")
    )

    calibration_points = pd.concat(
        [
            calibration_scores.rename("raw_score"),
            smoothed_scores.rename("ewma_score"),
            calibration_regimes.rename("regime"),
            margins,
            transitions.rename("is_transition"),
            near1.rename("near_transition_1"),
            near2.rename("near_transition_2"),
        ],
        axis=1,
    )
    calibration_points["is_q99_tail"] = (
        calibration_points["ewma_score"] >= q99_cutoff
    )
    calibration_points["is_q995_tail"] = (
        calibration_points["ewma_score"] >= q995_cutoff
    )

    output_dir = ROOT / config["outputs"]["local_output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_points.to_parquet(
        output_dir / "calibration_points.parquet"
    )
    transition_windows.to_parquet(
        output_dir / "transition_windows.parquet",
        index=False,
    )
    bootstrap_table.to_parquet(
        output_dir / "bootstrap_thresholds.parquet",
        index=False,
    )

    payload = {
        "study": {
            "role": config["study"]["role"],
            "evidence_class": config["study"]["evidence_class"],
            "posthoc": True,
            "production_promotion_allowed": False,
            "data_boundary": {
                "train_loaded": True,
                "calibration_loaded": True,
                "test_loaded": False,
                "incident_labels_loaded": False,
            },
        },
        "data": {
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "feature_count": len(features),
        },
        "candidate_reproduction": {
            "status": "PASS",
            "component_count": int(detector.component_count),
            "global_explained_variance_ratio": float(
                detector.global_explained_variance_ratio
            ),
            "current_threshold": float(detector.current_threshold),
            "pressure_threshold": float(detector.pressure_threshold),
            "pooled_calibration_threshold": observed_threshold,
            "train_regime_counts": {
                key: int(value)
                for key, value in detector.train_regime_counts.items()
            },
            "calibration_stability": stability,
        },
        "diagnostics": {
            "occupancy": occupancy,
            "raw_score_distributions": raw_distributions,
            "ewma_score_distributions": ewma_distributions,
            "pooled_tail_composition": tail_composition,
            "router_transition": transition_diagnostics,
            "tail_transition_proximity": transition_tail,
            "regime_runs": run_summary,
            "autocorrelation": autocorrelation,
            "bootstrap_mixture_sensitivity": bootstrap_composition,
            "conditional_thresholds_diagnostic_only": conditional,
        },
        "guardrails": {
            "no_test_loaded": True,
            "no_incidents_loaded": True,
            "no_detector_parameters_changed": True,
            "conditional_thresholds_are_diagnostic_only": True,
        },
    }

    result_path = ROOT / config["outputs"]["result_json"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Audit complete.")
    print("Result:", result_path.relative_to(ROOT))
    print("Local tables:", output_dir.relative_to(ROOT))
    print("TEST DATA: NOT LOADED")
    print("INCIDENT LABELS: NOT LOADED")


if __name__ == "__main__":
    main()
