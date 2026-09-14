from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.data.features import model_feature_columns
from ml.detection.operating_cycle_support_diagnostics import (
    incident_window_mask,
    joint_pattern_summary,
    make_support_flags,
    prevalence_summary,
    quality_summary,
    router_coverage_summary,
    run_summary,
    score_association_summary,
)
from ml.detection.regime_pca import (
    assign_regimes,
    fit_regime_pca_detector,
    score_regime_pca_detector,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs" / "operating_cycle_support_audit.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return payload


def _assert_close(
    name: str,
    observed: float,
    expected: float,
    tolerance: float,
) -> None:
    if abs(observed - expected) > tolerance:
        raise RuntimeError(
            f"{name} mismatch: observed={observed!r}, "
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
    expected = config["reproduction"]["expected"]
    tolerance = float(
        config["reproduction"]["tolerance"]
    )

    exact = {
        "train_rows": len(train),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "feature_count": len(features),
    }
    for name, observed in exact.items():
        if int(observed) != int(expected[name]):
            raise RuntimeError(
                f"{name} mismatch: {observed} != {expected[name]}"
            )

    print("Study role:", config["study"]["role"])
    print("Rows:", exact)
    print("Incidents:", len(incidents))

    router = config["regime_router"]
    detector = fit_regime_pca_detector(
        train,
        features,
        current_feature=str(router["current_feature"]),
        pressure_feature=str(router["pressure_feature"]),
        high_quantile=float(router["high_quantile"]),
        variance_retained=float(
            config["representation"]["variance_retained"]
        ),
    )

    if detector.component_count != int(
        expected["regime_component_count"]
    ):
        raise RuntimeError(
            "Regime component-count reproduction failed."
        )
    _assert_close(
        "current_threshold",
        detector.current_threshold,
        float(expected["current_threshold"]),
        tolerance,
    )
    _assert_close(
        "pressure_threshold",
        detector.pressure_threshold,
        float(expected["pressure_threshold"]),
        tolerance,
    )

    calibration_scores = score_regime_pca_detector(
        calibration,
        detector,
    )
    test_scores = score_regime_pca_detector(
        test,
        detector,
    )
    threshold = float(
        calibration_scores.quantile(
            float(config["diagnostics"]["high_score_quantile"]),
            interpolation="higher",
        )
    )
    _assert_close(
        "calibration_regime_raw_q995",
        threshold,
        float(expected["calibration_regime_raw_q995"]),
        tolerance,
    )
    print("Frozen regime-PCA reproduction: PASS")

    relevant_mask = incident_window_mask(
        test.index,
        incidents,
        hours_before=int(
            config["diagnostics"][
                "relevant_window_hours_before_incident"
            ]
        ),
    )
    background_test = test.loc[~relevant_mask]
    relevant_test = test.loc[relevant_mask]
    background_scores = test_scores.loc[~relevant_mask]

    test_regimes = assign_regimes(
        test,
        current_feature=detector.current_feature,
        pressure_feature=detector.pressure_feature,
        current_threshold=detector.current_threshold,
        pressure_threshold=detector.pressure_threshold,
    )
    background_regimes = test_regimes.loc[~relevant_mask]

    atomic_config = config["support_flags"]
    composite_config = config["composite_flags"]

    calibration_flags = make_support_flags(
        calibration,
        train,
        flag_config=atomic_config,
        composite_config=composite_config,
    )
    background_flags = make_support_flags(
        background_test,
        train,
        flag_config=atomic_config,
        composite_config=composite_config,
    )
    relevant_flags = make_support_flags(
        relevant_test,
        train,
        flag_config=atomic_config,
        composite_config=composite_config,
    )

    prevalence = {
        "calibration": prevalence_summary(
            calibration_flags
        ),
        "background_test": prevalence_summary(
            background_flags
        ),
        "relevant_test": prevalence_summary(
            relevant_flags
        ),
    }

    router_coverage = router_coverage_summary(
        background_flags,
        background_regimes,
    )

    score_association = score_association_summary(
        background_flags,
        background_scores,
        threshold=threshold,
    )

    atomic_names = list(atomic_config.keys())
    joint_patterns = joint_pattern_summary(
        background_flags[atomic_names],
        background_scores,
        background_regimes,
        threshold=threshold,
        top_n=int(
            config["diagnostics"]["top_joint_patterns"]
        ),
    )

    temporal = {
        column: run_summary(
            background_flags[column],
            gap_minutes=int(
                config["diagnostics"]["run_gap_minutes"]
            ),
        )
        for column in background_flags.columns
    }

    quality = {
        "train": quality_summary(train),
        "calibration": quality_summary(calibration),
        "background_test": quality_summary(
            background_test
        ),
        "background_cycle_novel": quality_summary(
            background_test.loc[
                background_flags["any_cycle_novel"]
            ]
        ),
        "background_cycle_supported": quality_summary(
            background_test.loc[
                ~background_flags["any_cycle_novel"]
            ]
        ),
    }

    output_dir = ROOT / config["outputs"]["local_output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    local_table = pd.concat(
        [
            background_flags,
            background_scores.rename("raw_score"),
            background_regimes.rename("regime"),
        ],
        axis=1,
    )
    local_table.to_parquet(
        output_dir / "background_cycle_support.parquet"
    )

    payload = {
        "study": config["study"],
        "data": {
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "test_rows": len(test),
            "background_test_rows": len(background_test),
            "relevant_test_rows": len(relevant_test),
            "feature_count": len(features),
            "incident_count": len(incidents),
        },
        "reproduction": {
            "status": "PASS",
            "regime_component_count": detector.component_count,
            "current_threshold": detector.current_threshold,
            "pressure_threshold": detector.pressure_threshold,
            "calibration_regime_raw_q995": threshold,
        },
        "train_support_thresholds": {
            name: {
                "feature": spec["feature"],
                "direction": spec["direction"],
                "train_min": float(
                    train[str(spec["feature"])].min()
                ),
                "train_max": float(
                    train[str(spec["feature"])].max()
                ),
            }
            for name, spec in atomic_config.items()
        },
        "diagnostics": {
            "support_prevalence": prevalence,
            "existing_router_coverage": router_coverage,
            "score_association": score_association,
            "joint_support_patterns": joint_patterns,
            "temporal_support_runs": temporal,
            "data_quality": quality,
        },
        "guardrails": {
            "posthoc_reused_test": True,
            "train_only_support_thresholds": True,
            "no_router_intervention": True,
            "no_detector_intervention": True,
            "no_alert_intervention": True,
        },
    }

    result_path = ROOT / config["outputs"]["result_json"]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Audit complete.")
    print("Result:", result_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
