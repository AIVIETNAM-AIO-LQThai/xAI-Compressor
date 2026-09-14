from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.data.features import model_feature_columns
from ml.detection.coverage_conditioned_stationarity_diagnostics import (
    association_summary,
    coverage_strata,
    feature_support_by_quality,
    incident_window_mask,
    joint_patterns_full_coverage,
    make_support_flags,
    matched_transport_summary,
    prevalence_by_stratum,
    quality_novelty_cells,
    sample_count_strata,
)
from ml.detection.regime_pca import (
    assign_regimes,
    fit_regime_pca_detector,
    score_regime_pca_detector,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "coverage_conditioned_stationarity_audit.yaml"
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

    relevant_mask = incident_window_mask(
        test.index,
        incidents,
        hours_before=int(
            config["diagnostics"][
                "relevant_window_hours_before_incident"
            ]
        ),
    )
    background = test.loc[~relevant_mask]
    background_scores = test_scores.loc[~relevant_mask]

    background_regimes = assign_regimes(
        background,
        current_feature=detector.current_feature,
        pressure_feature=detector.pressure_feature,
        current_threshold=detector.current_threshold,
        pressure_threshold=detector.pressure_threshold,
    )

    background_flags = make_support_flags(
        background,
        train,
        flag_config=config["support_flags"],
        composite_config=config["composite_flags"],
    )

    observed_novel = int(
        background_flags["any_cycle_novel"].sum()
    )
    if observed_novel != int(
        expected["background_any_cycle_novel_count"]
    ):
        raise RuntimeError(
            "Background novelty reproduction failed: "
            f"{observed_novel} != "
            f"{expected['background_any_cycle_novel_count']}"
        )

    print("Frozen detector/support reproduction: PASS")

    coverage = coverage_strata(background)
    sample_strata = sample_count_strata(background)

    novelty_by_quality = {
        "coverage": prevalence_by_stratum(
            background_flags,
            coverage,
        ),
        "sample_count": prevalence_by_stratum(
            background_flags,
            sample_strata,
        ),
        "summary": {
            "novel_full_coverage_fraction": float(
                (
                    background.loc[
                        background_flags["any_cycle_novel"],
                        "coverage_ratio",
                    ].astype(float)
                    >= 1.0
                ).mean()
            ),
            "supported_full_coverage_fraction": float(
                (
                    background.loc[
                        ~background_flags["any_cycle_novel"],
                        "coverage_ratio",
                    ].astype(float)
                    >= 1.0
                ).mean()
            ),
            "novel_complete_sample_count_fraction": float(
                (
                    background.loc[
                        background_flags["any_cycle_novel"],
                        "sample_count",
                    ].astype(float)
                    >= 30
                ).mean()
            ),
            "supported_complete_sample_count_fraction": float(
                (
                    background.loc[
                        ~background_flags["any_cycle_novel"],
                        "sample_count",
                    ].astype(float)
                    >= 30
                ).mean()
            ),
        },
    }

    cells = quality_novelty_cells(
        background,
        background_flags,
        background_scores,
        background_regimes,
        threshold=threshold,
    )

    matched_transport = matched_transport_summary(
        calibration_scores,
        calibration,
        background_scores,
        background,
        threshold=threshold,
    )

    focus_features = [
        str(feature)
        for feature in config["diagnostics"]["focus_features"]
    ]
    feature_support = feature_support_by_quality(
        train,
        calibration,
        background,
        features=focus_features,
    )

    associations = association_summary(
        background_flags,
        background,
    )

    atomic_names = list(config["support_flags"].keys())
    joint_patterns = joint_patterns_full_coverage(
        background_flags[atomic_names],
        background,
        background_scores,
        background_regimes,
        threshold=threshold,
        top_n=int(
            config["diagnostics"]["top_joint_patterns"]
        ),
    )

    output_dir = ROOT / config["outputs"]["local_output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    local_table = pd.concat(
        [
            background[
                ["coverage_ratio", "sample_count"]
            ],
            coverage,
            sample_strata,
            background_flags,
            background_scores.rename("raw_score"),
            background_regimes.rename("regime"),
        ],
        axis=1,
    )
    local_table.to_parquet(
        output_dir / "background_quality_conditioned_support.parquet"
    )

    payload = {
        "study": config["study"],
        "data": {
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "test_rows": len(test),
            "background_test_rows": len(background),
            "relevant_test_rows": int(relevant_mask.sum()),
            "feature_count": len(features),
            "incident_count": len(incidents),
        },
        "reproduction": {
            "status": "PASS",
            "regime_component_count": detector.component_count,
            "current_threshold": detector.current_threshold,
            "pressure_threshold": detector.pressure_threshold,
            "calibration_regime_raw_q995": threshold,
            "background_any_cycle_novel_count": observed_novel,
        },
        "diagnostics": {
            "novelty_by_quality": novelty_by_quality,
            "quality_x_novelty_score_cells": cells,
            "matched_quality_score_transport": matched_transport,
            "focus_feature_support_by_quality": feature_support,
            "quality_novelty_association": associations,
            "full_coverage_joint_patterns": joint_patterns,
        },
        "guardrails": {
            "posthoc_reused_test": True,
            "descriptive_conditioning_only": True,
            "no_rows_removed_from_detector": True,
            "no_detector_or_alert_intervention": True,
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
