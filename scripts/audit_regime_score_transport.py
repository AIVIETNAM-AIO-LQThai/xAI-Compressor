from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.data.features import model_feature_columns
from ml.detection.pca_detector import (
    fit_pca_detector,
    score_pca_detector,
)
from ml.detection.regime_pca import (
    assign_regimes,
    explain_regime_pca_detector,
    fit_regime_pca_detector,
    score_regime_pca_detector,
)
from ml.detection.score_transport_diagnostics import (
    contribution_share_summary,
    feature_drift_summary,
    high_score_run_summary,
    incident_window_mask,
    regime_transport_summary,
    transport_summary,
)
from ml.explainability.pca import aggregate_contributions

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT / "configs" / "regime_score_transport_audit.yaml"
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
    variance_retained = float(
        config["representation"]["variance_retained"]
    )
    router = config["regime_router"]
    diagnostics_config = config["diagnostics"]
    reproduction = config["reproduction"]
    expected = reproduction["expected"]
    tolerance = float(reproduction["tolerance"])

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

    robust = fit_pca_detector(
        train,
        features,
        variance_retained=variance_retained,
        scaler_name="robust",
    )
    standard = fit_pca_detector(
        train,
        features,
        variance_retained=variance_retained,
        scaler_name="standard",
    )
    regime = fit_regime_pca_detector(
        train,
        features,
        current_feature=str(router["current_feature"]),
        pressure_feature=str(router["pressure_feature"]),
        high_quantile=float(router["high_quantile"]),
        variance_retained=variance_retained,
    )

    if regime.component_count != int(
        expected["regime_component_count"]
    ):
        raise RuntimeError(
            "Regime PCA component-count reproduction failed."
        )
    _assert_close(
        "current_threshold",
        float(regime.current_threshold),
        float(expected["current_threshold"]),
        tolerance,
    )
    _assert_close(
        "pressure_threshold",
        float(regime.pressure_threshold),
        float(expected["pressure_threshold"]),
        tolerance,
    )
    print("Frozen detector identity reproduction: PASS")

    relevant_mask = incident_window_mask(
        test.index,
        incidents,
        hours_before=int(
            diagnostics_config[
                "relevant_window_hours_before_incident"
            ]
        ),
    )
    background_mask = ~relevant_mask

    background_test = test.loc[background_mask]

    threshold_quantile = float(
        diagnostics_config["raw_score_threshold_quantile"]
    )

    representation_results: dict[str, Any] = {}

    for name, detector in (
        ("global_robust_pca", robust),
        ("global_standard_pca", standard),
    ):
        calibration_scores = score_pca_detector(
            calibration,
            detector,
        )
        all_test_scores = score_pca_detector(
            test,
            detector,
        )

        result = transport_summary(
            calibration_scores,
            all_test_scores,
            threshold_quantile=threshold_quantile,
        )
        threshold = float(
            result["calibration_raw_threshold"]
        )

        result["background_test"] = transport_summary(
            calibration_scores,
            all_test_scores.loc[background_mask],
            threshold_quantile=threshold_quantile,
        )["test"]
        result["relevant_test"] = transport_summary(
            calibration_scores,
            all_test_scores.loc[relevant_mask],
            threshold_quantile=threshold_quantile,
        )["test"]
        result["background_threshold_exceedance_fraction"] = float(
            (
                all_test_scores.loc[background_mask]
                >= threshold
            ).mean()
        )
        result["relevant_threshold_exceedance_fraction"] = float(
            (
                all_test_scores.loc[relevant_mask]
                >= threshold
            ).mean()
        )

        representation_results[name] = result

    calibration_regime_scores = score_regime_pca_detector(
        calibration,
        regime,
    )
    test_regime_scores = score_regime_pca_detector(
        test,
        regime,
    )
    calibration_regimes = assign_regimes(
        calibration,
        current_feature=regime.current_feature,
        pressure_feature=regime.pressure_feature,
        current_threshold=regime.current_threshold,
        pressure_threshold=regime.pressure_threshold,
    )
    test_regimes = assign_regimes(
        test,
        current_feature=regime.current_feature,
        pressure_feature=regime.pressure_feature,
        current_threshold=regime.current_threshold,
        pressure_threshold=regime.pressure_threshold,
    )

    regime_result = transport_summary(
        calibration_regime_scores,
        test_regime_scores,
        threshold_quantile=threshold_quantile,
    )
    regime_raw_threshold = float(
        regime_result["calibration_raw_threshold"]
    )
    regime_result["background_test"] = transport_summary(
        calibration_regime_scores,
        test_regime_scores.loc[background_mask],
        threshold_quantile=threshold_quantile,
    )["test"]
    regime_result["relevant_test"] = transport_summary(
        calibration_regime_scores,
        test_regime_scores.loc[relevant_mask],
        threshold_quantile=threshold_quantile,
    )["test"]
    regime_result["background_threshold_exceedance_fraction"] = float(
        (
            test_regime_scores.loc[background_mask]
            >= regime_raw_threshold
        ).mean()
    )
    regime_result["relevant_threshold_exceedance_fraction"] = float(
        (
            test_regime_scores.loc[relevant_mask]
            >= regime_raw_threshold
        ).mean()
    )
    regime_result["by_regime_all_test"] = regime_transport_summary(
        calibration_regime_scores,
        calibration_regimes,
        test_regime_scores,
        test_regimes,
        threshold_quantile=threshold_quantile,
    )
    regime_result["by_regime_background_test"] = (
        regime_transport_summary(
            calibration_regime_scores,
            calibration_regimes,
            test_regime_scores.loc[background_mask],
            test_regimes.loc[background_mask],
            threshold_quantile=threshold_quantile,
        )
    )
    representation_results["regime_standard_pca"] = regime_result

    print("Raw score transport calculated.")

    feature_drift = feature_drift_summary(
        calibration,
        background_test,
        features=features,
        scaler=regime.scaler,
        abs_z_thresholds=[
            float(value)
            for value in diagnostics_config[
                "feature_abs_z_thresholds"
            ]
        ],
    )
    top_n = int(
        diagnostics_config["top_feature_count"]
    )
    feature_drift[
        "ranked_by_background_test_q99_abs_z"
    ] = feature_drift[
        "ranked_by_background_test_q99_abs_z"
    ][:top_n]
    feature_drift[
        "ranked_by_q99_abs_z_ratio"
    ] = feature_drift[
        "ranked_by_q99_abs_z_ratio"
    ][:top_n]

    calibration_xai = explain_regime_pca_detector(
        calibration,
        regime,
    )
    test_xai = explain_regime_pca_detector(
        test,
        regime,
    )
    calibration_groups = aggregate_contributions(
        calibration_xai.contributions
    )
    test_groups = aggregate_contributions(
        test_xai.contributions
    )

    contribution_transport = {
        "calibration": contribution_share_summary(
            calibration_groups,
            calibration_xai.score,
            threshold=regime_raw_threshold,
        ),
        "background_test": contribution_share_summary(
            test_groups.loc[background_mask],
            test_xai.score.loc[background_mask],
            threshold=regime_raw_threshold,
        ),
        "relevant_test": contribution_share_summary(
            test_groups.loc[relevant_mask],
            test_xai.score.loc[relevant_mask],
            threshold=regime_raw_threshold,
        ),
    }

    run_summary, run_table = high_score_run_summary(
        test_regime_scores.loc[background_mask],
        test_regimes.loc[background_mask],
        threshold=regime_raw_threshold,
        gap_minutes=int(
            diagnostics_config[
                "high_score_run_gap_minutes"
            ]
        ),
    )

    output_dir = ROOT / config["outputs"]["local_output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    run_table.to_parquet(
        output_dir / "background_high_score_runs.parquet",
        index=False,
    )

    payload = {
        "study": config["study"],
        "data": {
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "test_rows": len(test),
            "relevant_test_rows": int(relevant_mask.sum()),
            "background_test_rows": int(background_mask.sum()),
            "feature_count": len(features),
            "incident_count": len(incidents),
        },
        "reproduction": {
            "status": "PASS",
            "regime_component_count": regime.component_count,
            "current_threshold": regime.current_threshold,
            "pressure_threshold": regime.pressure_threshold,
        },
        "diagnostics": {
            "raw_score_transport": representation_results,
            "standard_scaler_feature_drift": feature_drift,
            "regime_pca_contribution_transport": contribution_transport,
            "background_high_score_runs": run_summary,
        },
        "guardrails": {
            "posthoc_reused_test": True,
            "raw_scores_primary": True,
            "no_parameters_changed": True,
            "no_candidate_simulated": True,
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
