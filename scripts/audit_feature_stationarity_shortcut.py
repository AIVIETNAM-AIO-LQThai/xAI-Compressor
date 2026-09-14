from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.data.features import model_feature_columns
from ml.detection.common import transform_frame
from ml.detection.feature_stationarity_diagnostics import (
    analog_min_coherence,
    contribution_share,
    digital_support_summary,
    extreme_run_summary,
    grouped_feature_share,
    incident_window_mask,
    support_and_scaler_summary,
)
from ml.detection.regime_pca import (
    explain_regime_pca_detector,
    fit_regime_pca_detector,
    score_regime_pca_detector,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "feature_stationarity_shortcut_audit.yaml"
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
    focus = config["focus"]
    diagnostics = config["diagnostics"]
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
    calibration_threshold = float(
        calibration_scores.quantile(
            float(diagnostics["high_score_quantile"]),
            interpolation="higher",
        )
    )
    _assert_close(
        "calibration_regime_raw_q995",
        calibration_threshold,
        float(expected["calibration_regime_raw_q995"]),
        tolerance,
    )
    print("Frozen regime-PCA reproduction: PASS")

    relevant_mask = incident_window_mask(
        test.index,
        incidents,
        hours_before=int(
            diagnostics[
                "relevant_window_hours_before_incident"
            ]
        ),
    )
    background_test = test.loc[~relevant_mask]

    explicit_features = [
        str(feature)
        for feature in focus["explicit_features"]
    ]
    missing = [
        feature for feature in explicit_features
        if feature not in features
    ]
    if missing:
        raise RuntimeError(
            f"Focus features missing from model features: {missing}"
        )

    support = support_and_scaler_summary(
        train,
        calibration,
        background_test,
        features=explicit_features,
        all_model_features=features,
        scaler=detector.scaler,
        abs_z_thresholds=[
            float(value)
            for value in diagnostics[
                "standard_abs_z_thresholds"
            ]
        ],
    )

    analog = {}
    for sensor in focus["analog_sensors"]:
        analog[str(sensor)] = {
            "calibration": analog_min_coherence(
                calibration,
                sensor=str(sensor),
                all_model_features=features,
                scaler=detector.scaler,
                min_z_threshold=float(
                    diagnostics[
                        "analog_min_excursion_z_threshold"
                    ]
                ),
                context_z_threshold=float(
                    diagnostics[
                        "analog_context_z_threshold"
                    ]
                ),
            ),
            "background_test": analog_min_coherence(
                background_test,
                sensor=str(sensor),
                all_model_features=features,
                scaler=detector.scaler,
                min_z_threshold=float(
                    diagnostics[
                        "analog_min_excursion_z_threshold"
                    ]
                ),
                context_z_threshold=float(
                    diagnostics[
                        "analog_context_z_threshold"
                    ]
                ),
            ),
        }

    digital = {}
    for sensor in focus["digital_sensors"]:
        digital[str(sensor)] = {
            "train": digital_support_summary(
                train,
                sensor=str(sensor),
            ),
            "calibration": digital_support_summary(
                calibration,
                sensor=str(sensor),
            ),
            "background_test": digital_support_summary(
                background_test,
                sensor=str(sensor),
            ),
        }

    calibration_xai = explain_regime_pca_detector(
        calibration,
        detector,
    )
    test_xai = explain_regime_pca_detector(
        test,
        detector,
    )

    background_contributions = (
        test_xai.contributions.loc[~relevant_mask]
    )
    background_scores = test_xai.score.loc[~relevant_mask]

    contribution_sections = {
        "calibration_all": contribution_share(
            calibration_xai.contributions
        ),
        "background_test_all": contribution_share(
            background_contributions
        ),
        "background_test_high_score": contribution_share(
            background_contributions.loc[
                background_scores >= calibration_threshold
            ]
        ),
    }

    for section in contribution_sections.values():
        section["grouped_focus_share"] = grouped_feature_share(
            section["feature_share"],
            explicit_features=explicit_features,
        )
        top_n = int(diagnostics["top_feature_count"])
        section["top_features"] = list(
            section["feature_share"].keys()
        )[:top_n]

    z_background = transform_frame(
        background_test,
        features,
        detector.scaler,
    )
    z_frame = pd.DataFrame(
        z_background,
        index=background_test.index,
        columns=features,
    )

    temporal = {}
    for feature in explicit_features:
        temporal[feature] = {}
        for threshold in diagnostics[
            "standard_abs_z_thresholds"
        ]:
            value = float(threshold)
            temporal[feature][str(value)] = (
                extreme_run_summary(
                    z_frame[feature],
                    threshold=value,
                    gap_minutes=int(
                        diagnostics["run_gap_minutes"]
                    ),
                )
            )

    output_dir = ROOT / config["outputs"]["local_output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    z_frame.loc[:, explicit_features].to_parquet(
        output_dir / "background_focus_standard_z.parquet"
    )

    payload = {
        "study": config["study"],
        "data": {
            "train_rows": len(train),
            "calibration_rows": len(calibration),
            "test_rows": len(test),
            "background_test_rows": len(background_test),
            "relevant_test_rows": int(relevant_mask.sum()),
            "feature_count": len(features),
            "incident_count": len(incidents),
        },
        "reproduction": {
            "status": "PASS",
            "regime_component_count": detector.component_count,
            "current_threshold": detector.current_threshold,
            "pressure_threshold": detector.pressure_threshold,
            "calibration_regime_raw_q995": calibration_threshold,
        },
        "diagnostics": {
            "support_and_scaler": support,
            "analog_min_coherence": analog,
            "digital_support": digital,
            "feature_residual_share": contribution_sections,
            "temporal_extreme_runs": temporal,
        },
        "guardrails": {
            "posthoc_reused_test": True,
            "no_feature_removed_or_modified": True,
            "no_model_intervention": True,
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
