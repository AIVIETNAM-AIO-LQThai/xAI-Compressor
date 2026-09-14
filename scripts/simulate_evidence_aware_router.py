from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ml.data.metropt2_features import (
    model_feature_columns as metropt2_feature_columns,
)
from ml.detection.alerts import causal_ewma
from ml.detection.common import fit_scaler
from ml.detection.pca_detector import (
    fit_pca_detector,
    score_pca_detector,
)
from ml.energy.evidence_aware_routing import (
    ROUTER_FEATURE_NAMES,
    chronological_split,
)
from ml.energy.evidence_aware_simulation import (
    benchmark_frozen_decision,
    counterfactual_mixture,
    fixed_reference_router_features,
    frozen_router_probability,
    location_shift,
    monotonic_nondecreasing,
    operating_regimes,
    route_summary,
    scale_shift,
    tail_inflation,
)
from ml.energy.support_aware_routing import (
    feature_support_statistic,
    pca_latent_support_statistic,
)
from ml.temporal.detector import prepare_temporal_batch

ROOT = Path(__file__).resolve().parents[1]
STUDY_CONFIG_PATH = (
    ROOT / "configs/evidence_aware_compute_routing.yaml"
)
SIM_CONFIG_PATH = (
    ROOT / "configs/evidence_aware_router_simulation.yaml"
)
DEVELOPMENT_PATH = (
    ROOT / "docs/research/evidence_aware_router_development.json"
)
PREFLIGHT_PATH = (
    ROOT / "docs/research/evidence_aware_router_preflight.json"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return payload


def _load_partition(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(frame.index, name="timestamp")
    return frame.sort_index()


def _feature_list(
    name: str,
    train: pd.DataFrame,
    data_cfg: dict[str, Any],
    primary_features: list[str],
) -> list[str]:
    if name == "metropt3":
        if any(feature not in train.columns for feature in primary_features):
            raise RuntimeError("MetroPT-3 feature identity failed.")
        return list(primary_features)

    if data_cfg["feature_variant"] != "flowmeter_excluded":
        raise RuntimeError("MetroPT2 variant changed.")

    features = metropt2_feature_columns(
        train.columns,
        excluded_groups=["flowmeter"],
    )
    if features != primary_features:
        raise RuntimeError("MetroPT2 feature identity changed.")
    return features


def _shift_features(
    features: list[str],
    prefixes: list[str],
) -> list[str]:
    selected = [
        feature
        for feature in features
        if any(feature.startswith(prefix) for prefix in prefixes)
    ]
    if not selected:
        raise RuntimeError("Synthetic shift feature scope is empty.")
    return selected


def _signals(
    frame: pd.DataFrame,
    *,
    pca: Any,
    support_scaler: Any,
    features: list[str],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    pca_raw = score_pca_detector(frame, pca)
    pca_ewma = causal_ewma(
        pca_raw,
        alpha=0.20,
        reset_gap_minutes=10,
    )
    feature_support = feature_support_statistic(
        frame,
        features=features,
        scaler=support_scaler,
        quantile=0.95,
    )
    latent_support = pca_latent_support_statistic(
        frame,
        detector=pca,
    )
    return pca_ewma, feature_support, latent_support


def _router_probability(
    feature_frame: pd.DataFrame,
    model: dict[str, Any],
) -> np.ndarray:
    return frozen_router_probability(
        feature_frame,
        feature_names=list(model["feature_names"]),
        scaler_mean=np.asarray(model["feature_scaler"]["mean"], dtype=float),
        scaler_scale=np.asarray(model["feature_scaler"]["scale"], dtype=float),
        coef=np.asarray(model["logistic"]["coef"][0], dtype=float),
        intercept=float(model["logistic"]["intercept"][0]),
    )


def main() -> None:
    study = _load_yaml(STUDY_CONFIG_PATH)
    simulation = _load_yaml(SIM_CONFIG_PATH)
    development = _load_json(DEVELOPMENT_PATH)
    preflight = _load_json(PREFLIGHT_PATH)

    freeze_commit = str(simulation["study"]["policy_freeze_commit"])
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    if head != freeze_commit:
        raise RuntimeError(
            f"Simulation must start from frozen policy commit {freeze_commit}; "
            f"current HEAD is {head}."
        )

    if not preflight["ready_for_simulated_mechanism_tests"]:
        raise RuntimeError("Frozen policy is not cleared for simulation.")
    if preflight["ready_for_primary_evaluation"]:
        raise RuntimeError("Simulation stage must precede primary evaluation.")

    selected = preflight["selected_policy"]
    model = preflight["selected_model"]

    frozen = simulation["frozen_policy"]
    if float(selected["C"]) != float(frozen["C"]):
        raise RuntimeError("Frozen C mismatch.")
    if selected["class_weight"] != frozen["class_weight"]:
        raise RuntimeError("Frozen class_weight mismatch.")
    if float(selected["probability_threshold"]) != float(
        frozen["probability_threshold"]
    ):
        raise RuntimeError("Frozen probability threshold mismatch.")
    if model["feature_names"] != ROUTER_FEATURE_NAMES:
        raise RuntimeError("Frozen router feature identity mismatch.")

    dev_cfg = study["development_data"]
    reference = _load_json(
        ROOT / dev_cfg["metropt3"]["feature_reference"]
    )
    primary_features = [str(x) for x in reference["features"]["names"]]

    split_fraction = float(study["development_split"]["router_fit_fraction"])
    rolling_bins = int(study["cheap_router_features"]["rolling_history_bins"])
    threshold = float(selected["probability_threshold"])

    dataset_results = {}
    pooled_baseline_features = []

    for name in ("metropt3", "metropt2"):
        data_cfg = dev_cfg[name]
        train = _load_partition(ROOT / data_cfg["train_file"])
        calibration = _load_partition(ROOT / data_cfg["calibration_file"])
        features = _feature_list(name, train, data_cfg, primary_features)

        pca = fit_pca_detector(
            train,
            features,
            variance_retained=0.95,
            scaler_name="robust",
        )
        support_scaler = fit_scaler(
            train,
            features,
            method="standard",
        )

        sequence_batch = prepare_temporal_batch(
            calibration,
            features=features,
            scaler=support_scaler,
            sequence_length=12,
            bin_minutes=5,
        )
        target_index = pd.DatetimeIndex(sequence_batch.target_index)
        split = chronological_split(
            target_index,
            fit_fraction=split_fraction,
        )
        validation_index = split.validation_index

        references = development["development_datasets"][name][
            "feature_references"
        ]

        baseline_signals = _signals(
            calibration,
            pca=pca,
            support_scaler=support_scaler,
            features=features,
        )
        baseline_features = fixed_reference_router_features(
            pca_ewma=baseline_signals[0],
            feature_support_raw=baseline_signals[1],
            latent_support_raw=baseline_signals[2],
            target_index=target_index,
            references=references,
            rolling_history_bins=rolling_bins,
        )
        baseline_validation = baseline_features.reindex(validation_index)
        baseline_probabilities = _router_probability(
            baseline_validation,
            model,
        )
        baseline_summary = route_summary(
            baseline_probabilities,
            threshold=threshold,
        )

        expected_fraction = float(
            selected["datasets"][name]["tcn_invocation_fraction"]
        )
        if not np.isclose(
            baseline_summary["tcn_invocation_fraction"],
            expected_fraction,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"{name} frozen baseline reproduction failed: "
                f"{baseline_summary['tcn_invocation_fraction']} != "
                f"{expected_fraction}"
            )

        pooled_baseline_features.append(baseline_validation)

        shift_columns = _shift_features(
            features,
            [str(x) for x in simulation["shift_feature_scope"]["prefixes"]],
        )

        stress = simulation["stress_tests"]
        result = {
            "baseline": baseline_summary,
            "baseline_reproduction_passed": True,
            "shift_feature_count": len(shift_columns),
            "feature_location_shift": [],
            "feature_scale_shift": [],
            "support_tail_inflation": [],
            "operating_mix_shift": {},
        }

        location_rates = []
        for severity in stress["feature_location_shift"]["severity_sigma"]:
            shifted = location_shift(
                calibration,
                train,
                features=shift_columns,
                severity_sigma=float(severity),
            )
            signals = _signals(
                shifted,
                pca=pca,
                support_scaler=support_scaler,
                features=features,
            )
            router_features = fixed_reference_router_features(
                pca_ewma=signals[0],
                feature_support_raw=signals[1],
                latent_support_raw=signals[2],
                target_index=target_index,
                references=references,
                rolling_history_bins=rolling_bins,
            ).reindex(validation_index)
            summary = route_summary(
                _router_probability(router_features, model),
                threshold=threshold,
            )
            summary["severity_sigma"] = float(severity)
            summary["delta_invocation_fraction_vs_baseline"] = float(
                summary["tcn_invocation_fraction"]
                - baseline_summary["tcn_invocation_fraction"]
            )
            result["feature_location_shift"].append(summary)
            location_rates.append(float(summary["tcn_invocation_fraction"]))

        scale_rates = []
        for factor in stress["feature_scale_shift"]["factors"]:
            shifted = scale_shift(
                calibration,
                train,
                features=shift_columns,
                factor=float(factor),
            )
            signals = _signals(
                shifted,
                pca=pca,
                support_scaler=support_scaler,
                features=features,
            )
            router_features = fixed_reference_router_features(
                pca_ewma=signals[0],
                feature_support_raw=signals[1],
                latent_support_raw=signals[2],
                target_index=target_index,
                references=references,
                rolling_history_bins=rolling_bins,
            ).reindex(validation_index)
            summary = route_summary(
                _router_probability(router_features, model),
                threshold=threshold,
            )
            summary["factor"] = float(factor)
            summary["delta_invocation_fraction_vs_baseline"] = float(
                summary["tcn_invocation_fraction"]
                - baseline_summary["tcn_invocation_fraction"]
            )
            result["feature_scale_shift"].append(summary)
            scale_rates.append(float(summary["tcn_invocation_fraction"]))

        tail_cfg = stress["support_tail_inflation"]
        tail_reference = np.asarray(
            references["feature_support_reference"],
            dtype=float,
        )
        tail_threshold = float(
            pd.Series(tail_reference).quantile(
                float(tail_cfg["tail_quantile"]),
                interpolation="higher",
            )
        )
        tail_mask = baseline_signals[1] >= tail_threshold
        tail_rates = []

        for factor in tail_cfg["factors"]:
            shifted = tail_inflation(
                calibration,
                train,
                features=shift_columns,
                tail_mask=tail_mask,
                factor=float(factor),
            )
            signals = _signals(
                shifted,
                pca=pca,
                support_scaler=support_scaler,
                features=features,
            )
            router_features = fixed_reference_router_features(
                pca_ewma=signals[0],
                feature_support_raw=signals[1],
                latent_support_raw=signals[2],
                target_index=target_index,
                references=references,
                rolling_history_bins=rolling_bins,
            ).reindex(validation_index)
            summary = route_summary(
                _router_probability(router_features, model),
                threshold=threshold,
            )
            summary["factor"] = float(factor)
            summary["tail_threshold"] = tail_threshold
            summary["affected_calibration_fraction"] = float(tail_mask.mean())
            summary["delta_invocation_fraction_vs_baseline"] = float(
                summary["tcn_invocation_fraction"]
                - baseline_summary["tcn_invocation_fraction"]
            )
            result["support_tail_inflation"].append(summary)
            tail_rates.append(float(summary["tcn_invocation_fraction"]))

        mix_cfg = stress["operating_mix_shift"]
        motor_feature = str(mix_cfg["regime_feature"])
        low_q = float(mix_cfg["train_quantiles"]["low_upper"])
        high_q = float(mix_cfg["train_quantiles"]["high_lower"])
        low_upper = float(
            train[motor_feature].quantile(low_q, interpolation="linear")
        )
        high_lower = float(
            train[motor_feature].quantile(high_q, interpolation="linear")
        )
        regimes = operating_regimes(
            calibration[motor_feature],
            low_upper=low_upper,
            high_lower=high_lower,
        ).reindex(validation_index)
        route_series = pd.Series(
            baseline_probabilities >= threshold,
            index=validation_index,
        )

        observed_weights = {
            regime: float((regimes == regime).mean())
            for regime in ("low", "middle", "high")
        }
        result["operating_mix_shift"]["train_thresholds"] = {
            "low_upper": low_upper,
            "high_lower": high_lower,
        }
        result["operating_mix_shift"]["observed_validation_weights"] = (
            observed_weights
        )
        result["operating_mix_shift"]["scenarios"] = {}

        for scenario_name, weights in mix_cfg["scenarios"].items():
            result["operating_mix_shift"]["scenarios"][scenario_name] = (
                counterfactual_mixture(
                    route=route_series,
                    regimes=regimes,
                    weights={key: float(value) for key, value in weights.items()},
                )
            )

        result["mechanism_checks"] = {
            "feature_location_shift_nondecreasing": monotonic_nondecreasing(
                location_rates
            ),
            "feature_scale_shift_nondecreasing": monotonic_nondecreasing(
                scale_rates
            ),
            "support_tail_inflation_nondecreasing": monotonic_nondecreasing(
                tail_rates
            ),
        }

        dataset_results[name] = result

    pooled = pd.concat(
        pooled_baseline_features,
        axis=0,
        ignore_index=True,
    )
    runtime_cfg = simulation["router_runtime"]
    runtime = benchmark_frozen_decision(
        pooled,
        feature_names=list(model["feature_names"]),
        scaler_mean=np.asarray(model["feature_scaler"]["mean"], dtype=float),
        scaler_scale=np.asarray(model["feature_scaler"]["scale"], dtype=float),
        coef=np.asarray(model["logistic"]["coef"][0], dtype=float),
        intercept=float(model["logistic"]["intercept"][0]),
        probability_threshold=threshold,
        repetitions=int(runtime_cfg["repetitions"]),
        minimum_seconds=float(runtime_cfg["minimum_seconds_per_repetition"]),
    )

    payload = {
        "schema_version": "aeroxai.evidence_aware_router_simulation.v1",
        "study": simulation["study"],
        "data_use": {
            "metropt3_test_opened": False,
            "metropt2_test_opened": False,
            "new_evaluation_data_opened": False,
            "incident_labels_used": False,
        },
        "policy": {
            "C": float(selected["C"]),
            "class_weight": selected["class_weight"],
            "probability_threshold": threshold,
            "refit_performed": False,
            "feature_changes_performed": False,
        },
        "datasets": dataset_results,
        "router_runtime": runtime,
        "all_baseline_reproductions_passed": all(
            bool(value["baseline_reproduction_passed"])
            for value in dataset_results.values()
        ),
        "all_support_shift_mechanism_checks_passed": all(
            all(bool(flag) for flag in value["mechanism_checks"].values())
            for value in dataset_results.values()
        ),
        "interpretation": {
            "evidence_class": "SIMULATED",
            "tcn_evidence_on_synthetic_inputs_evaluated": False,
            "primary_generalization_claim_satisfied": False,
            "router_cpu_energy": "UNKNOWN",
            "whole_system_energy": "UNKNOWN",
            "physical_compressor_energy": "NOT_MEASURED",
        },
        "guardrails": simulation["guardrails"],
    }

    output_path = ROOT / simulation["outputs"]["result_json"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    compact = {
        "policy_freeze_commit": freeze_commit,
        "all_baseline_reproductions_passed": (
            payload["all_baseline_reproductions_passed"]
        ),
        "all_support_shift_mechanism_checks_passed": (
            payload["all_support_shift_mechanism_checks_passed"]
        ),
        "datasets": {
            name: {
                "baseline_invocation_fraction": value["baseline"][
                    "tcn_invocation_fraction"
                ],
                "location_invocation_fractions": [
                    item["tcn_invocation_fraction"]
                    for item in value["feature_location_shift"]
                ],
                "scale_invocation_fractions": [
                    item["tcn_invocation_fraction"]
                    for item in value["feature_scale_shift"]
                ],
                "tail_invocation_fractions": [
                    item["tcn_invocation_fraction"]
                    for item in value["support_tail_inflation"]
                ],
                "mechanism_checks": value["mechanism_checks"],
                "operating_mix_counterfactuals": {
                    scenario: values["counterfactual_invocation_fraction"]
                    for scenario, values
                    in value["operating_mix_shift"]["scenarios"].items()
                },
            }
            for name, value in dataset_results.items()
        },
        "router_runtime_mean_microseconds_per_row": runtime[
            "mean_microseconds_per_row"
        ],
        "router_cpu_energy": "UNKNOWN",
        "output": str(output_path.relative_to(ROOT)),
    }

    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
