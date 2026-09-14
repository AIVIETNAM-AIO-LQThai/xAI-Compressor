from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
from sklearn.preprocessing import StandardScaler

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
    build_router_features,
    chronological_split,
    evidence_coverage,
    fit_logistic_model,
    logistic_candidate_grid,
    route_from_probability,
    route_probabilities,
    select_candidate,
    serialize_logistic_model,
    serialize_standard_scaler,
)
from ml.energy.hierarchical_inference import (
    higher_quantile,
    sha256_file,
)
from ml.energy.support_aware_routing import (
    feature_support_statistic,
    pca_latent_support_statistic,
)
from ml.temporal.detector import (
    TemporalDetector,
    score_temporal_detector,
)
from ml.temporal.tcn import (
    TCNForecaster,
    TemporalForecastConfig,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "evidence_aware_compute_routing.yaml"
)

PREREGISTRATION_COMMIT = (
    "0b3a3135223e764ca0f3a855b1f18d3788db3d5e"
)


def _load_yaml(
    path: Path,
) -> dict[str, Any]:
    payload = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected mapping in {path}."
        )
    return payload


def _load_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected mapping in {path}."
        )
    return payload


def _load_partition(
    path: Path,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(
        frame.index,
        name="timestamp",
    )
    return frame.sort_index()


def _build_tcn_detector(
    *,
    train: pd.DataFrame,
    features: list[str],
    checkpoint_path: Path,
    expected_sha256: str,
) -> tuple[
    TemporalDetector,
    dict[str, Any],
]:
    observed_sha = sha256_file(
        checkpoint_path
    )

    if observed_sha != expected_sha256:
        raise RuntimeError(
            "TCN checkpoint SHA256 mismatch: "
            f"expected {expected_sha256}, "
            f"observed {observed_sha}."
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
            "Frozen TCN checkpoint must use "
            "StandardScaler."
        )

    checkpoint_features = [
        str(value)
        for value in checkpoint["features"]
    ]

    if checkpoint_features != features:
        raise RuntimeError(
            "Frozen TCN feature identity "
            "does not match development data."
        )

    scaler = fit_scaler(
        train,
        features,
        method="standard",
    )

    checkpoint_location = np.asarray(
        checkpoint["scaler_location"],
        dtype=float,
    )
    checkpoint_scale = np.asarray(
        checkpoint["scaler_scale"],
        dtype=float,
    )

    if not np.allclose(
        scaler.mean_,
        checkpoint_location,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Rebuilt TCN scaler mean does not "
            "match frozen checkpoint."
        )

    if not np.allclose(
        scaler.scale_,
        checkpoint_scale,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Rebuilt TCN scaler scale does not "
            "match frozen checkpoint."
        )

    model_config = checkpoint[
        "model_config"
    ]

    model = TCNForecaster(
        TemporalForecastConfig(
            input_dim=int(
                model_config["input_dim"]
            ),
            hidden_dim=int(
                model_config["hidden_dim"]
            ),
            kernel_size=int(
                model_config["kernel_size"]
            ),
            dilations=tuple(
                int(value)
                for value
                in model_config["dilations"]
            ),
            dropout=float(
                model_config["dropout"]
            ),
        )
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    sequence = checkpoint["sequence"]

    detector = TemporalDetector(
        features=features,
        scaler=scaler,
        model=model,
        sequence_length=int(
            sequence["history_bins"]
        ),
        bin_minutes=int(
            sequence["bin_minutes"]
        ),
    )

    return detector, checkpoint


def _feature_list(
    *,
    dataset_name: str,
    train: pd.DataFrame,
    data_config: dict[str, Any],
    primary_features: list[str],
) -> list[str]:
    if dataset_name == "metropt3":
        missing = [
            feature
            for feature in primary_features
            if feature not in train.columns
        ]
        if missing:
            raise RuntimeError(
                "MetroPT-3 feature identity "
                f"failed: {missing}"
            )
        return list(
            primary_features
        )

    if dataset_name != "metropt2":
        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )

    if (
        data_config[
            "feature_variant"
        ]
        != "flowmeter_excluded"
    ):
        raise RuntimeError(
            "MetroPT2 feature variant must "
            "remain flowmeter_excluded."
        )

    features = metropt2_feature_columns(
        train.columns,
        excluded_groups=["flowmeter"],
    )

    if features != primary_features:
        raise RuntimeError(
            "MetroPT2 feature names/order do "
            "not match primary 63-feature "
            "reference."
        )

    if len(features) != int(
        data_config[
            "expected_feature_count"
        ]
    ):
        raise RuntimeError(
            "MetroPT2 feature count changed."
        )

    return features


def _build_dataset(
    *,
    dataset_name: str,
    data_config: dict[str, Any],
    config: dict[str, Any],
    primary_features: list[str],
) -> dict[str, Any]:
    train = _load_partition(
        ROOT / data_config["train_file"]
    )
    calibration = _load_partition(
        ROOT
        / data_config["calibration_file"]
    )

    # No forbidden TEST path is opened here.
    features = _feature_list(
        dataset_name=dataset_name,
        train=train,
        data_config=data_config,
        primary_features=primary_features,
    )

    detector, checkpoint = (
        _build_tcn_detector(
            train=train,
            features=features,
            checkpoint_path=(
                ROOT
                / data_config[
                    "tcn_checkpoint"
                ]
            ),
            expected_sha256=str(
                data_config[
                    "tcn_checkpoint_sha256"
                ]
            ),
        )
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tcn_scores = score_temporal_detector(
        calibration,
        detector,
        device=device,
        batch_size=1024,
    )

    split_cfg = config[
        "development_split"
    ]

    split = chronological_split(
        pd.DatetimeIndex(
            tcn_scores.index
        ),
        fit_fraction=float(
            split_cfg[
                "router_fit_fraction"
            ]
        ),
    )

    fit_scores = tcn_scores.reindex(
        split.fit_index
    )
    validation_scores = (
        tcn_scores.reindex(
            split.validation_index
        )
    )

    teacher = config["teacher"]

    high_threshold = higher_quantile(
        fit_scores,
        float(
            teacher[
                "high_evidence"
            ]["quantile"]
        ),
    )
    alert_threshold = higher_quantile(
        fit_scores,
        float(
            teacher[
                "alert_evidence"
            ]["quantile"]
        ),
    )

    fit_high_target = (
        fit_scores >= high_threshold
    )
    fit_alert_target = (
        fit_scores >= alert_threshold
    )
    validation_high_target = (
        validation_scores
        >= high_threshold
    )
    validation_alert_target = (
        validation_scores
        >= alert_threshold
    )

    pca = fit_pca_detector(
        train,
        features,
        variance_retained=0.95,
        scaler_name="robust",
    )

    pca_raw = score_pca_detector(
        calibration,
        pca,
    )

    pca_ewma = causal_ewma(
        pca_raw,
        alpha=0.20,
        reset_gap_minutes=10,
    )

    support_scaler = fit_scaler(
        train,
        features,
        method="standard",
    )

    feature_support_raw = (
        feature_support_statistic(
            calibration,
            features=features,
            scaler=support_scaler,
            quantile=0.95,
        )
    )

    latent_support_raw = (
        pca_latent_support_statistic(
            calibration,
            detector=pca,
        )
    )

    router_features, references = (
        build_router_features(
            pca_ewma=pca_ewma,
            feature_support_raw=(
                feature_support_raw
            ),
            latent_support_raw=(
                latent_support_raw
            ),
            target_index=pd.DatetimeIndex(
                tcn_scores.index
            ),
            fit_index=split.fit_index,
            rolling_history_bins=int(
                config[
                    "cheap_router_features"
                ][
                    "rolling_history_bins"
                ]
            ),
        )
    )

    if (
        router_features.columns.tolist()
        != ROUTER_FEATURE_NAMES
    ):
        raise RuntimeError(
            "Frozen router feature order "
            "changed."
        )

    fit_features = router_features.reindex(
        split.fit_index
    )
    validation_features = (
        router_features.reindex(
            split.validation_index
        )
    )

    pca_fit_scores = pca_ewma.reindex(
        split.fit_index
    )
    pca_validation_scores = (
        pca_ewma.reindex(
            split.validation_index
        )
    )

    pca_fit_q90 = higher_quantile(
        pca_fit_scores,
        0.90,
    )
    pca_fit_q95 = higher_quantile(
        pca_fit_scores,
        0.95,
    )
    pca_fit_q99 = higher_quantile(
        pca_fit_scores,
        0.99,
    )

    baseline_thresholds = {
        "route_q90": pca_fit_q90,
        "route_q95": pca_fit_q95,
        "route_q99": pca_fit_q99,
    }

    baselines = {}

    for (
        name,
        threshold,
    ) in baseline_thresholds.items():
        route = (
            pca_validation_scores
            >= threshold
        ).to_numpy(dtype=bool)

        baselines[name] = {
            "threshold": float(
                threshold
            ),
            "tcn_invocation_fraction": (
                float(
                    route.mean()
                )
            ),
            "high_evidence_coverage": (
                evidence_coverage(
                    route,
                    validation_high_target.to_numpy(
                        dtype=bool
                    ),
                )
            ),
            "tcn_alert_coverage": (
                evidence_coverage(
                    route,
                    validation_alert_target.to_numpy(
                        dtype=bool
                    ),
                )
            ),
        }

    return {
        "dataset_name": dataset_name,
        "device": device,
        "features": features,
        "checkpoint_sha256": str(
            data_config[
                "tcn_checkpoint_sha256"
            ]
        ),
        "checkpoint_calibration_threshold": (
            float(
                checkpoint[
                    "calibration_threshold"
                ]
            )
        ),
        "train_rows": len(train),
        "calibration_rows": (
            len(calibration)
        ),
        "tcn_target_rows": len(
            tcn_scores
        ),
        "split": {
            "fit_rows": len(
                split.fit_index
            ),
            "validation_rows": len(
                split.validation_index
            ),
            "fit_start": str(
                split.fit_index[0]
            ),
            "fit_end": str(
                split.fit_index[-1]
            ),
            "validation_start": str(
                split.validation_index[0]
            ),
            "validation_end": str(
                split.validation_index[-1]
            ),
        },
        "teacher_thresholds": {
            "high_evidence_q90_fit": (
                high_threshold
            ),
            "alert_evidence_q995_fit": (
                alert_threshold
            ),
        },
        "teacher_counts": {
            "fit_high_evidence_bins": int(
                fit_high_target.sum()
            ),
            "fit_alert_bins": int(
                fit_alert_target.sum()
            ),
            "validation_high_evidence_bins": (
                int(
                    validation_high_target.sum()
                )
            ),
            "validation_alert_bins": int(
                validation_alert_target.sum()
            ),
        },
        "fit_features": fit_features,
        "validation_features": (
            validation_features
        ),
        "fit_target": (
            fit_high_target.astype(int)
        ),
        "validation_high_target": (
            validation_high_target
        ),
        "validation_alert_target": (
            validation_alert_target
        ),
        "feature_references": (
            references
        ),
        "fixed_pca_baselines": (
            baselines
        ),
    }


def _evaluate_candidate(
    *,
    probabilities: dict[str, np.ndarray],
    threshold: float,
    bundles: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    selection = config["selection"]
    constraints = selection[
        "constraints"
    ]
    adequacy = selection[
        "data_adequacy"
    ]

    min_high = float(
        constraints[
            "minimum_high_evidence_coverage_each_validation_dataset"
        ]
    )
    min_alert = float(
        constraints[
            "minimum_tcn_alert_coverage_each_validation_dataset"
        ]
    )
    min_alert_bins = int(
        adequacy[
            "minimum_alert_bins_for_alert_constraint"
        ]
    )

    dataset_results = {}
    invocation_fractions = []
    high_coverages = []
    alert_coverages_for_tie = []
    eligible = True

    for name, bundle in bundles.items():
        route = route_from_probability(
            probabilities[name],
            threshold=threshold,
        )

        high_target = bundle[
            "validation_high_target"
        ].to_numpy(dtype=bool)
        alert_target = bundle[
            "validation_alert_target"
        ].to_numpy(dtype=bool)

        high_coverage = (
            evidence_coverage(
                route,
                high_target,
            )
        )
        alert_coverage = (
            evidence_coverage(
                route,
                alert_target,
            )
        )

        if high_coverage is None:
            raise RuntimeError(
                f"{name} has no validation "
                "high-evidence bins."
            )

        alert_count = int(
            alert_target.sum()
        )

        high_pass = (
            high_coverage >= min_high
        )

        if alert_count >= min_alert_bins:
            alert_pass = (
                alert_coverage is not None
                and alert_coverage
                >= min_alert
            )
            tie_alert = (
                float(alert_coverage)
                if alert_coverage is not None
                else 0.0
            )
        else:
            alert_pass = True
            tie_alert = 1.0

        dataset_eligible = (
            high_pass
            and alert_pass
        )
        eligible = (
            eligible
            and dataset_eligible
        )

        invocation_fraction = float(
            route.mean()
        )
        invocation_fractions.append(
            invocation_fraction
        )
        high_coverages.append(
            float(high_coverage)
        )
        alert_coverages_for_tie.append(
            tie_alert
        )

        dataset_results[name] = {
            "validation_rows": len(
                route
            ),
            "tcn_invocations": int(
                route.sum()
            ),
            "tcn_invocation_fraction": (
                invocation_fraction
            ),
            "high_evidence_bins": int(
                high_target.sum()
            ),
            "high_evidence_coverage": (
                high_coverage
            ),
            "tcn_alert_bins": alert_count,
            "tcn_alert_coverage": (
                alert_coverage
            ),
            "alert_constraint_applied": (
                alert_count
                >= min_alert_bins
            ),
            "high_evidence_constraint_passed": (
                high_pass
            ),
            "alert_constraint_passed": (
                alert_pass
            ),
            "eligible": (
                dataset_eligible
            ),
        }

    return {
        "probability_threshold": (
            float(threshold)
        ),
        "datasets": dataset_results,
        "mean_tcn_invocation_fraction": (
            float(
                np.mean(
                    invocation_fractions
                )
            )
        ),
        "worst_dataset_high_evidence_coverage": (
            float(
                min(
                    high_coverages
                )
            )
        ),
        "worst_dataset_alert_coverage": (
            float(
                min(
                    alert_coverages_for_tie
                )
            )
        ),
        "eligible": eligible,
    }


def main() -> None:
    config = _load_yaml(
        CONFIG_PATH
    )

    if (
        config["study"]["branch"]
        != "research/evidence-aware-compute-routing"
    ):
        raise RuntimeError(
            "Unexpected branch in study "
            "configuration."
        )

    if (
        config["study"]["base_commit"]
        != "7561a8acdd1e9e03fd6212272bae95908bb3dd45"
    ):
        raise RuntimeError(
            "Evidence-aware study parent "
            "commit changed."
        )

    if (
        config["new_evaluation_data"][
            "status"
        ]
        != "not_yet_configured"
    ):
        raise RuntimeError(
            "Development must finish before "
            "new evaluation data is configured."
        )

    development = config[
        "development_data"
    ]

    primary_reference = _load_json(
        ROOT
        / development[
            "metropt3"
        ]["feature_reference"]
    )
    primary_features = [
        str(value)
        for value
        in primary_reference[
            "features"
        ]["names"]
    ]

    if len(primary_features) != 63:
        raise RuntimeError(
            "Expected frozen primary "
            "63-feature reference."
        )

    bundles = {
        name: _build_dataset(
            dataset_name=name,
            data_config=development[name],
            config=config,
            primary_features=(
                primary_features
            ),
        )
        for name in (
            "metropt3",
            "metropt2",
        )
    }

    adequacy = config[
        "selection"
    ]["data_adequacy"]

    minimum_high_bins = int(
        adequacy[
            "minimum_high_evidence_bins_each_validation_dataset"
        ]
    )

    for name, bundle in bundles.items():
        high_bins = int(
            bundle[
                "teacher_counts"
            ][
                "validation_high_evidence_bins"
            ]
        )

        if high_bins < minimum_high_bins:
            raise RuntimeError(
                f"{name} has only {high_bins} "
                "validation high-evidence bins; "
                f"minimum is {minimum_high_bins}."
            )

    pooled_fit = pd.concat(
        [
            bundle["fit_features"]
            for bundle in bundles.values()
        ],
        axis=0,
        ignore_index=True,
    )
    pooled_target = np.concatenate(
        [
            bundle["fit_target"]
            .to_numpy(dtype=int)
            for bundle in bundles.values()
        ]
    )

    router_scaler = StandardScaler(
        with_mean=True,
        with_std=True,
    )
    router_scaler.fit(
        pooled_fit.loc[
            :,
            ROUTER_FEATURE_NAMES,
        ].to_numpy(dtype=float)
    )

    x_fit = router_scaler.transform(
        pooled_fit.loc[
            :,
            ROUTER_FEATURE_NAMES,
        ].to_numpy(dtype=float)
    )

    x_validation = {
        name: router_scaler.transform(
            bundle[
                "validation_features"
            ].loc[
                :,
                ROUTER_FEATURE_NAMES,
            ].to_numpy(dtype=float)
        )
        for name, bundle
        in bundles.items()
    }

    router_model_cfg = config[
        "router_model"
    ]
    candidate_cfg = config[
        "candidate_grid"
    ]

    grid = logistic_candidate_grid(
        C_values=[
            float(value)
            for value in candidate_cfg["C"]
        ],
        class_weights=[
            (
                None
                if value is None
                else str(value)
            )
            for value
            in candidate_cfg[
                "class_weight"
            ]
        ],
        probability_thresholds=[
            float(value)
            for value
            in candidate_cfg[
                "probability_threshold"
            ]
        ],
    )

    if len(grid) != int(
        candidate_cfg[
            "total_candidates"
        ]
    ):
        raise RuntimeError(
            "Candidate-grid size differs from "
            "preregistration."
        )

    model_cache = {}
    candidate_reports = []

    unique_model_keys = sorted(
        {
            (
                candidate.C,
                candidate.class_weight,
            )
            for candidate in grid
        },
        key=lambda item: (
            item[0],
            ""
            if item[1] is None
            else item[1],
        ),
    )

    for C, class_weight in (
        unique_model_keys
    ):
        model = fit_logistic_model(
            x_fit,
            pooled_target,
            C=C,
            class_weight=class_weight,
            solver=str(
                router_model_cfg["solver"]
            ),
            max_iter=int(
                router_model_cfg[
                    "max_iter"
                ]
            ),
            random_state=int(
                router_model_cfg[
                    "random_state"
                ]
            ),
        )
        model_cache[
            (C, class_weight)
        ] = model

    probability_cache = {
        (
            name,
            C,
            class_weight,
        ): route_probabilities(
            model_cache[
                (C, class_weight)
            ],
            x_validation[name],
        )
        for name in bundles
        for C, class_weight
        in unique_model_keys
    }

    for candidate in grid:
        probabilities = {
            name: probability_cache[
                (
                    name,
                    candidate.C,
                    candidate.class_weight,
                )
            ]
            for name in bundles
        }

        report = _evaluate_candidate(
            probabilities=probabilities,
            threshold=(
                candidate.probability_threshold
            ),
            bundles=bundles,
            config=config,
        )

        report.update(
            {
                "C": candidate.C,
                "class_weight": (
                    candidate.class_weight
                ),
            }
        )
        candidate_reports.append(
            report
        )

    selection_error = None
    selected = None

    try:
        selected = select_candidate(
            candidate_reports
        )
    except RuntimeError as exc:
        selection_error = str(exc)

    selected_model_payload = None

    if selected is not None:
        selected_key = (
            float(selected["C"]),
            selected["class_weight"],
        )
        selected_model = model_cache[
            selected_key
        ]

        selected_model_payload = {
            "feature_names": (
                ROUTER_FEATURE_NAMES
            ),
            "feature_scaler": (
                serialize_standard_scaler(
                    router_scaler
                )
            ),
            "logistic": (
                serialize_logistic_model(
                    selected_model
                )
            ),
            "C": float(
                selected["C"]
            ),
            "class_weight": (
                selected[
                    "class_weight"
                ]
            ),
            "probability_threshold": (
                float(
                    selected[
                        "probability_threshold"
                    ]
                )
            ),
            "solver": str(
                router_model_cfg["solver"]
            ),
            "max_iter": int(
                router_model_cfg[
                    "max_iter"
                ]
            ),
            "random_state": int(
                router_model_cfg[
                    "random_state"
                ]
            ),
        }

    dataset_reports = {}

    for name, bundle in bundles.items():
        dataset_reports[name] = {
            "device": bundle["device"],
            "feature_count": len(
                bundle["features"]
            ),
            "checkpoint_sha256": (
                bundle[
                    "checkpoint_sha256"
                ]
            ),
            "train_rows": (
                bundle["train_rows"]
            ),
            "calibration_rows": (
                bundle[
                    "calibration_rows"
                ]
            ),
            "tcn_target_rows": (
                bundle[
                    "tcn_target_rows"
                ]
            ),
            "split": (
                bundle["split"]
            ),
            "teacher_thresholds": (
                bundle[
                    "teacher_thresholds"
                ]
            ),
            "teacher_counts": (
                bundle[
                    "teacher_counts"
                ]
            ),
            "feature_references": (
                bundle[
                    "feature_references"
                ]
            ),
            "fixed_pca_baselines": (
                bundle[
                    "fixed_pca_baselines"
                ]
            ),
        }

    payload = {
        "schema_version": (
            "aeroxai.evidence_aware_router_development.v1"
        ),
        "study": (
            config["study"]["name"]
        ),
        "preregistration_commit": (
            PREREGISTRATION_COMMIT
        ),
        "parent_failure_commit": (
            config["study"][
                "parent_failure_commit"
            ]
        ),
        "purpose": (
            "Fit and select a cheap "
            "evidence-aware logistic router "
            "using chronological CAL "
            "fit/validation only."
        ),
        "data_use": {
            "metropt3_test_opened": False,
            "metropt2_test_opened": False,
            "incident_labels_used": False,
            "new_evaluation_data_opened": False,
        },
        "development_split": (
            config[
                "development_split"
            ]
        ),
        "router_features": (
            ROUTER_FEATURE_NAMES
        ),
        "pooled_router_fit": {
            "rows": len(pooled_fit),
            "positive_high_evidence_rows": (
                int(
                    pooled_target.sum()
                )
            ),
            "positive_fraction": float(
                pooled_target.mean()
            ),
            "feature_scaler": (
                serialize_standard_scaler(
                    router_scaler
                )
            ),
        },
        "development_datasets": (
            dataset_reports
        ),
        "candidate_count": len(
            candidate_reports
        ),
        "candidate_grid": (
            candidate_reports
        ),
        "selection": {
            "selected": selected,
            "selection_error": (
                selection_error
            ),
            "selected_model": (
                selected_model_payload
            ),
        },
        "software": {
            "sklearn_version": (
                sklearn.__version__
            ),
            "torch_version": (
                torch.__version__
            ),
        },
        "guardrails": (
            config["guardrails"]
        ),
    }

    output_path = (
        ROOT
        / config["outputs"][
            "development_calibration"
        ]
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
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
                "metropt3_test_opened": False,
                "metropt2_test_opened": False,
                "new_evaluation_data_opened": False,
                "candidate_count": len(
                    candidate_reports
                ),
                "eligible_candidates": sum(
                    int(
                        bool(
                            report[
                                "eligible"
                            ]
                        )
                    )
                    for report
                    in candidate_reports
                ),
                "validation_teacher_counts": {
                    name: {
                        "high_evidence_bins": (
                            bundle[
                                "teacher_counts"
                            ][
                                "validation_high_evidence_bins"
                            ]
                        ),
                        "alert_bins": (
                            bundle[
                                "teacher_counts"
                            ][
                                "validation_alert_bins"
                            ]
                        ),
                    }
                    for name, bundle
                    in bundles.items()
                },
                "selected": selected,
                "selection_error": (
                    selection_error
                ),
                "output": str(
                    output_path.relative_to(
                        ROOT
                    )
                ),
            },
            indent=2,
        )
    )

    if selected is None:
        raise RuntimeError(
            selection_error
            or "No candidate selected."
        )


if __name__ == "__main__":
    main()
