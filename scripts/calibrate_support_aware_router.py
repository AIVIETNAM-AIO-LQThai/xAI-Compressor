from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ml.data.metropt2_features import (
    model_feature_columns as metropt2_feature_columns,
)
from ml.detection.alerts import causal_ewma
from ml.detection.common import (
    fit_scaler,
)
from ml.detection.pca_detector import (
    fit_pca_detector,
    score_pca_detector,
)
from ml.energy.hierarchical_inference import (
    higher_quantile,
    sha256_file,
)
from ml.energy.support_aware_routing import (
    SupportPolicy,
    calibration_support_percentiles,
    candidate_grid,
    evidence_coverage,
    feature_support_statistic,
    fixed_route,
    pca_latent_support_statistic,
    select_policy,
    support_aware_route,
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
    / "support_aware_energy_routing.yaml"
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
            "Support-aware study requires the "
            "frozen StandardScaler TCN."
        )

    checkpoint_features = [
        str(value)
        for value in checkpoint["features"]
    ]

    if checkpoint_features != features:
        raise RuntimeError(
            "TCN checkpoint feature identity "
            "does not match the development "
            "representation."
        )

    scaler = fit_scaler(
        train,
        features,
        method="standard",
    )

    location = np.asarray(
        checkpoint["scaler_location"],
        dtype=float,
    )
    scale = np.asarray(
        checkpoint["scaler_scale"],
        dtype=float,
    )

    if not np.allclose(
        scaler.mean_,
        location,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Rebuilt TCN StandardScaler mean "
            "does not match checkpoint."
        )

    if not np.allclose(
        scaler.scale_,
        scale,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise RuntimeError(
            "Rebuilt TCN StandardScaler scale "
            "does not match checkpoint."
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


def _dataset_bundle(
    *,
    name: str,
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

    # No forbidden TEST path is opened in this function.
    if name == "metropt3":
        features = list(
            primary_features
        )
        missing = [
            feature
            for feature in features
            if feature not in train.columns
        ]
        if missing:
            raise RuntimeError(
                "MetroPT-3 development feature "
                f"identity failed: {missing}"
            )
    elif name == "metropt2":
        features = (
            metropt2_feature_columns(
                train.columns,
                excluded_groups=[
                    str(
                        data_config[
                            "feature_variant"
                        ]
                    ).replace(
                        "flowmeter_excluded",
                        "flowmeter",
                    )
                ],
            )
        )
        if (
            data_config[
                "feature_variant"
            ]
            != "flowmeter_excluded"
        ):
            raise RuntimeError(
                "MetroPT2 development variant "
                "must remain flowmeter_excluded."
            )
        if features != primary_features:
            raise RuntimeError(
                "MetroPT2 development feature "
                "names/order do not match the "
                "primary 63-feature reference."
            )
        if len(features) != int(
            data_config[
                "expected_feature_count"
            ]
        ):
            raise RuntimeError(
                "MetroPT2 development feature "
                "count mismatch."
            )
    else:
        raise ValueError(
            f"Unknown development dataset: {name}"
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

    teacher = config[
        "support_aware_policy"
    ]["offline_teacher"]

    high_evidence_threshold = (
        higher_quantile(
            tcn_scores,
            float(
                teacher[
                    "high_evidence_quantile"
                ]
            ),
        )
    )

    anomaly_threshold = (
        higher_quantile(
            tcn_scores,
            float(
                teacher[
                    "anomaly_alert_quantile"
                ]
            ),
        )
    )

    checkpoint_threshold = float(
        checkpoint[
            "calibration_threshold"
        ]
    )

    if not np.isclose(
        anomaly_threshold,
        checkpoint_threshold,
        rtol=1.0e-6,
        atol=1.0e-6,
    ):
        raise RuntimeError(
            "Recomputed q0.995 TCN threshold "
            "does not match the frozen "
            "checkpoint calibration threshold."
        )

    pca_config = config[
        "frozen_models"
    ]["pca"]

    pca = fit_pca_detector(
        train,
        features,
        variance_retained=float(
            pca_config[
                "variance_retained"
            ]
        ),
        scaler_name=str(
            pca_config["scaler"]
        ),
    )

    pca_cal_raw = (
        score_pca_detector(
            calibration,
            pca,
        )
    )

    pca_cal_ewma = causal_ewma(
        pca_cal_raw,
        alpha=float(
            pca_config["ewma_alpha"]
        ),
        reset_gap_minutes=int(
            pca_config[
                "reset_gap_minutes"
            ]
        ),
    )

    router_cfg = config["router"]

    router_thresholds = {
        operating_id: higher_quantile(
            pca_cal_ewma,
            float(quantile),
        )
        for operating_id, quantile
        in router_cfg[
            "threshold_quantiles"
        ].items()
    }

    feature_scaler = fit_scaler(
        train,
        features,
        method="standard",
    )

    feature_support = (
        feature_support_statistic(
            calibration,
            features=features,
            scaler=feature_scaler,
            quantile=0.95,
        )
    )

    latent_support = (
        pca_latent_support_statistic(
            calibration,
            detector=pca,
        )
    )

    support = (
        calibration_support_percentiles(
            feature_support=feature_support,
            latent_support=latent_support,
        )
    )

    # TCN target rows are a strict subset because
    # 12 contiguous historical bins are required.
    target_index = pd.DatetimeIndex(
        tcn_scores.index
    )

    pca_aligned = pca_cal_ewma.reindex(
        target_index
    )
    support_aligned = support.reindex(
        target_index
    )

    if pca_aligned.isna().any():
        raise RuntimeError(
            "PCA calibration alignment to "
            "TCN targets contains NaN."
        )

    if support_aligned.isna().any().any():
        raise RuntimeError(
            "Support calibration alignment to "
            "TCN targets contains NaN."
        )

    high_evidence = (
        tcn_scores
        >= high_evidence_threshold
    )
    alert_evidence = (
        tcn_scores
        >= anomaly_threshold
    )

    fixed_baselines: dict[
        str,
        dict[str, float | int | None],
    ] = {}

    for (
        operating_id,
        threshold,
    ) in router_thresholds.items():
        route = fixed_route(
            pca_aligned,
            threshold=threshold,
        )

        fixed_baselines[
            operating_id
        ] = {
            "tcn_invocations": int(
                route.sum()
            ),
            "tcn_invocation_fraction": (
                float(route.mean())
            ),
            "high_evidence_bin_coverage": (
                evidence_coverage(
                    route,
                    high_evidence,
                )
            ),
            "tcn_alert_bin_coverage": (
                evidence_coverage(
                    route,
                    alert_evidence,
                )
            ),
        }

    return {
        "name": name,
        "train_rows": len(train),
        "calibration_rows": (
            len(calibration)
        ),
        "tcn_score_rows": len(
            tcn_scores
        ),
        "features": features,
        "checkpoint_sha256": str(
            data_config[
                "tcn_checkpoint_sha256"
            ]
        ),
        "device": device,
        "tcn_scores": tcn_scores,
        "teacher_thresholds": {
            "high_evidence_q90": (
                high_evidence_threshold
            ),
            "anomaly_alert_q995": (
                anomaly_threshold
            ),
        },
        "teacher_counts": {
            "high_evidence_bins": int(
                high_evidence.sum()
            ),
            "tcn_alert_bins": int(
                alert_evidence.sum()
            ),
        },
        "high_evidence": (
            high_evidence
        ),
        "alert_evidence": (
            alert_evidence
        ),
        "pca_scores": pca_aligned,
        "router_thresholds": (
            router_thresholds
        ),
        "support": support_aligned,
        "support_raw_summary": {
            "feature": {
                "minimum": float(
                    feature_support.min()
                ),
                "median": float(
                    feature_support.median()
                ),
                "maximum": float(
                    feature_support.max()
                ),
            },
            "latent": {
                "minimum": float(
                    latent_support.min()
                ),
                "median": float(
                    latent_support.median()
                ),
                "maximum": float(
                    latent_support.max()
                ),
            },
        },
        "fixed_baselines": (
            fixed_baselines
        ),
    }


def _evaluate_candidate(
    *,
    policy: SupportPolicy,
    bundles: dict[
        str,
        dict[str, Any],
    ],
    config: dict[str, Any],
) -> dict[str, Any]:
    constraints = config[
        "support_aware_policy"
    ]["eligibility_constraints"]

    min_high = float(
        constraints[
            "minimum_high_evidence_bin_coverage_each_development_dataset"
        ]
    )
    min_alert = float(
        constraints[
            "minimum_tcn_alert_bin_coverage_each_development_dataset"
        ]
    )
    min_alert_bins = int(
        constraints[
            "minimum_alert_bins_for_alert_constraint"
        ]
    )

    dataset_results: dict[
        str,
        dict[str, Any],
    ] = {}

    invocation_fractions = []
    alert_coverages_for_tie = []
    eligible = True

    for name, bundle in (
        bundles.items()
    ):
        route = support_aware_route(
            bundle["pca_scores"],
            bundle["support"][
                "combined_support_percentile"
            ],
            policy=policy,
            router_thresholds=(
                bundle[
                    "router_thresholds"
                ]
            ),
        )

        high_coverage = (
            evidence_coverage(
                route,
                bundle[
                    "high_evidence"
                ],
            )
        )
        alert_coverage = (
            evidence_coverage(
                route,
                bundle[
                    "alert_evidence"
                ],
            )
        )

        if high_coverage is None:
            raise RuntimeError(
                f"{name} has no q0.90 "
                "high-evidence bins."
            )

        alert_count = int(
            bundle[
                "teacher_counts"
            ]["tcn_alert_bins"]
        )

        high_pass = (
            high_coverage >= min_high
        )

        if alert_count >= min_alert_bins:
            if alert_coverage is None:
                raise RuntimeError(
                    f"{name} alert coverage "
                    "is unexpectedly undefined."
                )
            alert_pass = (
                alert_coverage
                >= min_alert
            )
            tie_alert = (
                alert_coverage
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
        alert_coverages_for_tie.append(
            float(tie_alert)
        )

        dataset_results[name] = {
            "tcn_invocations": int(
                route.sum()
            ),
            "tcn_invocation_fraction": (
                invocation_fraction
            ),
            "high_evidence_bins": int(
                bundle[
                    "teacher_counts"
                ][
                    "high_evidence_bins"
                ]
            ),
            "high_evidence_bin_coverage": (
                high_coverage
            ),
            "tcn_alert_bins": (
                alert_count
            ),
            "tcn_alert_bin_coverage": (
                alert_coverage
            ),
            "high_evidence_constraint_passed": (
                high_pass
            ),
            "alert_constraint_applied": (
                alert_count
                >= min_alert_bins
            ),
            "alert_constraint_passed": (
                alert_pass
            ),
            "eligible": dataset_eligible,
        }

    return {
        "mid_support_percentile": (
            policy.mid_support_percentile
        ),
        "high_support_percentile": (
            policy.high_support_percentile
        ),
        "datasets": dataset_results,
        "mean_tcn_invocation_fraction": (
            float(
                np.mean(
                    invocation_fractions
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
        != "research/support-aware-energy-routing"
    ):
        raise RuntimeError(
            "Unexpected study branch in "
            "preregistration."
        )

    if (
        config["new_evaluation_data"][
            "status"
        ]
        != "not_yet_configured"
    ):
        raise RuntimeError(
            "Calibration stage expects new "
            "evaluation data to remain "
            "unconfigured."
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
            "Expected frozen 63-feature "
            "primary reference."
        )

    bundles = {
        name: _dataset_bundle(
            name=name,
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

    grid = config[
        "support_aware_policy"
    ]["calibration_candidate_grid"]

    policies = candidate_grid(
        mid_values=[
            float(value)
            for value
            in grid[
                "mid_support_percentile"
            ]
        ],
        high_values=[
            float(value)
            for value
            in grid[
                "high_support_percentile"
            ]
        ],
    )

    candidate_reports = [
        _evaluate_candidate(
            policy=policy,
            bundles=bundles,
            config=config,
        )
        for policy in policies
    ]

    selection_error = None
    selected = None

    try:
        selected = select_policy(
            candidate_reports
        )
    except RuntimeError as exc:
        selection_error = str(exc)

    dataset_reports = {}

    for name, bundle in (
        bundles.items()
    ):
        dataset_reports[name] = {
            "train_rows": (
                bundle["train_rows"]
            ),
            "calibration_rows": (
                bundle[
                    "calibration_rows"
                ]
            ),
            "tcn_score_rows": (
                bundle[
                    "tcn_score_rows"
                ]
            ),
            "feature_count": len(
                bundle["features"]
            ),
            "checkpoint_sha256": (
                bundle[
                    "checkpoint_sha256"
                ]
            ),
            "device": (
                bundle["device"]
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
            "router_thresholds": (
                bundle[
                    "router_thresholds"
                ]
            ),
            "support_raw_summary": (
                bundle[
                    "support_raw_summary"
                ]
            ),
            "fixed_baselines": (
                bundle[
                    "fixed_baselines"
                ]
            ),
        }

    payload = {
        "schema_version": (
            "aeroxai.support_aware_router_calibration.v1"
        ),
        "study": (
            config["study"]["name"]
        ),
        "preregistration_commit": (
            "04f906258a910e68d04cb44c2a282a4ebb62e597"
        ),
        "purpose": (
            "Select one support-aware routing "
            "policy using TRAIN/CALIBRATION only."
        ),
        "data_use": {
            "metropt3_test_opened": False,
            "metropt2_test_opened": False,
            "incident_labels_used_for_selection": (
                False
            ),
            "new_evaluation_data_opened": (
                False
            ),
        },
        "support_definition": {
            "feature_statistic": (
                "row q0.95 of absolute "
                "TRAIN-Standard-scaled features"
            ),
            "latent_statistic": (
                "sqrt(sum(z_k^2 / "
                "TRAIN_PCA_explained_variance_k))"
            ),
            "empirical_percentile": (
                "right-inclusive empirical CDF: "
                "count(CAL <= value) / N_CAL"
            ),
            "combined": (
                "max(feature percentile, "
                "latent percentile)"
            ),
        },
        "development_datasets": (
            dataset_reports
        ),
        "candidate_grid": (
            candidate_reports
        ),
        "selection": {
            "selected": selected,
            "selection_error": (
                selection_error
            ),
        },
        "guardrails": (
            config["guardrails"]
        ),
    }

    output_path = (
        ROOT
        / config["outputs"][
            "calibration_policy"
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

    compact = {
        "metropt3_test_opened": False,
        "metropt2_test_opened": False,
        "new_evaluation_data_opened": False,
        "candidates": len(
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
            for report in candidate_reports
        ),
        "selected": selected,
        "selection_error": (
            selection_error
        ),
        "output": str(
            output_path.relative_to(ROOT)
        ),
    }

    print(
        json.dumps(
            compact,
            indent=2,
        )
    )

    if selected is None:
        raise RuntimeError(
            selection_error
            or "No policy selected."
        )


if __name__ == "__main__":
    main()
