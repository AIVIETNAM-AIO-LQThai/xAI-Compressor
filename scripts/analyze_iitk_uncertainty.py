from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.data.iitk_air_compressor import (
    model_feature_columns,
)
from ml.detection.iitk_external_benchmark import (
    calibration_threshold,
    detection_metrics,
)
from ml.detection.iitk_uncertainty import (
    bootstrap_calibration_thresholds,
    fit_representation_scores,
    leave_one_out_thresholds,
    quantile_order_information,
    stratified_auc_bootstrap,
    summarize_threshold_operating_distributions,
    top_calibration_scores,
    wilson_interval,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "iitk_air_compressor.yaml"
)


def _load_yaml(
    path: Path,
) -> dict[str, Any]:
    payload = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            f"{path} must contain a mapping."
        )

    return payload


def _load_json(
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            f"{path} must contain an object."
        )

    return payload


def _wilson_reference_intervals(
    *,
    metrics: dict[str, Any],
    confidence_level: float,
) -> dict[str, Any]:
    healthy_trials = int(
        metrics[
            "healthy_recordings"
        ]
    )

    healthy_false_positives = round(
        metrics["heldout_healthy_false_positive_rate"] * healthy_trials
    )

    fault_trials = int(metrics["fault_recordings"])

    fault_true_positives = round(
        metrics["overall_fault_recall"] * fault_trials
    )

    per_fault = {}

    for condition, recall in (
        metrics["per_fault_recall"].items()
    ):
        trials = 225

        successes = round(
            float(recall) * trials
        )

        per_fault[condition] = wilson_interval(
            successes,
            trials,
            confidence_level=(
                confidence_level
            ),
        )

    return {
        "heldout_healthy_false_positive_rate": (
            wilson_interval(
                healthy_false_positives,
                healthy_trials,
                confidence_level=(
                    confidence_level
                ),
            )
        ),
        "overall_fault_recall": (
            wilson_interval(
                fault_true_positives,
                fault_trials,
                confidence_level=(
                    confidence_level
                ),
            )
        ),
        "per_fault_recall": (
            per_fault
        ),
    }


def main() -> None:
    config = _load_yaml(
        CONFIG_PATH
    )

    preprocessing = config[
        "preprocessing"
    ]

    validation = config[
        "external_validation"
    ]

    uncertainty = config[
        "uncertainty_analysis"
    ]

    benchmark = _load_json(
        ROOT
        / validation[
            "output_file"
        ]
    )

    preprocessing_report = (
        _load_json(
            ROOT
            / preprocessing[
                "report_file"
            ]
        )
    )

    frame = pd.read_parquet(
        ROOT
        / preprocessing[
            "output_file"
        ]
    )

    features = model_feature_columns(
        frame
    )

    if len(
        features
    ) != int(
        preprocessing_report[
            "feature_count"
        ]
    ):
        raise RuntimeError(
            "IITK feature schema drifted."
        )

    train_mask = (
        frame[
            "split"
        ]
        == "train"
    )

    calibration_mask = (
        frame[
            "split"
        ]
        == "calibration"
    )

    test_mask = (
        frame[
            "split"
        ]
        == "test"
    )

    train = frame.loc[
        train_mask,
        features,
    ].copy()

    calibration = frame.loc[
        calibration_mask,
        features,
    ].copy()

    test = frame.loc[
        test_mask,
        features,
    ].copy()

    calibration_metadata = (
        frame.loc[
            calibration_mask,
            [
                "condition",
                "reading",
                "is_fault",
            ],
        ]
        .copy()
    )

    test_metadata = (
        frame.loc[
            test_mask,
            [
                "condition",
                "reading",
                "is_fault",
            ],
        ]
        .copy()
    )

    confidence_level = float(
        uncertainty[
            "confidence_level"
        ]
    )

    threshold_quantile = float(
        validation[
            "threshold_quantile"
        ]
    )

    bootstrap_config = (
        uncertainty[
            "calibration_bootstrap"
        ]
    )

    bootstrap_iterations = int(
        bootstrap_config[
            "iterations"
        ]
    )

    bootstrap_seed = int(
        bootstrap_config[
            "seed"
        ]
    )

    auc_config = uncertainty[
        "auc_bootstrap"
    ]

    results = {}

    representation_names = list(
        validation[
            "candidate_representations"
        ]
    )

    for representation_index, name in enumerate(
        representation_names
    ):
        print(
            f"Analyzing uncertainty for {name}..."
        )

        specification = (
            validation[
                "candidate_representations"
            ][name]
        )

        (
            calibration_scores,
            test_scores,
        ) = fit_representation_scores(
            train=train,
            calibration=calibration,
            test=test,
            features=features,
            specification=specification,
        )

        observed_threshold = (
            calibration_threshold(
                calibration_scores,
                quantile=(
                    threshold_quantile
                ),
            )
        )

        observed_metrics = (
            detection_metrics(
                scores=test_scores,
                metadata=test_metadata,
                threshold=(
                    observed_threshold
                ),
            )
        )

        frozen = benchmark[
            "representations"
        ][name]

        frozen_threshold = float(
            frozen[
                "model"
            ][
                "threshold"
            ]
        )

        threshold_difference = abs(
            observed_threshold
            - frozen_threshold
        )

        if threshold_difference > 1.0e-12:
            raise RuntimeError(
                "Frozen threshold no longer "
                f"reproduces for {name}: "
                f"{observed_threshold} vs "
                f"{frozen_threshold}"
            )

        for metric_name in (
            "roc_auc",
            "heldout_healthy_false_positive_rate",
            "overall_fault_recall",
            "balanced_accuracy",
            "macro_fault_recall",
        ):
            observed = float(
                observed_metrics[
                    metric_name
                ]
            )

            expected = float(
                frozen[
                    "metrics"
                ][
                    metric_name
                ]
            )

            if abs(
                observed
                - expected
            ) > 1.0e-12:
                raise RuntimeError(
                    "Frozen metric no longer "
                    f"reproduces for {name} / "
                    f"{metric_name}: "
                    f"{observed} vs {expected}"
                )

        wilson = (
            _wilson_reference_intervals(
                metrics=observed_metrics,
                confidence_level=(
                    confidence_level
                ),
            )
        )

        loo_thresholds = (
            leave_one_out_thresholds(
                calibration_scores,
                quantile=(
                    threshold_quantile
                ),
            )
        )

        loo_summary = (
            summarize_threshold_operating_distributions(
                thresholds=(
                    loo_thresholds
                ),
                test_scores=test_scores,
                metadata=test_metadata,
                confidence_level=(
                    confidence_level
                ),
            )
        )

        bootstrap_modes = {}

        mode_specs = [
            (
                "iid",
                None,
            ),
            *[
                (
                    f"moving_block_{block_length}",
                    int(
                        block_length
                    ),
                )
                for block_length
                in bootstrap_config[
                    "moving_block_lengths"
                ]
            ],
        ]

        for mode_index, (
            mode_name,
            block_length,
        ) in enumerate(
            mode_specs
        ):
            seed = (
                bootstrap_seed
                + 1000
                * representation_index
                + mode_index
            )

            thresholds = (
                bootstrap_calibration_thresholds(
                    calibration_scores,
                    quantile=(
                        threshold_quantile
                    ),
                    iterations=(
                        bootstrap_iterations
                    ),
                    seed=seed,
                    block_length=(
                        block_length
                    ),
                )
            )

            bootstrap_modes[
                mode_name
            ] = {
                "seed": int(
                    seed
                ),
                "block_length": (
                    block_length
                ),
                "iterations": (
                    bootstrap_iterations
                ),
                **(
                    summarize_threshold_operating_distributions(
                        thresholds=(
                            thresholds
                        ),
                        test_scores=(
                            test_scores
                        ),
                        metadata=(
                            test_metadata
                        ),
                        confidence_level=(
                            confidence_level
                        ),
                    )
                ),
            }

        auc_seed = (
            int(
                auc_config[
                    "seed"
                ]
            )
            + representation_index
        )

        auc_uncertainty = (
            stratified_auc_bootstrap(
                scores=test_scores,
                metadata=test_metadata,
                iterations=int(
                    auc_config[
                        "iterations"
                    ]
                ),
                seed=auc_seed,
                confidence_level=(
                    confidence_level
                ),
            )
        )

        auc_uncertainty[
            "seed"
        ] = int(
            auc_seed
        )

        results[
            name
        ] = {
            "frozen_reproduction": {
                "threshold": (
                    observed_threshold
                ),
                "threshold_absolute_difference": (
                    threshold_difference
                ),
                "roc_auc": float(
                    observed_metrics[
                        "roc_auc"
                    ]
                ),
                "healthy_fpr": float(
                    observed_metrics[
                        "heldout_healthy_false_positive_rate"
                    ]
                ),
                "overall_fault_recall": float(
                    observed_metrics[
                        "overall_fault_recall"
                    ]
                ),
            },
            "threshold_order": (
                quantile_order_information(
                    len(
                        calibration_scores
                    ),
                    quantile=(
                        threshold_quantile
                    ),
                )
            ),
            "top_calibration_scores": (
                top_calibration_scores(
                    calibration_scores,
                    calibration_metadata,
                    count=int(
                        uncertainty[
                            "calibration_tail"
                        ][
                            "top_scores_to_report"
                        ]
                    ),
                )
            ),
            "wilson_reference_intervals": (
                wilson
            ),
            "leave_one_out_calibration": (
                loo_summary
            ),
            "calibration_bootstrap": (
                bootstrap_modes
            ),
            "roc_auc_bootstrap": (
                auc_uncertainty
            ),
        }

    report = {
        "schema_version": (
            "aeroxai.iitk_uncertainty.v1"
        ),
        "dataset": config[
            "dataset"
        ][
            "short_name"
        ],
        "evidence_class": "REAL",
        "causal_claim": False,
        "analysis_status": uncertainty[
            "status"
        ],
        "confidence_level": (
            confidence_level
        ),
        "frozen_operating_rule": {
            "threshold_quantile": (
                threshold_quantile
            ),
            "threshold_interpolation": (
                validation[
                    "threshold_interpolation"
                ]
            ),
            "calibration_healthy": len(
                    calibration
                ),
            "test_retuning": False,
        },
        "representations": results,
        "interpretation_guardrails": (
            uncertainty[
                "interpretation_guardrails"
            ]
        ),
    }

    output_path = (
        ROOT
        / uncertainty[
            "output_file"
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    compact = {
        "frozen_operating_rule": (
            report[
                "frozen_operating_rule"
            ]
        ),
        "representations": {},
        "output": str(
            output_path
        ),
    }

    for name, result in (
        results.items()
    ):
        compact[
            "representations"
        ][name] = {
            "frozen_reproduction": (
                result[
                    "frozen_reproduction"
                ]
            ),
            "threshold_order": (
                result[
                    "threshold_order"
                ]
            ),
            "top_calibration_scores": (
                result[
                    "top_calibration_scores"
                ]
            ),
            "wilson_reference_intervals": (
                result[
                    "wilson_reference_intervals"
                ]
            ),
            "leave_one_out_calibration": (
                result[
                    "leave_one_out_calibration"
                ]
            ),
            "calibration_bootstrap": (
                result[
                    "calibration_bootstrap"
                ]
            ),
            "roc_auc_bootstrap": (
                result[
                    "roc_auc_bootstrap"
                ]
            ),
        }

    print(
        json.dumps(
            compact,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
