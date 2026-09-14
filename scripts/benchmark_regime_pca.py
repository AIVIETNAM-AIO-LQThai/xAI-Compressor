from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.data.features import (
    model_feature_columns,
)
from ml.detection.pca_detector import (
    fit_pca_detector,
    score_pca_detector,
)
from ml.detection.pca_research_benchmark import (
    AlertProtocol,
    benchmark_pca_variant,
)
from ml.detection.regime_pca_benchmark import (
    benchmark_regime_pca_variant,
    calibration_threshold_stability,
    strict_dominance,
)

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT
    / "configs"
    / "regime_pca_benchmark.yaml"
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
            f"Expected mapping in {path}."
        )

    return payload


def _protocol(
    config: dict[str, Any],
) -> AlertProtocol:
    alerting = config[
        "alerting"
    ]

    evaluation = config[
        "evaluation"
    ]

    return AlertProtocol(
        ewma_alpha=float(
            alerting[
                "ewma_alpha"
            ]
        ),
        threshold_quantile=float(
            alerting[
                "threshold_quantile"
            ]
        ),
        persistence_hits=int(
            alerting[
                "persistence_hits"
            ]
        ),
        persistence_window=int(
            alerting[
                "persistence_window"
            ]
        ),
        reset_gap_minutes=int(
            alerting[
                "reset_gap_minutes"
            ]
        ),
        merge_minutes=int(
            alerting[
                "merge_minutes"
            ]
        ),
        early_warning_hours=int(
            evaluation[
                "early_warning_hours"
            ]
        ),
        late_tolerance_hours=int(
            evaluation[
                "late_tolerance_hours"
            ]
        ),
        bin_minutes=int(
            evaluation[
                "bin_minutes"
            ]
        ),
    )


def _assert_frozen_reproduction(
    result: dict[str, Any],
    config: dict[str, Any],
) -> None:
    reproduction = config[
        "frozen_robust_reproduction"
    ]

    expected = reproduction[
        "expected"
    ]

    tolerance = float(
        reproduction[
            "tolerance"
        ]
    )

    metrics = result[
        "metrics"
    ]

    failures = []

    for key, expected_value in (
        expected.items()
    ):
        observed = float(
            metrics[key]
        )

        difference = abs(
            observed
            - float(
                expected_value
            )
        )

        if difference > tolerance:
            failures.append(
                {
                    "metric": key,
                    "observed": observed,
                    "expected": float(
                        expected_value
                    ),
                    "difference": (
                        difference
                    ),
                }
            )

    if failures:
        raise RuntimeError(
            "Frozen Robust-PCA reproduction "
            "guard failed:\n"
            + json.dumps(
                failures,
                indent=2,
            )
        )


def _global_calibration_scores(
    *,
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    features: list[str],
    variance_retained: float,
    scaler_name: str,
) -> pd.Series:
    detector = fit_pca_detector(
        train,
        features,
        variance_retained=(
            variance_retained
        ),
        scaler_name=scaler_name,
    )

    return score_pca_detector(
        calibration,
        detector,
    )


def main() -> None:
    config = _load_yaml(
        CONFIG_PATH
    )

    dataset = config[
        "dataset"
    ]

    train = pd.read_parquet(
        ROOT
        / dataset[
            "train_file"
        ]
    ).sort_index()

    calibration = pd.read_parquet(
        ROOT
        / dataset[
            "calibration_file"
        ]
    ).sort_index()

    test = pd.read_parquet(
        ROOT
        / dataset[
            "test_file"
        ]
    ).sort_index()

    metropt = _load_yaml(
        ROOT
        / dataset[
            "incidents_config"
        ]
    )

    incidents = list(
        metropt[
            "dataset"
        ][
            "reported_incidents"
        ]
    )

    features = model_feature_columns(
        train.columns
    )

    variance_retained = float(
        config[
            "representation"
        ][
            "variance_retained"
        ]
    )

    protocol = _protocol(
        config
    )

    print(
        "Study role:",
        config[
            "study"
        ][
            "role"
        ],
    )

    print(
        "Rows:",
        {
            "train": len(train),
            "calibration": len(
                calibration
            ),
            "test": len(test),
        },
    )

    print(
        "Features:",
        len(features),
    )

    print(
        "Benchmarking frozen Robust-PCA..."
    )

    robust = benchmark_pca_variant(
        train=train,
        calibration=calibration,
        test=test,
        features=features,
        incidents=incidents,
        protocol=protocol,
        variance_retained=(
            variance_retained
        ),
        scaler_name="robust",
    )

    _assert_frozen_reproduction(
        robust,
        config,
    )

    print(
        "Frozen Robust-PCA reproduction: PASS"
    )

    print(
        "Benchmarking global Standard-PCA..."
    )

    standard = benchmark_pca_variant(
        train=train,
        calibration=calibration,
        test=test,
        features=features,
        incidents=incidents,
        protocol=protocol,
        variance_retained=(
            variance_retained
        ),
        scaler_name="standard",
    )

    router = config[
        "regime_router"
    ]

    print(
        "Benchmarking regime-conditioned "
        "Standard-PCA..."
    )

    regime, regime_calibration_scores = (
        benchmark_regime_pca_variant(
            train=train,
            calibration=calibration,
            test=test,
            features=features,
            incidents=incidents,
            protocol=protocol,
            variance_retained=(
                variance_retained
            ),
            current_feature=str(
                router[
                    "current_feature"
                ]
            ),
            pressure_feature=str(
                router[
                    "pressure_feature"
                ]
            ),
            high_quantile=float(
                router[
                    "high_quantile"
                ]
            ),
        )
    )

    stability_config = config[
        "calibration_stability"
    ]

    stability_kwargs = {
        "protocol": protocol,
        "bootstrap_replicates": int(
            stability_config[
                "bootstrap_replicates"
            ]
        ),
        "block_lengths": [
            int(value)
            for value in stability_config[
                "circular_block_lengths"
            ]
        ],
        "seed": int(
            stability_config[
                "seed"
            ]
        ),
        "confidence": float(
            stability_config[
                "confidence"
            ]
        ),
    }

    print(
        "Quantifying calibration stability..."
    )

    robust_calibration_scores = (
        _global_calibration_scores(
            train=train,
            calibration=calibration,
            features=features,
            variance_retained=(
                variance_retained
            ),
            scaler_name="robust",
        )
    )

    standard_calibration_scores = (
        _global_calibration_scores(
            train=train,
            calibration=calibration,
            features=features,
            variance_retained=(
                variance_retained
            ),
            scaler_name="standard",
        )
    )

    robust_stability = (
        calibration_threshold_stability(
            robust_calibration_scores,
            **stability_kwargs,
        )
    )

    standard_stability = (
        calibration_threshold_stability(
            standard_calibration_scores,
            **stability_kwargs,
        )
    )

    regime_stability = (
        calibration_threshold_stability(
            regime_calibration_scores,
            **stability_kwargs,
        )
    )

    dominance = strict_dominance(
        candidate=regime,
        references=[
            robust,
            standard,
        ],
        candidate_stability=(
            regime_stability
        ),
        reference_stabilities=[
            robust_stability,
            standard_stability,
        ],
    )

    result = {
        "schema_version": (
            "aeroxai.regime_pca_benchmark.v1"
        ),
        "study_role": config[
            "study"
        ][
            "role"
        ],
        "evidence_class": config[
            "study"
        ][
            "evidence_class"
        ],
        "data": {
            "train_rows": len(
                train
            ),
            "calibration_rows": len(
                calibration
            ),
            "test_rows": len(
                test
            ),
            "feature_count": len(
                features
            ),
            "incident_count": len(
                incidents
            ),
        },
        "protocol": {
            "variance_retained": (
                variance_retained
            ),
            "alert": {
                "ewma_alpha": (
                    protocol.ewma_alpha
                ),
                "threshold_quantile": (
                    protocol.threshold_quantile
                ),
                "persistence_hits": (
                    protocol.persistence_hits
                ),
                "persistence_window": (
                    protocol.persistence_window
                ),
                "reset_gap_minutes": (
                    protocol.reset_gap_minutes
                ),
                "merge_minutes": (
                    protocol.merge_minutes
                ),
            },
            "evaluation": {
                "early_warning_hours": (
                    protocol.early_warning_hours
                ),
                "late_tolerance_hours": (
                    protocol.late_tolerance_hours
                ),
                "bin_minutes": (
                    protocol.bin_minutes
                ),
            },
        },
        "frozen_robust_reproduction": {
            "status": "PASS",
            "result": robust,
            "calibration_stability": (
                robust_stability
            ),
        },
        "global_standard_pca": {
            "result": standard,
            "calibration_stability": (
                standard_stability
            ),
        },
        "regime_conditioned_standard_pca": {
            "result": regime,
            "calibration_stability": (
                regime_stability
            ),
        },
        "decision": dominance,
        "guardrails": config[
            "guardrails"
        ],
    }

    output_path = (
        ROOT
        / config[
            "outputs"
        ][
            "result_json"
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    compact = {
        "frozen_robust": robust[
            "metrics"
        ],
        "global_standard": standard[
            "metrics"
        ],
        "regime_conditioned": regime[
            "metrics"
        ],
        "strict_dominance": dominance,
        "output": str(
            output_path
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
