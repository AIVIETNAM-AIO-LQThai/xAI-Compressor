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
    run_representation,
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

    audit_report = _load_json(
        ROOT
        / config[
            "audit"
        ][
            "report_file"
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

    if not audit_report[
        "audit_passed"
    ]:
        raise RuntimeError(
            "Frozen IITK data audit did not pass."
        )

    leakage = (
        preprocessing_report[
            "leakage_checks"
        ]
    )

    if (
        leakage[
            "train_contains_fault"
        ]
        or leakage[
            "calibration_contains_fault"
        ]
        or not leakage[
            "faults_test_only"
        ]
    ):
        raise RuntimeError(
            "Frozen IITK preprocessing leakage "
            "checks failed."
        )

    frame = pd.read_parquet(
        ROOT
        / preprocessing[
            "output_file"
        ]
    )

    features = (
        model_feature_columns(
            frame
        )
    )

    if len(
        features
    ) != int(
        preprocessing_report[
            "feature_count"
        ]
    ):
        raise RuntimeError(
            "Feature schema drifted from frozen "
            "preprocessing report."
        )

    train = (
        frame.loc[
            frame[
                "split"
            ]
            == "train",
            features,
        ]
        .copy()
    )

    calibration = (
        frame.loc[
            frame[
                "split"
            ]
            == "calibration",
            features,
        ]
        .copy()
    )

    test_mask = (
        frame[
            "split"
        ]
        == "test"
    )

    test = (
        frame.loc[
            test_mask,
            features,
        ]
        .copy()
    )

    metadata = (
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

    expected = (
        preprocessing_report[
            "split_counts"
        ]
    )

    if (
        len(
            train
        )
        != int(
            expected[
                "train"
            ]
        )
        or len(
            calibration
        )
        != int(
            expected[
                "calibration"
            ]
        )
        or len(
            test
        )
        != int(
            expected[
                "test"
            ]
        )
    ):
        raise RuntimeError(
            "IITK split counts drifted from "
            "frozen preprocessing report."
        )

    results = {}

    for name, specification in (
        validation[
            "candidate_representations"
        ].items()
    ):
        print(
            f"Running {name}..."
        )

        results[
            name
        ] = run_representation(
            train=train,
            calibration=(
                calibration
            ),
            test=test,
            features=features,
            metadata=metadata,
            specification=(
                specification
            ),
            threshold_quantile=float(
                validation[
                    "threshold_quantile"
                ]
            ),
        )

    report = {
        "schema_version": (
            "aeroxai.iitk_external_benchmark.v1"
        ),
        "dataset": config[
            "dataset"
        ][
            "short_name"
        ],
        "evidence_class": "REAL",
        "causal_claim": False,
        "benchmark_role": validation[
            "role"
        ],
        "protocol": {
            "train_healthy": len(
                    train
                ),
            "calibration_healthy": len(
                    calibration
                ),
            "test_recordings": len(
                    test
                ),
            "test_healthy": int(
                (
                    ~metadata[
                        "is_fault"
                    ].astype(
                        bool
                    )
                ).sum()
            ),
            "test_fault": int(
                metadata[
                    "is_fault"
                ].astype(
                    bool
                ).sum()
            ),
            "feature_count": len(
                    features
                ),
            "feature_group_count": 5,
            "threshold_quantile": float(
                validation[
                    "threshold_quantile"
                ]
            ),
            "threshold_interpolation": (
                validation[
                    "threshold_interpolation"
                ]
            ),
            "temporal_alert_logic": (
                False
            ),
            "test_retuning": False,
        },
        "representations": (
            results
        ),
        "interpretation_guardrails": (
            validation[
                "interpretation_guardrails"
            ]
        ),
    }

    output_path = (
        ROOT
        / validation[
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
        "protocol": report[
            "protocol"
        ],
        "representations": {},
        "output": str(
            output_path
        ),
    }

    for name, result in (
        results.items()
    ):
        metrics = result[
            "metrics"
        ]

        compact[
            "representations"
        ][name] = {
            "model": result[
                "model"
            ],
            "roc_auc": (
                metrics[
                    "roc_auc"
                ]
            ),
            "average_precision": (
                metrics[
                    "average_precision"
                ]
            ),
            "average_precision_baseline": (
                metrics[
                    "average_precision_prevalence_baseline"
                ]
            ),
            "healthy_fpr": (
                metrics[
                    "heldout_healthy_false_positive_rate"
                ]
            ),
            "overall_fault_recall": (
                metrics[
                    "overall_fault_recall"
                ]
            ),
            "balanced_accuracy": (
                metrics[
                    "balanced_accuracy"
                ]
            ),
            "macro_fault_recall": (
                metrics[
                    "macro_fault_recall"
                ]
            ),
            "per_fault_recall": (
                metrics[
                    "per_fault_recall"
                ]
            ),
            "per_fault_roc_auc": (
                metrics[
                    "per_fault_roc_auc"
                ]
            ),
            "explanations_all_test": (
                result[
                    "explanations_all_test"
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
