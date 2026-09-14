from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.data.metropt2_features import (
    aggregate_metropt2_split,
    load_metropt2_compressor_frame,
    model_feature_columns,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "metropt2.yaml"


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return yaml.safe_load(
            handle
        )


def _split_summary(
    result,
) -> dict[str, Any]:
    features = result.features

    return {
        "raw_rows": result.raw_rows,
        "raw_start": (
            None
            if result.raw_start is None
            else str(
                result.raw_start
            )
        ),
        "raw_end": (
            None
            if result.raw_end is None
            else str(
                result.raw_end
            )
        ),
        "candidate_bins": (
            result.candidate_bins
        ),
        "valid_bins": (
            result.valid_bins
        ),
        "coverage_rejected_bins": (
            result.coverage_rejected_bins
        ),
        "incomplete_rejected_bins": (
            result.incomplete_rejected_bins
        ),
        "gap_tainted_bins": (
            result.gap_tainted_bins
        ),
        "empty_bins": (
            result.empty_bins
        ),
        "first_valid_bin": (
            None
            if features.empty
            else str(
                features.index.min()
            )
        ),
        "last_valid_bin": (
            None
            if features.empty
            else str(
                features.index.max()
            )
        ),
        "valid_exposure_days": (
            result.valid_bins
            * 5.0
            / (60.0 * 24.0)
        ),
    }


def _incident_coverage(
    test_features: pd.DataFrame,
    incidents: list[dict[str, Any]],
    *,
    bin_minutes: int,
) -> list[dict[str, Any]]:
    expected_pre_bins = int(
        24 * 60 / bin_minutes
    )

    rows: list[
        dict[str, Any]
    ] = []

    for incident in incidents:
        start = pd.Timestamp(
            incident["start"]
        )
        end = pd.Timestamp(
            incident["end"]
        )

        pre_start = (
            start
            - pd.Timedelta(
                hours=24
            )
        )

        pre_mask = (
            (
                test_features.index
                >= pre_start
            )
            & (
                test_features.index
                < start
            )
        )

        incident_mask = (
            (
                test_features.index
                >= start
            )
            & (
                test_features.index
                <= end
            )
        )

        valid_pre_bins = int(
            pre_mask.sum()
        )

        rows.append(
            {
                "id": int(
                    incident["id"]
                ),
                "condition": (
                    incident[
                        "condition"
                    ]
                ),
                "start": str(
                    start
                ),
                "end": str(
                    end
                ),
                "pre_onset_window_start": (
                    str(
                        pre_start
                    )
                ),
                "expected_24h_pre_bins": (
                    expected_pre_bins
                ),
                "valid_24h_pre_bins": (
                    valid_pre_bins
                ),
                "pre_onset_coverage_fraction": (
                    valid_pre_bins
                    / expected_pre_bins
                ),
                "valid_bins_during_incident": (
                    int(
                        incident_mask.sum()
                    )
                ),
            }
        )

    return rows


def main() -> None:
    config = _load_config()

    dataset = config[
        "dataset"
    ]
    preprocessing = config[
        "preprocessing"
    ]
    splits = config[
        "split"
    ]

    raw_path = ROOT / dataset[
        "raw_csv"
    ]

    if not raw_path.exists():
        raise FileNotFoundError(
            f"MetroPT2 CSV not found: {raw_path}"
        )

    print(
        "Loading compressor signals only..."
    )

    frame = (
        load_metropt2_compressor_frame(
            str(
                raw_path
            )
        )
    )

    print(
        f"Loaded {len(frame):,} rows "
        f"and {len(frame.columns) - 1} "
        "compressor signals."
    )

    split_results = {}

    for split_name in (
        "train",
        "calibration",
        "test",
    ):
        split_config = splits[
            split_name
        ]

        print(
            f"Aggregating {split_name}..."
        )

        split_results[
            split_name
        ] = (
            aggregate_metropt2_split(
                frame,
                start=str(
                    split_config[
                        "start"
                    ]
                ),
                end=str(
                    split_config[
                        "end"
                    ]
                ),
                bin_minutes=int(
                    preprocessing[
                        "bin_minutes"
                    ]
                ),
                expected_samples_per_bin=int(
                    preprocessing[
                        "expected_samples_per_bin"
                    ]
                ),
                minimum_coverage=float(
                    preprocessing[
                        "minimum_coverage"
                    ]
                ),
                max_gap_seconds=float(
                    preprocessing[
                        "max_gap_seconds"
                    ]
                ),
            )
        )

    output_paths = {
        "train": (
            ROOT
            / preprocessing[
                "train_file"
            ]
        ),
        "calibration": (
            ROOT
            / preprocessing[
                "calibration_file"
            ]
        ),
        "test": (
            ROOT
            / preprocessing[
                "test_file"
            ]
        ),
    }

    for split_name, path in (
        output_paths.items()
    ):
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        split_results[
            split_name
        ].features.to_parquet(
            path
        )

    variants = {}

    reference_columns = (
        split_results[
            "train"
        ].features.columns
    )

    for variant_name, variant in (
        preprocessing[
            "feature_variants"
        ].items()
    ):
        columns = model_feature_columns(
            reference_columns,
            excluded_groups=variant[
                "excluded_groups"
            ],
        )

        variants[
            variant_name
        ] = {
            "excluded_groups": (
                variant[
                    "excluded_groups"
                ]
            ),
            "model_feature_count": (
                len(
                    columns
                )
            ),
        }

    train_end = pd.Timestamp(
        splits["train"][
            "end"
        ]
    )
    calibration_start = pd.Timestamp(
        splits["calibration"][
            "start"
        ]
    )
    calibration_end = pd.Timestamp(
        splits["calibration"][
            "end"
        ]
    )
    test_start = pd.Timestamp(
        splits["test"][
            "start"
        ]
    )

    incident_membership = []

    for incident in dataset[
        "reported_incidents"
    ]:
        start = pd.Timestamp(
            incident["start"]
        )
        end = pd.Timestamp(
            incident["end"]
        )

        incident_membership.append(
            {
                "id": int(
                    incident["id"]
                ),
                "condition": (
                    incident[
                        "condition"
                    ]
                ),
                "entirely_in_test": bool(
                    start
                    >= test_start
                    and end
                    > test_start
                ),
                "overlaps_train": bool(
                    start
                    <= train_end
                ),
                "overlaps_calibration": bool(
                    start
                    <= calibration_end
                    and end
                    >= calibration_start
                ),
            }
        )

    report = {
        "schema_version": (
            "aeroxai.metropt2_preprocessing.v1"
        ),
        "dataset": dataset[
            "name"
        ],
        "evidence_class": "REAL",
        "causal_claim": False,
        "protocol": {
            "aggregation": (
                "5-minute causal bins, "
                "label=right, closed=right"
            ),
            "split_aggregation": (
                "Each chronological split is "
                "aggregated independently to "
                "prevent cross-split bin leakage."
            ),
            "minimum_coverage": float(
                preprocessing[
                    "minimum_coverage"
                ]
            ),
            "expected_samples_per_bin": int(
                preprocessing[
                    "expected_samples_per_bin"
                ]
            ),
            "large_gap_rule_seconds": float(
                preprocessing[
                    "max_gap_seconds"
                ]
            ),
            "large_gap_handling": (
                "Any 5-minute bin containing "
                "the first observation after "
                "a raw gap above the threshold "
                "is excluded. Empty and "
                "under-covered bins are also "
                "excluded. No interpolation "
                "or forward fill is used."
            ),
            "gps_used": False,
        },
        "splits": {
            name: _split_summary(
                result
            )
            for name, result
            in split_results.items()
        },
        "feature_variants": variants,
        "incident_split_membership": (
            incident_membership
        ),
        "test_incident_coverage": (
            _incident_coverage(
                split_results[
                    "test"
                ].features,
                dataset[
                    "reported_incidents"
                ],
                bin_minutes=int(
                    preprocessing[
                        "bin_minutes"
                    ]
                ),
            )
        ),
        "outputs": {
            name: str(
                path
            )
            for name, path
            in output_paths.items()
        },
        "interpretation": (
            "Preprocessing and split validation "
            "only. Detector performance has not "
            "been inspected in this checkpoint."
        ),
    }

    report_path = (
        ROOT
        / preprocessing[
            "report_file"
        ]
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "splits": report[
                    "splits"
                ],
                "feature_variants": (
                    report[
                        "feature_variants"
                    ]
                ),
                "incident_split_membership": (
                    report[
                        "incident_split_membership"
                    ]
                ),
                "test_incident_coverage": (
                    report[
                        "test_incident_coverage"
                    ]
                ),
                "report": str(
                    report_path
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
