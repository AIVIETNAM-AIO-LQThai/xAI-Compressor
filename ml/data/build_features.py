from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import yaml

from ml.data.features import (
    aggregate_causal_bins,
    filter_valid_bins,
    model_feature_columns,
    select_time_range,
)
from ml.data.schema import normalize_raw_frame

ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = (
    ROOT
    / "configs"
    / "metropt.yaml"
)

REPORT_PATH = (
    ROOT
    / "docs"
    / "preprocessing_report.json"
)


def load_config() -> dict:
    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return yaml.safe_load(handle)


def frame_summary(
    frame: pd.DataFrame,
) -> dict:
    if frame.empty:
        return {
            "rows": 0,
            "start": None,
            "end": None,
        }

    return {
        "rows": len(frame),
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
    }


def main() -> None:
    config = load_config()

    dataset_config = config["dataset"]
    preprocessing = config["preprocessing"]
    split_config = config["split"]

    raw_path = (
        ROOT
        / dataset_config["raw_csv"]
    )

    print(f"Reading {raw_path} ...")

    raw = pd.read_csv(
        raw_path,
        low_memory=False,
    )

    frame = normalize_raw_frame(raw)

    print(
        f"Normalized rows: {len(frame):,}"
    )

    features = aggregate_causal_bins(
        frame,
        bin_minutes=preprocessing[
            "bin_minutes"
        ],
        expected_samples=preprocessing[
            "expected_samples_per_bin"
        ],
    )

    model_columns = model_feature_columns(
        features.columns
    )

    adequate_coverage = (
        features["coverage_ratio"]
        >= preprocessing[
            "minimum_coverage"
        ]
    )

    covered_features = cast(
        pd.DataFrame,
        features.loc[
            adequate_coverage,
            model_columns,
        ],
    )

    complete_after_coverage = (
        covered_features
        .notna()
        .all(axis=1)
    )

    clean = filter_valid_bins(
        features,
        minimum_coverage=preprocessing[
            "minimum_coverage"
        ],
    )

    train = select_time_range(
        clean,
        **split_config["train"],
    )

    calibration = select_time_range(
        clean,
        **split_config["calibration"],
    )

    test = select_time_range(
        clean,
        **split_config["test"],
    )

    outputs = {
        "features_file": clean,
        "train_file": train,
        "calibration_file": calibration,
        "test_file": test,
    }

    for key, output_frame in outputs.items():
        target = (
            ROOT
            / preprocessing[key]
        )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_frame.to_parquet(
            target,
            engine="pyarrow",
        )

        print(
            f"Wrote {len(output_frame):,} "
            f"rows → {target}"
        )

    low_coverage_bins = int(
        (~adequate_coverage).sum()
    )

    incomplete_bins = int(
        
            adequate_coverage.sum()
            - complete_after_coverage.sum()
        
    )

    incident_coverage = []

    bin_minutes = preprocessing[
        "bin_minutes"
    ]

    for incident in dataset_config[
        "reported_incidents"
    ]:
        start = pd.Timestamp(
            incident["start"]
        )

        end = pd.Timestamp(
            incident["end"]
        )

        valid_mask = (
            (clean.index >= start)
            & (clean.index <= end)
        )

        expected_bins = int(
            (
                (end - start)
                / pd.Timedelta(
                    minutes=bin_minutes
                )
            )
            + 1
        )

        valid_bins = int(
            valid_mask.sum()
        )

        incident_coverage.append(
            {
                "id": incident["id"],
                "expected_bins":
                    expected_bins,
                "valid_bins":
                    valid_bins,
                "coverage_ratio":
                    (
                        valid_bins
                        / expected_bins
                        if expected_bins
                        else None
                    ),
            }
        )

    coverage_quantiles = (
        features["coverage_ratio"]
        .quantile(
            [
                0.00,
                0.01,
                0.05,
                0.50,
                0.95,
                0.99,
                1.00,
            ]
        )
        .to_dict()
    )

    report = {
        "raw_rows": len(frame),
        "bins_before_quality_filter":
            len(features),
        "bins_after_quality_filter":
            len(clean),
        "low_coverage_bins":
            low_coverage_bins,
        "incomplete_feature_bins":
            incomplete_bins,
        "model_feature_count":
            len(model_columns),
        "quality_columns": [
            "sample_count",
            "coverage_ratio",
        ],
        "coverage_quantiles": {
            str(key): float(value)
            for key, value
            in coverage_quantiles.items()
        },
        "splits": {
            "train":
                frame_summary(train),
            "calibration":
                frame_summary(
                    calibration
                ),
            "test":
                frame_summary(test),
        },
        "incident_coverage":
            incident_coverage,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=== Preprocessing Summary ===")
    print(
        f"Model features: "
        f"{len(model_columns)}"
    )
    print(
        f"Valid bins: "
        f"{len(clean):,}"
    )
    print(
        f"Train: "
        f"{len(train):,}"
    )
    print(
        f"Calibration: "
        f"{len(calibration):,}"
    )
    print(
        f"Test: "
        f"{len(test):,}"
    )

    print()
    print(
        f"Report: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()