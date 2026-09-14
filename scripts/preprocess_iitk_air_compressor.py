from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import yaml

from ml.data.iitk_air_compressor import (
    FEATURE_GROUPS,
    extract_recording_features,
    feature_group_columns,
    healthy_split_for_reading,
    read_recording,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "iitk_air_compressor.yaml"
)


def _load_config() -> dict:
    return yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )


def main() -> None:
    config = _load_config()

    dataset = config[
        "dataset"
    ]

    preprocessing = config[
        "preprocessing"
    ]

    zip_path = ROOT / dataset[
        "raw_zip"
    ]

    healthy_directory = (
        dataset[
            "conditions"
        ][
            "healthy"
        ]
    )

    faults = dataset[
        "conditions"
    ][
        "faults"
    ]

    condition_labels = {
        "healthy": (
            healthy_directory
        ),
        **faults,
    }

    rows = []

    band_edges = (
        preprocessing[
            "feature_protocol"
        ][
            "spectral_band_edges"
        ]
    )

    with ZipFile(
        zip_path
    ) as archive:
        for label, condition in (
            condition_labels.items()
        ):
            for reading in range(
                1,
                int(
                    dataset[
                        "readings_per_condition"
                    ]
                )
                + 1,
            ):
                recording = (
                    read_recording(
                        archive,
                        archive_root=dataset[
                            "archive_root"
                        ],
                        condition=condition,
                        reading=reading,
                    )
                )

                if recording.values.size != int(
                    dataset[
                        "samples_per_recording"
                    ]
                ):
                    raise RuntimeError(
                        "Unexpected recording "
                        f"length for "
                        f"{condition}/"
                        f"Reading{reading}: "
                        f"{recording.values.size}"
                    )

                features = (
                    extract_recording_features(
                        recording.values,
                        spectral_band_edges=(
                            band_edges
                        ),
                    )
                )

                if label == (
                    "healthy"
                ):
                    split = (
                        healthy_split_for_reading(
                            reading,
                            preprocessing[
                                "healthy_split"
                            ],
                        )
                    )
                    is_fault = False
                else:
                    split = "test"
                    is_fault = True

                rows.append(
                    {
                        "condition": (
                            label
                        ),
                        "archive_condition": (
                            condition
                        ),
                        "reading": (
                            reading
                        ),
                        "split": (
                            split
                        ),
                        "is_fault": (
                            is_fault
                        ),
                        **features,
                    }
                )

                if (
                    reading % 25
                    == 0
                ):
                    print(
                        f"{condition}: "
                        f"{reading}/"
                        f"{dataset['readings_per_condition']}"
                    )

    frame = pd.DataFrame(
        rows
    )

    feature_columns = list(
        FEATURE_GROUPS
    )

    if frame[
        feature_columns
    ].isna().any().any():
        raise RuntimeError(
            "Feature table contains NaN."
        )

    train = frame.loc[
        frame[
            "split"
        ]
        == "train"
    ]

    calibration = frame.loc[
        frame[
            "split"
        ]
        == "calibration"
    ]

    test = frame.loc[
        frame[
            "split"
        ]
        == "test"
    ]

    if train[
        "is_fault"
    ].any():
        raise RuntimeError(
            "Fault recording leaked into train."
        )

    if calibration[
        "is_fault"
    ].any():
        raise RuntimeError(
            "Fault recording leaked into calibration."
        )

    expected_train = 120
    expected_calibration = 60
    expected_healthy_test = 45

    if len(
        train
    ) != expected_train:
        raise RuntimeError(
            "Healthy train split changed."
        )

    if len(
        calibration
    ) != expected_calibration:
        raise RuntimeError(
            "Healthy calibration split changed."
        )

    healthy_test_count = int(
        (
            (
                test[
                    "condition"
                ]
                == "healthy"
            )
        ).sum()
    )

    if healthy_test_count != (
        expected_healthy_test
    ):
        raise RuntimeError(
            "Healthy test split changed."
        )

    output_path = (
        ROOT
        / preprocessing[
            "output_file"
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_parquet(
        output_path,
        index=False,
    )

    class_counts = (
        frame.groupby(
            [
                "split",
                "condition",
            ]
        )
        .size()
        .rename(
            "count"
        )
        .reset_index()
    )

    report = {
        "schema_version": (
            "aeroxai.iitk_air_compressor_preprocessing.v1"
        ),
        "dataset": (
            dataset[
                "short_name"
            ]
        ),
        "recordings": len(
                frame
            ),
        "feature_count": len(
                feature_columns
            ),
        "feature_groups": (
            feature_group_columns(
                feature_columns
            )
        ),
        "split_counts": {
            "train": len(
                    train
                ),
            "calibration": len(
                    calibration
                ),
            "test": len(
                    test
                ),
            "test_healthy": (
                healthy_test_count
            ),
            "test_fault": int(
                test[
                    "is_fault"
                ].sum()
            ),
        },
        "class_counts": (
            class_counts.to_dict(
                orient="records"
            )
        ),
        "leakage_checks": {
            "train_contains_fault": bool(
                train[
                    "is_fault"
                ].any()
            ),
            "calibration_contains_fault": bool(
                calibration[
                    "is_fault"
                ].any()
            ),
            "faults_test_only": bool(
                frame.loc[
                    frame[
                        "is_fault"
                    ],
                    "split",
                ]
                .eq(
                    "test"
                )
                .all()
            ),
        },
        "frequency_semantics": {
            "sample_rate_hz": (
                dataset[
                    "sample_rate_hz"
                ]
            ),
            "sample_rate_status": (
                dataset[
                    "sample_rate_status"
                ]
            ),
            "absolute_frequency_features": (
                dataset[
                    "absolute_frequency_features"
                ]
            ),
            "spectral_axis": (
                "fraction_of_nyquist"
            ),
        },
        "output_file": str(
            output_path
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
            report,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
