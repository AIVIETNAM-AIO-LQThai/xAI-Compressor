from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.data.cobra import (
    aggregate_five_minute,
    assert_train_only_day,
    day_from_archive_name,
    semantic_representation_sha256,
)

ROOT = Path(__file__).resolve().parents[1]

ADAPTATION_CONFIG_PATH = (
    ROOT
    / "configs/cobra_router_v2_adaptation.yaml"
)

SCHEMA_CONFIG_PATH = (
    ROOT
    / "configs/cobra_schema_chronology.yaml"
)

SCHEMA_MANIFEST_PATH = (
    ROOT
    / "docs/research/cobra_schema_chronology_manifest.json"
)

SOURCE_LOCK_PATH = (
    ROOT
    / "docs/research/cobra_source_lock_manifest.json"
)

FEATURE_CONTRACT_PATH = (
    ROOT
    / "docs/research/cobra_router_v2_raw_feature_contract.json"
)

OUTPUT_DATA_PATH = (
    ROOT
    / "data/processed/cobra_router_v2/train_5min.csv"
)

OUTPUT_RESULT_PATH = (
    ROOT
    / "docs/research/cobra_router_v2_train_ingestion.json"
)

REQUIRED_FEATURE_CONTRACT_COMMIT = (
    "f507a90fdb0ce005602d231f16ccccb6d0b75ec9"
)

EXPECTED_FEATURE_SHA256 = (
    "ba54ffa70cda5690ceaef4d0a58d6d14129dd36a0e03abbc95c20344e3e4a9aa"
)

IMPLEMENTATION_PATHS = [
    "ml/data/cobra.py",
    "scripts/build_cobra_train_representation.py",
    "tests/test_cobra_train_representation.py",
]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected JSON mapping in {path}."
        )

    return value


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected YAML mapping in {path}."
        )

    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def _git_head() -> str:
    result = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return result.stdout.strip()


def _verify_execution_boundary() -> None:
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            REQUIRED_FEATURE_CONTRACT_COMMIT,
            "HEAD",
        ],
        cwd=ROOT,
        check=False,
    )

    if ancestor.returncode != 0:
        raise RuntimeError(
            "Frozen CoBra raw-feature contract "
            "is not an ancestor of HEAD."
        )

    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            *IMPLEMENTATION_PATHS,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    if status.stdout.strip():
        raise RuntimeError(
            "CoBra TRAIN ingestion implementation "
            "is not clean and committed."
        )

    if OUTPUT_RESULT_PATH.exists():
        raise RuntimeError(
            "TRAIN ingestion result already exists; "
            "refusing to overwrite."
        )

    if OUTPUT_DATA_PATH.exists():
        raise RuntimeError(
            "TRAIN representation already exists; "
            "refusing to overwrite."
        )


def _validate_adaptation_split_counts(
    adaptation_split: dict[str, Any],
    *,
    train_days: set[str],
    calibration_days: set[str],
    evaluation_days: set[str],
) -> None:
    expected = {
        "train_days": len(train_days),
        "calibration_days": len(calibration_days),
        "evaluation_days": len(evaluation_days),
    }

    for key, expected_count in expected.items():
        value = adaptation_split.get(key)

        if (
            isinstance(value, bool)
            or not isinstance(value, int)
        ):
            raise TypeError(
                f"Adaptation {key} must be an "
                "integer count."
            )

        if value != expected_count:
            raise RuntimeError(
                f"Adaptation {key} count differs "
                "from the frozen schema split: "
                f"{value} != {expected_count}."
            )

    if adaptation_split.get(
        "reassignment_allowed"
    ) is not False:
        raise RuntimeError(
            "CoBra split reassignment must remain "
            "forbidden."
        )


def _archive_metadata_by_day(
    schema_manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    result: dict[
        str,
        dict[str, Any],
    ] = {}

    for item in schema_manifest[
        "archives"
    ]:
        archive = str(
            item["archive"]
        )

        day = day_from_archive_name(
            archive
        )

        if day in result:
            raise RuntimeError(
                f"Duplicate archive day: {day}"
            )

        result[day] = item

    return result


def _read_train_archive(
    *,
    archive_path: Path,
    archive_metadata: dict[str, Any],
    feature_names: list[str],
) -> pd.DataFrame:
    primary_member = str(
        archive_metadata[
            "primary_csv_member"
        ]
    )

    source_encoding = str(
        archive_metadata[
            "source_encoding"
        ]
    )

    delimiter = str(
        archive_metadata[
            "delimiter"
        ]
    )

    timestamp_column = str(
        archive_metadata[
            "timestamp_column"
        ]
    )

    required_columns = [
        timestamp_column,
        *feature_names,
    ]

    source_columns = [
        str(value)
        for value in archive_metadata[
            "columns"
        ]
    ]

    missing = [
        name
        for name in required_columns
        if name not in source_columns
    ]

    if missing:
        raise RuntimeError(
            "Frozen feature(s) absent from TRAIN "
            f"archive {archive_path.name}: "
            f"{missing[:5]}"
        )

    with zipfile.ZipFile(
        archive_path,
        mode="r",
    ) as archive:
        members = archive.namelist()

        if primary_member not in members:
            raise RuntimeError(
                "Frozen primary CSV member is "
                f"missing from {archive_path.name}."
            )

        with archive.open(
            primary_member,
            mode="r",
        ) as raw, io.TextIOWrapper(
            raw,
            encoding=source_encoding,
            errors="strict",
            newline="",
        ) as text:
            frame = pd.read_csv(
                text,
                sep=delimiter,
                usecols=required_columns,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
            )

    return frame.loc[
        :,
        required_columns,
    ]


def main() -> None:
    _verify_execution_boundary()

    adaptation = _yaml(
        ADAPTATION_CONFIG_PATH
    )
    schema_config = _yaml(
        SCHEMA_CONFIG_PATH
    )
    schema_manifest = _json(
        SCHEMA_MANIFEST_PATH
    )
    source_lock = _json(
        SOURCE_LOCK_PATH
    )
    feature_contract = _json(
        FEATURE_CONTRACT_PATH
    )

    feature_names = [
        str(value)
        for value in feature_contract[
            "final_raw_features"
        ]
    ]

    if len(feature_names) != 219:
        raise RuntimeError(
            "Frozen final CoBra raw-feature "
            "count changed."
        )

    if feature_contract[
        "final_raw_feature_sha256"
    ] != EXPECTED_FEATURE_SHA256:
        raise RuntimeError(
            "Frozen raw-feature SHA256 changed."
        )

    if feature_contract[
        "status"
    ] != "FROZEN_METADATA_ONLY_RAW_FEATURE_SET":
        raise RuntimeError(
            "Unexpected raw-feature contract status."
        )

    split = schema_config[
        "split"
    ]

    train_days = {
        str(value)
        for value in split[
            "train_days"
        ]
    }

    calibration_days = {
        str(value)
        for value in split[
            "calibration_days"
        ]
    }

    evaluation_days = {
        str(value)
        for value in split[
            "evaluation_days"
        ]
    }

    if len(train_days) != 14:
        raise RuntimeError(
            "Frozen TRAIN day count changed."
        )

    if len(calibration_days) != 4:
        raise RuntimeError(
            "Frozen CALIBRATION day count changed."
        )

    if len(evaluation_days) != 5:
        raise RuntimeError(
            "Frozen EVALUATION day count changed."
        )

    if (
        train_days
        & calibration_days
        or train_days
        & evaluation_days
        or calibration_days
        & evaluation_days
    ):
        raise RuntimeError(
            "Frozen split partitions overlap."
        )

    adaptation_split = adaptation[
        "split"
    ]

    _validate_adaptation_split_counts(
        adaptation_split,
        train_days=train_days,
        calibration_days=calibration_days,
        evaluation_days=evaluation_days,
    )

    archive_by_day = (
        _archive_metadata_by_day(
            schema_manifest
        )
    )

    source_dir = (
        ROOT
        / schema_config["source"][
            "directory"
        ]
    )

    primary_files = source_lock[
        "primary_local_files"
    ]

    ordered_train_days = [
        str(value)
        for value in split[
            "train_days"
        ]
    ]

    day_frames: list[
        pd.DataFrame
    ] = []

    per_day: list[
        dict[str, Any]
    ] = []

    opened_archives: list[str] = []

    global_segment_offset = 0

    for index, day in enumerate(
        ordered_train_days,
        start=1,
    ):
        assert_train_only_day(
            day,
            train_days=train_days,
            calibration_days=(
                calibration_days
            ),
            evaluation_days=(
                evaluation_days
            ),
        )

        metadata = archive_by_day[
            day
        ]

        if metadata[
            "partition"
        ] != "TRAIN":
            raise RuntimeError(
                "Schema metadata does not mark "
                f"{day} as TRAIN."
            )

        archive_name = str(
            metadata[
                "archive"
            ]
        )

        expected_name = (
            "CoBra"
            + day.replace("-", "")
            + ".zip"
        )

        if archive_name != expected_name:
            raise RuntimeError(
                "TRAIN archive/day mapping changed."
            )

        source_record = primary_files[
            archive_name
        ]

        archive_path = (
            ROOT
            / source_record[
                "path"
            ]
        )

        if archive_path.parent.resolve() != (
            source_dir.resolve()
        ):
            raise RuntimeError(
                "TRAIN archive path differs from "
                "frozen source directory."
            )

        if not archive_path.is_file():
            raise FileNotFoundError(
                archive_path
            )

        observed_sha = _sha256(
            archive_path
        )

        if observed_sha != source_record[
            "sha256"
        ]:
            raise RuntimeError(
                "Source-lock SHA256 mismatch for "
                f"{archive_name}."
            )

        print(
            f"[{index}/14] opening TRAIN "
            f"{archive_name}",
            flush=True,
        )

        source_frame = (
            _read_train_archive(
                archive_path=archive_path,
                archive_metadata=metadata,
                feature_names=feature_names,
            )
        )

        opened_archives.append(
            archive_name
        )

        five_minute = (
            aggregate_five_minute(
                source_frame,
                timestamp_column=str(
                    metadata[
                        "timestamp_column"
                    ]
                ),
                feature_names=feature_names,
            )
        )

        day_frame = (
            five_minute.frame.copy()
        )

        if not day_frame.empty:
            local_segments = (
                day_frame[
                    "segment_id"
                ]
                .to_numpy(
                    dtype="int64"
                )
            )

            day_frame[
                "segment_id"
            ] = (
                local_segments
                + global_segment_offset
            )

            global_segment_offset = (
                int(
                    day_frame[
                        "segment_id"
                    ].max()
                )
                + 1
            )

        day_frame.insert(
            0,
            "day",
            day,
        )

        day_frames.append(
            day_frame
        )

        segment_count = (
            int(
                day_frame[
                    "segment_id"
                ].nunique()
            )
            if not day_frame.empty
            else 0
        )

        per_day.append(
            {
                "day": day,
                "archive": archive_name,
                "source_sha256": (
                    observed_sha
                ),
                "source_rows": (
                    five_minute.source_rows
                ),
                "produced_5min_bins": (
                    five_minute.produced_bins
                ),
                "valid_5min_bins": (
                    five_minute.valid_bins
                ),
                "invalid_5min_bins": (
                    five_minute.invalid_bins
                ),
                "missing_numeric_cells": (
                    five_minute.missing_numeric_cells
                ),
                "infinite_numeric_cells": (
                    five_minute.infinite_numeric_cells
                ),
                "contiguous_segments": (
                    segment_count
                ),
            }
        )

    expected_opened = [
        "CoBra"
        + day.replace("-", "")
        + ".zip"
        for day in ordered_train_days
    ]

    if opened_archives != expected_opened:
        raise RuntimeError(
            "Opened TRAIN archive sequence "
            "changed."
        )

    combined = pd.concat(
        day_frames,
        axis=0,
        ignore_index=True,
    )

    ordered_columns = [
        "day",
        "timestamp",
        "segment_id",
        *feature_names,
    ]

    combined = combined.loc[
        :,
        ordered_columns,
    ]

    if len(combined.columns) != (
        3 + 219
    ):
        raise RuntimeError(
            "TRAIN representation column "
            "count changed."
        )

    if (
        not combined.empty
        and combined[
            ["day", "timestamp"]
        ].duplicated().any()
    ):
        raise RuntimeError(
            "Duplicate day/timestamp bins "
            "remain after aggregation."
        )

    semantic_sha = (
        semantic_representation_sha256(
            combined,
            feature_names=feature_names,
        )
    )

    OUTPUT_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        OUTPUT_DATA_PATH,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        date_format="%Y-%m-%d %H:%M:%S",
        float_format="%.17g",
    )

    output_file_sha = _sha256(
        OUTPUT_DATA_PATH
    )

    result = {
        "schema_version": (
            "aeroxai.cobra_router_v2_train_ingestion.v1"
        ),
        "study": (
            "cobra_router_v2_adaptation"
        ),
        "status": (
            "TRAIN_REPRESENTATION_BUILT"
        ),
        "implementation_commit": (
            _git_head()
        ),
        "raw_feature_contract_commit": (
            REQUIRED_FEATURE_CONTRACT_COMMIT
        ),
        "raw_feature_count": 219,
        "raw_feature_sha256": (
            EXPECTED_FEATURE_SHA256
        ),
        "partition_opened": "TRAIN",
        "train_day_count": 14,
        "opened_train_archives": (
            opened_archives
        ),
        "calibration_archives_opened": (
            False
        ),
        "evaluation_archives_opened": (
            False
        ),
        "representation": {
            "bin_minutes": 5,
            "aggregation": (
                "arithmetic_mean_of_finite_observed_samples"
            ),
            "invalid_bin_rule": (
                "invalid_if_any_frozen_feature_has_no_finite_sample"
            ),
            "interpolation": False,
            "forward_fill": False,
            "backward_fill": False,
            "fabricated_seconds": False,
            "fabricated_bins": False,
            "day_boundary_reset": True,
            "exact_contiguity_minutes": 5,
        },
        "totals": {
            "source_rows": sum(
                int(item["source_rows"])
                for item in per_day
            ),
            "produced_5min_bins": sum(
                int(
                    item[
                        "produced_5min_bins"
                    ]
                )
                for item in per_day
            ),
            "valid_5min_bins": len(
                combined
            ),
            "invalid_5min_bins": sum(
                int(
                    item[
                        "invalid_5min_bins"
                    ]
                )
                for item in per_day
            ),
            "contiguous_segments": (
                int(
                    combined[
                        "segment_id"
                    ].nunique()
                )
                if not combined.empty
                else 0
            ),
        },
        "per_day": per_day,
        "output": {
            "path": str(
                OUTPUT_DATA_PATH.relative_to(
                    ROOT
                )
            ),
            "file_sha256": (
                output_file_sha
            ),
            "semantic_sha256": (
                semantic_sha
            ),
            "rows": len(
                combined
            ),
            "columns": len(
                combined.columns
            ),
        },
        "model_outcomes": {
            "pca_fitted": False,
            "pca_scores_computed": False,
            "ewma_scores_computed": False,
            "tcn_fitted": False,
            "tcn_scores_computed": False,
            "router_scores_computed": False,
            "teacher_thresholds_computed": False,
            "energy_metrics_computed": False,
        },
        "firewall": {
            "calibration_sensor_values_used": (
                False
            ),
            "evaluation_sensor_values_used": (
                False
            ),
            "evaluation_behavior_used": (
                False
            ),
        },
    }

    OUTPUT_RESULT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": result[
                    "status"
                ],
                "train_days_opened": 14,
                "source_rows": result[
                    "totals"
                ][
                    "source_rows"
                ],
                "valid_5min_bins": result[
                    "totals"
                ][
                    "valid_5min_bins"
                ],
                "invalid_5min_bins": result[
                    "totals"
                ][
                    "invalid_5min_bins"
                ],
                "contiguous_segments": result[
                    "totals"
                ][
                    "contiguous_segments"
                ],
                "semantic_sha256": (
                    semantic_sha
                ),
                "calibration_sensor_values_used": (
                    False
                ),
                "evaluation_sensor_values_used": (
                    False
                ),
                "result": str(
                    OUTPUT_RESULT_PATH.relative_to(
                        ROOT
                    )
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
