from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

CORE_ANALOG_COLUMNS = (
    "tp2",
    "tp3",
    "h1",
    "dv_pressure",
    "reservoirs",
    "oil_temperature",
    "flowmeter",
    "motor_current",
)

CORE_DIGITAL_COLUMNS = (
    "comp",
    "lps",
)

REQUIRED_CORE_COLUMNS = (
    "timestamp",
    *CORE_ANALOG_COLUMNS,
    *CORE_DIGITAL_COLUMNS,
)

NOMINAL_1HZ_MIN_SECONDS = 0.5
NOMINAL_1HZ_MAX_SECONDS = 1.5


def canonicalize_metropt2_column_name(
    name: object,
) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def canonicalize_metropt2_columns(
    columns: Sequence[object],
) -> list[str]:
    canonical = [
        canonicalize_metropt2_column_name(column)
        for column in columns
    ]

    duplicates = sorted(
        {
            column
            for column in canonical
            if canonical.count(column) > 1
        }
    )

    if duplicates:
        raise ValueError(
            "MetroPT2 columns collapse to duplicate "
            f"canonical names: {duplicates}"
        )

    return canonical


def normalize_metropt2_chunk(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = canonicalize_metropt2_columns(
        normalized.columns
    )

    missing = sorted(
        set(REQUIRED_CORE_COLUMNS)
        - set(normalized.columns)
    )

    if missing:
        raise ValueError(
            "Missing required MetroPT2 core columns: "
            f"{missing}"
        )

    normalized["timestamp"] = pd.to_datetime(
        normalized["timestamp"],
        errors="coerce",
    )

    return normalized


def read_metropt2_header(
    path: str | Path,
) -> dict[str, list[str]]:
    csv_path = Path(path)

    frame = pd.read_csv(
        csv_path,
        nrows=0,
    )

    raw_columns = [
        str(column)
        for column in frame.columns
    ]

    canonical_columns = (
        canonicalize_metropt2_columns(
            raw_columns
        )
    )

    missing = sorted(
        set(REQUIRED_CORE_COLUMNS)
        - set(canonical_columns)
    )

    return {
        "raw_columns": raw_columns,
        "canonical_columns": canonical_columns,
        "missing_core_columns": missing,
    }


def iter_metropt2_chunks(
    path: str | Path,
    *,
    chunksize: int,
) -> Iterator[pd.DataFrame]:
    if chunksize <= 0:
        raise ValueError(
            "chunksize must be positive."
        )

    for frame in pd.read_csv(
        Path(path),
        chunksize=chunksize,
        low_memory=False,
    ):
        yield normalize_metropt2_chunk(
            frame
        )


def file_digests(
    path: str | Path,
) -> dict[str, str]:
    file_path = Path(path)

    md5 = hashlib.md5(
        usedforsecurity=False
    )
    sha256 = hashlib.sha256()

    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(
                8 * 1024 * 1024
            )
            if not chunk:
                break

            md5.update(chunk)
            sha256.update(chunk)

    return {
        "md5": md5.hexdigest(),
        "sha256": sha256.hexdigest(),
    }


def audit_metropt2_csv(
    path: str | Path,
    *,
    chunksize: int,
    reported_incidents: Sequence[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    csv_path = Path(path)

    header = read_metropt2_header(
        csv_path
    )

    if header["missing_core_columns"]:
        raise ValueError(
            "MetroPT2 header is missing required "
            "core columns: "
            f"{header['missing_core_columns']}"
        )

    row_count = 0
    valid_timestamp_count = 0
    invalid_timestamp_count = 0

    missing_counts = {
        column: 0
        for column in header[
            "canonical_columns"
        ]
    }

    timestamp_min: pd.Timestamp | None = None
    timestamp_max: pd.Timestamp | None = None
    previous_timestamp: pd.Timestamp | None = None

    positive_step_count = 0
    nominal_1hz_steps = 0
    short_positive_steps_lt_0_5s = 0
    gap_steps_gt_1_5s = 0
    gap_steps_gt_10s = 0
    duplicate_adjacent_steps = 0
    backward_steps = 0
    largest_gap_seconds = 0.0

    incident_counts = {
        int(item["id"]): 0
        for item in reported_incidents
    }

    parsed_incidents = [
        {
            "id": int(item["id"]),
            "condition": str(
                item["condition"]
            ),
            "start": pd.Timestamp(
                item["start"]
            ),
            "end": pd.Timestamp(
                item["end"]
            ),
        }
        for item in reported_incidents
    ]

    for frame in iter_metropt2_chunks(
        csv_path,
        chunksize=chunksize,
    ):
        row_count += len(frame)

        for column in frame.columns:
            missing_counts[column] += int(
                frame[column].isna().sum()
            )

        timestamps = frame[
            "timestamp"
        ]

        invalid_timestamp_count += int(
            timestamps.isna().sum()
        )

        valid = timestamps.dropna()

        if valid.empty:
            continue

        valid_timestamp_count += len(valid)

        chunk_min = valid.min()
        chunk_max = valid.max()

        if (
            timestamp_min is None
            or chunk_min < timestamp_min
        ):
            timestamp_min = chunk_min

        if (
            timestamp_max is None
            or chunk_max > timestamp_max
        ):
            timestamp_max = chunk_max

        cadence_values = valid

        if previous_timestamp is not None:
            cadence_values = pd.concat(
                [
                    pd.Series(
                        [previous_timestamp]
                    ),
                    valid.reset_index(
                        drop=True
                    ),
                ],
                ignore_index=True,
            )

        deltas = (
            cadence_values.diff()
            .dropna()
            .dt.total_seconds()
        )

        duplicate_adjacent_steps += int(
            (deltas == 0.0).sum()
        )
        backward_steps += int(
            (deltas < 0.0).sum()
        )

        positive = deltas[
            deltas > 0.0
        ]

        positive_step_count += len(
            positive
        )

        nominal_1hz_steps += int(
            (
                (
                    positive
                    >= NOMINAL_1HZ_MIN_SECONDS
                )
                & (
                    positive
                    <= NOMINAL_1HZ_MAX_SECONDS
                )
            ).sum()
        )

        short_positive_steps_lt_0_5s += int(
            (
                positive
                < NOMINAL_1HZ_MIN_SECONDS
            ).sum()
        )

        gap_steps_gt_1_5s += int(
            (
                positive
                > NOMINAL_1HZ_MAX_SECONDS
            ).sum()
        )

        gap_steps_gt_10s += int(
            (
                positive
                > 10.0
            ).sum()
        )

        if not positive.empty:
            largest_gap_seconds = max(
                largest_gap_seconds,
                float(positive.max()),
            )

        previous_timestamp = valid.iloc[
            -1
        ]

        for incident in parsed_incidents:
            mask = (
                (timestamps >= incident["start"])
                & (
                    timestamps
                    <= incident["end"]
                )
            )

            incident_counts[
                incident["id"]
            ] += int(mask.sum())

    cadence_denominator = max(
        positive_step_count,
        1,
    )

    observed_mean_interval_seconds = None

    if (
        timestamp_min is not None
        and timestamp_max is not None
        and valid_timestamp_count > 1
    ):
        observed_mean_interval_seconds = (
            (
                timestamp_max
                - timestamp_min
            ).total_seconds()
            / (
                valid_timestamp_count
                - 1
            )
        )

    incident_audit = []

    for incident in parsed_incidents:
        within_dataset = bool(
            timestamp_min is not None
            and timestamp_max is not None
            and (
                timestamp_min
                <= incident["start"]
                <= timestamp_max
            )
            and (
                timestamp_min
                <= incident["end"]
                <= timestamp_max
            )
        )

        incident_audit.append(
            {
                "id": incident["id"],
                "condition": (
                    incident["condition"]
                ),
                "start": str(
                    incident["start"]
                ),
                "end": str(
                    incident["end"]
                ),
                "within_dataset_range": (
                    within_dataset
                ),
                "rows_in_interval": (
                    incident_counts[
                        incident["id"]
                    ]
                ),
            }
        )

    return {
        "path": str(csv_path),
        "row_count": row_count,
        "attribute_count": len(
            header["canonical_columns"]
        ),
        "raw_columns": (
            header["raw_columns"]
        ),
        "canonical_columns": (
            header["canonical_columns"]
        ),
        "required_core_columns": list(
            REQUIRED_CORE_COLUMNS
        ),
        "missing_core_columns": (
            header["missing_core_columns"]
        ),
        "timestamp_start": (
            None
            if timestamp_min is None
            else str(timestamp_min)
        ),
        "timestamp_end": (
            None
            if timestamp_max is None
            else str(timestamp_max)
        ),
        "valid_timestamp_count": (
            valid_timestamp_count
        ),
        "invalid_timestamp_count": (
            invalid_timestamp_count
        ),
        "missing_counts": (
            missing_counts
        ),
        "cadence": {
            "published_logging_rate_hz": 1.0,
            "nominal_window_seconds": [
                NOMINAL_1HZ_MIN_SECONDS,
                NOMINAL_1HZ_MAX_SECONDS,
            ],
            "positive_step_count": (
                positive_step_count
            ),
            "nominal_1hz_steps": (
                nominal_1hz_steps
            ),
            "nominal_1hz_step_fraction": (
                nominal_1hz_steps
                / cadence_denominator
            ),
            "short_positive_steps_lt_0_5s": (
                short_positive_steps_lt_0_5s
            ),
            "gap_steps_gt_1_5s": (
                gap_steps_gt_1_5s
            ),
            "gap_steps_gt_10s": (
                gap_steps_gt_10s
            ),
            "duplicate_adjacent_steps": (
                duplicate_adjacent_steps
            ),
            "backward_steps": (
                backward_steps
            ),
            "largest_positive_gap_seconds": (
                largest_gap_seconds
            ),
            "observed_mean_interval_seconds": (
                observed_mean_interval_seconds
            ),
        },
        "reported_incidents": (
            incident_audit
        ),
    }
