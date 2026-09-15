from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S.%f",
    "%d.%m.%Y %H:%M",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M:%S.%f",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S.%f",
)


DECIMAL_COMMA_PATTERN = re.compile(
    r"^[+-]?(?:\d+(?:,\d*)?|,\d+)"
    r"(?:[eE][+-]?\d+)?$"
)


def normalize_numeric_text(
    value: object,
) -> str:
    stripped = str(value).strip()

    if (
        "," in stripped
        and DECIMAL_COMMA_PATTERN.fullmatch(
            stripped
        )
    ):
        return stripped.replace(",", ".")

    return stripped


@dataclass(frozen=True)
class FiveMinuteResult:
    frame: pd.DataFrame
    source_rows: int
    produced_bins: int
    valid_bins: int
    invalid_bins: int
    missing_numeric_cells: int
    infinite_numeric_cells: int


def day_from_archive_name(name: str) -> str:
    stem = Path(name).stem

    if not stem.startswith("CoBra"):
        raise ValueError(
            f"Unexpected CoBra archive name: {name}"
        )

    raw = stem.removeprefix("CoBra")

    if len(raw) != 8 or not raw.isdigit():
        raise ValueError(
            f"Unexpected CoBra archive name: {name}"
        )

    return (
        f"{raw[:4]}-"
        f"{raw[4:6]}-"
        f"{raw[6:]}"
    )


def assert_train_only_day(
    day: str,
    *,
    train_days: set[str],
    calibration_days: set[str],
    evaluation_days: set[str],
) -> None:
    if day in calibration_days:
        raise RuntimeError(
            f"CALIBRATION raw access forbidden here: {day}"
        )

    if day in evaluation_days:
        raise RuntimeError(
            f"EVALUATION raw access forbidden here: {day}"
        )

    if day not in train_days:
        raise RuntimeError(
            f"Day is outside the frozen TRAIN partition: {day}"
        )


def parse_source_timestamp(value: str) -> datetime:
    stripped = str(value).strip()

    if not stripped:
        raise ValueError("Blank source timestamp.")

    iso_candidate = stripped.replace(
        "Z",
        "+00:00",
    )

    try:
        parsed = datetime.fromisoformat(
            iso_candidate
        )

        if parsed.tzinfo is not None:
            parsed = parsed.replace(
                tzinfo=None
            )

        return parsed
    except ValueError:
        pass

    for format_string in TIMESTAMP_FORMATS:
        try:
            parsed = pd.to_datetime(
                stripped,
                format=format_string,
                errors="raise",
            )
            return parsed.to_pydatetime()
        except ValueError:
            continue

    raise ValueError(
        f"Unsupported source timestamp: {value!r}"
    )


def parse_timestamp_series(
    values: pd.Series,
) -> pd.Series:
    parsed: list[datetime] = []

    for value in values.tolist():
        parsed.append(
            parse_source_timestamp(value)
        )

    return pd.Series(
        pd.to_datetime(parsed),
        index=values.index,
        dtype="datetime64[us]",
    )


def assign_contiguous_segments(
    timestamps: pd.Series,
    *,
    step_minutes: int = 5,
) -> np.ndarray:
    if timestamps.empty:
        return np.empty(
            0,
            dtype=np.int64,
        )

    values = pd.DatetimeIndex(
        timestamps
    )

    if not values.is_monotonic_increasing:
        raise ValueError(
            "Timestamps must be sorted before "
            "segment assignment."
        )

    expected = pd.Timedelta(
        minutes=step_minutes
    )

    segment_ids = np.zeros(
        len(values),
        dtype=np.int64,
    )

    current_segment = 0

    for index in range(1, len(values)):
        delta = (
            values[index]
            - values[index - 1]
        )

        if delta != expected:
            current_segment += 1

        segment_ids[index] = (
            current_segment
        )

    return segment_ids


def aggregate_five_minute(
    frame: pd.DataFrame,
    *,
    timestamp_column: str,
    feature_names: list[str],
) -> FiveMinuteResult:
    if timestamp_column not in frame.columns:
        raise ValueError(
            "Timestamp column is missing."
        )

    missing_features = [
        name
        for name in feature_names
        if name not in frame.columns
    ]

    if missing_features:
        raise ValueError(
            "Frozen raw features missing from "
            f"source frame: {missing_features[:5]}"
        )

    if len(feature_names) != len(
        set(feature_names)
    ):
        raise ValueError(
            "Frozen raw feature list contains "
            "duplicates."
        )

    source_rows = len(frame)

    parsed = parse_timestamp_series(
        frame[timestamp_column]
    )

    if parsed.isna().any():
        raise RuntimeError(
            "Source timestamp parsing produced NaT."
        )

    bins = parsed.dt.floor("5min")

    raw = (
        frame.loc[:, feature_names]
        .astype(str)
        .map(normalize_numeric_text)
    )

    numeric = raw.apply(
        pd.to_numeric,
        errors="coerce",
    ).astype("float64")

    numeric_values = numeric.to_numpy(
        dtype=np.float64,
        copy=False,
    )

    missing_numeric_cells = int(
        np.isnan(numeric_values).sum()
    )

    infinite_numeric_cells = int(
        np.isinf(numeric_values).sum()
    )

    if infinite_numeric_cells:
        numeric = numeric.mask(
            np.isinf(
                numeric.to_numpy(
                    dtype=np.float64,
                    copy=False,
                )
            )
        )

    numeric = numeric.copy()
    numeric.insert(
        0,
        "__bin_timestamp__",
        bins.to_numpy(),
    )

    grouped = numeric.groupby(
        "__bin_timestamp__",
        sort=True,
        observed=True,
    )

    means = grouped[
        feature_names
    ].mean()

    finite_counts = grouped[
        feature_names
    ].count()

    valid_mask = finite_counts.ge(
        1
    ).all(axis=1)

    produced_bins = len(means)
    valid_bins = int(
        valid_mask.sum()
    )
    invalid_bins = int(
        produced_bins - valid_bins
    )

    valid = (
        means.loc[
            valid_mask,
            feature_names,
        ]
        .reset_index()
        .rename(
            columns={
                "__bin_timestamp__": (
                    "timestamp"
                )
            }
        )
    )

    if not valid.empty:
        values = valid[
            feature_names
        ].to_numpy(
            dtype=np.float64,
        )

        if not np.isfinite(values).all():
            raise RuntimeError(
                "Valid five-minute representation "
                "contains non-finite values."
            )

    valid["segment_id"] = (
        assign_contiguous_segments(
            valid["timestamp"],
            step_minutes=5,
        )
    )

    ordered = [
        "timestamp",
        "segment_id",
        *feature_names,
    ]

    valid = valid.loc[:, ordered]

    return FiveMinuteResult(
        frame=valid,
        source_rows=source_rows,
        produced_bins=produced_bins,
        valid_bins=valid_bins,
        invalid_bins=invalid_bins,
        missing_numeric_cells=(
            missing_numeric_cells
        ),
        infinite_numeric_cells=(
            infinite_numeric_cells
        ),
    )


def semantic_representation_sha256(
    frame: pd.DataFrame,
    *,
    feature_names: list[str],
) -> str:
    required = {
        "day",
        "timestamp",
        "segment_id",
        *feature_names,
    }

    if not required.issubset(
        frame.columns
    ):
        raise ValueError(
            "Representation is missing required "
            "columns."
        )

    digest = hashlib.sha256()

    digest.update(
        b"aeroxai.cobra.train_5min.v1\n"
    )

    for name in feature_names:
        digest.update(
            name.encode("utf-8")
        )
        digest.update(b"\n")

    for row in frame.itertuples(
        index=False,
        name=None,
    ):
        row_map = dict(
            zip(
                frame.columns,
                row,
                strict=True,
            )
        )

        day = str(
            row_map["day"]
        )
        timestamp = pd.Timestamp(
            row_map["timestamp"]
        )
        segment_id = int(
            row_map["segment_id"]
        )

        digest.update(
            day.encode("ascii")
        )
        digest.update(b"\0")

        digest.update(
            np.asarray(
                [timestamp.value],
                dtype="<i8",
            ).tobytes()
        )

        digest.update(
            np.asarray(
                [segment_id],
                dtype="<i8",
            ).tobytes()
        )

        feature_values = np.asarray(
            [
                row_map[name]
                for name in feature_names
            ],
            dtype="<f8",
        )

        if not np.isfinite(
            feature_values
        ).all():
            raise ValueError(
                "Semantic hash received "
                "non-finite feature values."
            )

        digest.update(
            feature_values.tobytes(
                order="C"
            )
        )

    return digest.hexdigest()
