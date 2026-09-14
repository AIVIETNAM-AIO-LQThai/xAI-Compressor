from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RCSDSplitResult:
    train: pd.DataFrame
    calibration: pd.DataFrame
    evaluation: pd.DataFrame
    timestamp_column: str
    source_rows: int
    train_dropped_rows: int
    calibration_dropped_rows: int


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_mask(
    index: pd.DatetimeIndex,
    *,
    start: str,
    end_exclusive: str | None = None,
    end_inclusive: str | None = None,
) -> np.ndarray:
    start_ts = pd.Timestamp(start)
    mask = index >= start_ts

    if end_exclusive is not None:
        mask &= index < pd.Timestamp(end_exclusive)

    if end_inclusive is not None:
        mask &= index <= pd.Timestamp(end_inclusive)

    return np.asarray(mask, dtype=bool)


def _coerce_sensors(
    frame: pd.DataFrame,
    *,
    sensors: list[str],
) -> tuple[pd.DataFrame, int]:
    converted = frame.copy()

    for sensor in sensors:
        converted[sensor] = pd.to_numeric(
            converted[sensor],
            errors="coerce",
        )

    values = converted.loc[:, sensors].to_numpy(dtype=float)
    finite_rows = np.isfinite(values).all(axis=1)
    dropped = int((~finite_rows).sum())

    cleaned = converted.loc[finite_rows].copy()

    for sensor in sensors:
        cleaned[sensor] = cleaned[sensor].astype("float64")

    return cleaned, dropped


def load_and_split_rcsd1yd(
    source_path: Path,
    *,
    sensors: list[str],
    expected_total_columns: int,
    expected_start: str,
    expected_end: str,
    train_cfg: dict[str, Any],
    calibration_cfg: dict[str, Any],
    evaluation_cfg: dict[str, Any],
) -> RCSDSplitResult:
    try:
        raw = pd.read_excel(
            source_path,
            sheet_name=0,
            engine="openpyxl",
        )
    except ImportError as exc:
        raise RuntimeError(
            "Reading RCSD-1YD requires openpyxl. "
            "Install it in the active environment."
        ) from exc

    if raw.shape[1] != expected_total_columns:
        raise RuntimeError(
            "RCSD-1YD column count mismatch: "
            f"expected {expected_total_columns}, got {raw.shape[1]}."
        )

    timestamp_column = str(raw.columns[0]).strip()

    if timestamp_column.lower() != "timestamp":
        raise RuntimeError(
            "Frozen protocol requires the first source column "
            f"to be Timestamp; observed {timestamp_column!r}."
        )

    observed_sensors = [str(value).strip() for value in raw.columns[1:]]

    if observed_sensors != sensors:
        missing = [value for value in sensors if value not in observed_sensors]
        extra = [value for value in observed_sensors if value not in sensors]
        raise RuntimeError(
            "Frozen 25-sensor schema mismatch. "
            f"Missing={missing}; extra={extra}; "
            "column order must also match the preregistration."
        )

    raw.columns = [timestamp_column, *observed_sensors]

    timestamps = pd.to_datetime(
        raw[timestamp_column],
        errors="coerce",
    )

    if timestamps.isna().any():
        raise RuntimeError(
            "RCSD-1YD contains an unparseable timestamp."
        )

    index = pd.DatetimeIndex(
        timestamps,
        name="timestamp",
    )

    if index.has_duplicates:
        raise RuntimeError(
            "RCSD-1YD contains duplicate timestamps."
        )

    if not index.is_monotonic_increasing:
        raise RuntimeError(
            "RCSD-1YD source timestamps are not monotonically increasing. "
            "Protocol forbids subjective reordering."
        )

    if index[0] != pd.Timestamp(expected_start):
        raise RuntimeError(
            "RCSD-1YD first timestamp mismatch: "
            f"expected {expected_start}, observed {index[0]}."
        )

    if index[-1] != pd.Timestamp(expected_end):
        raise RuntimeError(
            "RCSD-1YD final timestamp mismatch: "
            f"expected {expected_end}, observed {index[-1]}."
        )

    source = raw.drop(columns=[timestamp_column]).copy()
    source.index = index

    train_raw = source.loc[
        _split_mask(
            index,
            start=str(train_cfg["start"]),
            end_exclusive=str(train_cfg["end_exclusive"]),
        )
    ].copy()

    calibration_raw = source.loc[
        _split_mask(
            index,
            start=str(calibration_cfg["start"]),
            end_exclusive=str(calibration_cfg["end_exclusive"]),
        )
    ].copy()

    evaluation_raw = source.loc[
        _split_mask(
            index,
            start=str(evaluation_cfg["start"]),
            end_inclusive=str(evaluation_cfg["end_inclusive"]),
        )
    ].copy()

    if (
        len(train_raw)
        + len(calibration_raw)
        + len(evaluation_raw)
        != len(source)
    ):
        raise RuntimeError(
            "Frozen calendar split does not partition the source exactly."
        )

    train, train_dropped = _coerce_sensors(
        train_raw,
        sensors=sensors,
    )
    calibration, calibration_dropped = _coerce_sensors(
        calibration_raw,
        sensors=sensors,
    )
    evaluation, _ = _coerce_sensors(
        evaluation_raw,
        sensors=sensors,
    )

    for name, frame in (
        ("TRAIN", train),
        ("CALIBRATION", calibration),
        ("EVALUATION", evaluation),
    ):
        if frame.empty:
            raise RuntimeError(f"{name} is empty after frozen cleaning.")

        if frame.columns.tolist() != sensors:
            raise RuntimeError(f"{name} schema changed after cleaning.")

        if not frame.index.is_monotonic_increasing:
            raise RuntimeError(f"{name} timestamps lost chronological order.")

        if frame.index.has_duplicates:
            raise RuntimeError(f"{name} contains duplicate timestamps.")

    return RCSDSplitResult(
        train=train,
        calibration=calibration,
        evaluation=evaluation,
        timestamp_column=timestamp_column,
        source_rows=len(source),
        train_dropped_rows=train_dropped,
        calibration_dropped_rows=calibration_dropped,
    )


def count_gaps(
    frame: pd.DataFrame,
    *,
    greater_than_minutes: int,
) -> int:
    deltas = frame.index.to_series().diff()
    return int(
        (
            deltas
            > pd.Timedelta(minutes=greater_than_minutes)
        ).sum()
    )


def frame_bounds(frame: pd.DataFrame) -> dict[str, str]:
    if frame.empty:
        raise ValueError("Cannot summarize an empty frame.")

    return {
        "start": str(frame.index[0]),
        "end": str(frame.index[-1]),
    }
