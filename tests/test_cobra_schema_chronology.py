from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import yaml

from scripts.inspect_cobra_schema_chronology import (
    _coarse_class,
    _coarse_class_update,
    _decode_header_bytes,
    _delimiter_from_header,
    _inspect_archive,
    _parse_datetime,
    _primary_csv_member,
    _timestamp_column,
)


def _config() -> dict:
    return yaml.safe_load(
        Path("configs/cobra_schema_chronology.yaml")
        .read_text(encoding="utf-8")
    )


def test_split_is_frozen_14_4_5_and_chronological() -> None:
    config = _config()
    split = config["split"]

    train = list(split["train_days"])
    calibration = list(split["calibration_days"])
    evaluation = list(split["evaluation_days"])
    all_days = train + calibration + evaluation

    assert len(train) == 14
    assert len(calibration) == 4
    assert len(evaluation) == 5
    assert len(set(all_days)) == 23
    assert all_days == sorted(all_days)
    assert split["may_change_after_schema_inspection"] is False
    assert split["may_change_after_model_results"] is False


def test_forbidden_operations_are_all_enabled() -> None:
    config = _config()

    assert config["study"]["model_outcome_metrics_allowed"] is False
    assert (
        config["study"]["evaluation_sensor_exploration_allowed"]
        is False
    )
    assert all(bool(value) for value in config["forbidden"].values())


def test_header_encoding_falls_back_to_cp1252() -> None:
    header = b"Timestamp;Temperature [\xb0C];Pressure\n"

    encoding, decoded = _decode_header_bytes(header)

    assert encoding == "cp1252"
    assert chr(0x00B0) + "C" in decoded

def test_delimiter_from_header() -> None:
    assert _delimiter_from_header("Timestamp;A;B\n") == ";"
    assert _delimiter_from_header("Timestamp,A,B\n") == ","


def test_timestamp_column_prefers_exact_name() -> None:
    columns = ["Sensor date flag", "Timestamp", "Pressure"]

    assert _timestamp_column(columns) == 1


def test_parse_datetime_common_formats() -> None:
    assert _parse_datetime("2025-02-17 12:34:56").year == 2025
    assert _parse_datetime("17.02.2025 12:34:56").month == 2


def test_coarse_class_never_returns_values() -> None:
    numeric = {
        "nonblank": 0,
        "blank": 0,
        "all_numeric": True,
        "all_boolean": True,
    }
    for value in ("1.0", "2.5", ""):
        _coarse_class_update(numeric, value)

    assert _coarse_class(numeric) == "numeric"
    assert numeric["blank"] == 1


def test_primary_csv_member_exact_stem() -> None:
    members = [
        "notes/readme.txt",
        "CoBra20250217.csv",
        "secondary.csv",
    ]

    assert (
        _primary_csv_member("CoBra20250217.zip", members)
        == "CoBra20250217.csv"
    )


def test_synthetic_archive_inspection_is_schema_only(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "CoBra20250217.zip"

    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=";")
    writer.writerow(["Timestamp", "Pressure", "State"])
    writer.writerow(["2025-02-17 00:00:00", "1.0", "on"])
    writer.writerow(["2025-02-17 00:00:01", "2.0", "off"])
    writer.writerow(["2025-02-17 00:00:02", "", "on"])

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "CoBra20250217.csv",
            buffer.getvalue().encode("utf-8"),
        )

    result = _inspect_archive(
        archive_path,
        partition="EVALUATION",
    )

    assert result["rows"] == 3
    assert result["source_encoding"] == "utf-8-sig"
    assert result["timestamp_parse_errors"] == 0
    assert result["duplicate_timestamp_count"] == 0
    assert result["non_monotonic_timestamp_count"] == 0
    assert result["timestamp_step_seconds_counts"] == {"1": 2}
    assert result["coarse_schema_classes"]["Pressure"] == "numeric"
    assert result["coarse_schema_classes"]["State"] == "boolean"
    assert result["structural_blank_rows"] == 1
    assert "sensor_mean" not in result
    assert "sensor_min" not in result
    assert "sensor_max" not in result
