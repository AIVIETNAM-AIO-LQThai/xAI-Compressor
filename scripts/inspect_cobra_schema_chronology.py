from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/cobra_schema_chronology.yaml"
PROTOCOL_COMMIT = "d4a4e794a35a5a7716c50ed2a540931985a45b2a"

DELIMITERS = (",", ";", "\t", "|")
TIMESTAMP_EXACT_NAMES = {
    "timestamp",
    "time_stamp",
    "datetime",
    "date_time",
    "date time",
}
TIMESTAMP_HINTS = ("timestamp", "datetime", "date", "time")
TRUE_TEXT = {"true", "false", "yes", "no", "on", "off"}

TEXT_ENCODING_CANDIDATES = (
    "utf-8-sig",
    "cp1252",
    "latin-1",
)

DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S.%f",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M:%S.%f",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S.%f",
)


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return value


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return value


def _sha256(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _delimiter_from_header(header_line: str) -> str:
    counts = {delimiter: header_line.count(delimiter) for delimiter in DELIMITERS}
    maximum = max(counts.values())
    winners = [
        delimiter
        for delimiter, count in counts.items()
        if count == maximum and count > 0
    ]
    if len(winners) != 1:
        raise RuntimeError(
            "Could not determine a unique CSV delimiter from the header only."
        )
    return winners[0]


def _normalized_name(value: str) -> str:
    return " ".join(value.strip().lower().replace("-", "_").split())


def _timestamp_column(columns: list[str]) -> int:
    normalized = [_normalized_name(column) for column in columns]

    exact = [
        index
        for index, name in enumerate(normalized)
        if name in TIMESTAMP_EXACT_NAMES
    ]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise RuntimeError(
            f"Multiple exact timestamp columns found: "
            f"{[columns[index] for index in exact]}"
        )

    hinted = [
        index
        for index, name in enumerate(normalized)
        if any(hint in name for hint in TIMESTAMP_HINTS)
    ]
    if len(hinted) == 1:
        return hinted[0]

    raise RuntimeError(
        "Could not identify one timestamp column from header names only. "
        f"Candidates: {[columns[index] for index in hinted]}"
    )


def _parse_datetime(value: str) -> datetime:
    stripped = value.strip()
    if not stripped:
        raise ValueError("Blank timestamp.")

    iso_candidate = stripped.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(iso_candidate)
    except ValueError:
        pass

    for format_string in DATETIME_FORMATS:
        try:
            return datetime.strptime(stripped, format_string).replace(tzinfo=timezone.utc)
        except ValueError:
            continue

    raise ValueError("Unsupported timestamp format.")


def _decode_header_bytes(
    header_bytes: bytes,
) -> tuple[str, str]:
    if not header_bytes:
        raise ValueError("CSV header is empty.")

    for encoding in TEXT_ENCODING_CANDIDATES:
        try:
            return encoding, header_bytes.decode(
                encoding,
                errors="strict",
            )
        except UnicodeDecodeError:
            continue

    raise RuntimeError(
        "Could not decode CSV header with frozen encoding candidates."
    )


def _coarse_class_update(
    state: dict[str, bool | int],
    value: str,
) -> None:
    stripped = value.strip()
    if not stripped:
        state["blank"] = int(state["blank"]) + 1
        return

    state["nonblank"] = int(state["nonblank"]) + 1

    try:
        float(stripped)
    except ValueError:
        state["all_numeric"] = False

    if stripped.lower() not in TRUE_TEXT:
        state["all_boolean"] = False


def _coarse_class(state: dict[str, bool | int]) -> str:
    if int(state["nonblank"]) == 0:
        return "all_blank"
    if bool(state["all_numeric"]):
        return "numeric"
    if bool(state["all_boolean"]):
        return "boolean"
    return "text_or_mixed"


def _primary_csv_member(
    archive_name: str,
    members: list[str],
) -> str:
    csv_members = [
        member
        for member in members
        if member.lower().endswith(".csv")
        and not member.endswith("/")
    ]
    if not csv_members:
        raise RuntimeError(f"No CSV member in {archive_name}.")

    archive_stem = Path(archive_name).stem.lower()
    exact_stem = [
        member
        for member in csv_members
        if Path(member).stem.lower() == archive_stem
    ]
    if len(exact_stem) == 1:
        return exact_stem[0]

    if len(csv_members) == 1:
        return csv_members[0]

    raise RuntimeError(
        f"Ambiguous primary CSV in {archive_name}: {csv_members}"
    )


def _partition_for_day(config: dict[str, Any], day: str) -> str:
    split = config["split"]
    if day in split["train_days"]:
        return "TRAIN"
    if day in split["calibration_days"]:
        return "CALIBRATION"
    if day in split["evaluation_days"]:
        return "EVALUATION"
    raise RuntimeError(f"Day is outside frozen split: {day}")


def _day_from_archive_name(name: str) -> str:
    stem = Path(name).stem
    raw = stem.removeprefix("CoBra")
    if len(raw) != 8 or not raw.isdigit():
        raise RuntimeError(f"Unexpected CoBra archive name: {name}")
    return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"


def _inspect_archive(
    path: Path,
    *,
    partition: str,
) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        primary = _primary_csv_member(path.name, members)

        with archive.open(primary, "r") as raw:
            header_bytes = raw.readline()

        if not header_bytes:
            raise RuntimeError(f"Empty CSV in {path.name}.")

        source_encoding, header_line = _decode_header_bytes(
            header_bytes
        )

        with archive.open(primary, "r") as raw:
            text = io.TextIOWrapper(
                raw,
                encoding=source_encoding,
                newline="",
            )

            decoded_header = text.readline()
            if decoded_header != header_line:
                raise RuntimeError(
                    f"Header decoding was not deterministic in {path.name}."
                )

            delimiter = _delimiter_from_header(header_line)
            header_reader = csv.reader([header_line], delimiter=delimiter)
            columns = next(header_reader)

            if len(columns) < 2:
                raise RuntimeError(
                    f"Too few columns in {path.name}: {len(columns)}"
                )

            timestamp_index = _timestamp_column(columns)

            states = [
                {
                    "nonblank": 0,
                    "blank": 0,
                    "all_numeric": True,
                    "all_boolean": True,
                }
                for _ in columns
            ]

            row_count = 0
            structural_blank_rows = 0
            field_count_mismatch_rows = 0
            timestamp_parse_errors = 0
            duplicate_timestamp_count = 0
            non_monotonic_timestamp_count = 0
            step_counts: Counter[str] = Counter()

            first_timestamp: datetime | None = None
            last_timestamp: datetime | None = None
            previous_timestamp: datetime | None = None

            reader = csv.reader(text, delimiter=delimiter)

            for row in reader:
                row_count += 1

                if len(row) != len(columns):
                    field_count_mismatch_rows += 1
                    continue

                if any(
                    not value.strip()
                    for index, value in enumerate(row)
                    if index != timestamp_index
                ):
                    structural_blank_rows += 1

                current_timestamp: datetime | None
                try:
                    current_timestamp = _parse_datetime(
                        row[timestamp_index]
                    )
                except ValueError:
                    timestamp_parse_errors += 1
                    current_timestamp = None

                if current_timestamp is not None:
                    if first_timestamp is None:
                        first_timestamp = current_timestamp

                    if previous_timestamp is not None:
                        delta_seconds = (
                            current_timestamp - previous_timestamp
                        ).total_seconds()

                        if delta_seconds == 0:
                            duplicate_timestamp_count += 1
                        elif delta_seconds < 0:
                            non_monotonic_timestamp_count += 1

                        step_counts[
                            format(delta_seconds, ".9g")
                        ] += 1

                    previous_timestamp = current_timestamp
                    last_timestamp = current_timestamp

                for index, value in enumerate(row):
                    if index == timestamp_index:
                        continue
                    _coarse_class_update(states[index], value)

            schema_classes = {}
            blank_field_counts = {}

            for index, column in enumerate(columns):
                if index == timestamp_index:
                    schema_classes[column] = "timestamp"
                    blank_field_counts[column] = None
                else:
                    schema_classes[column] = _coarse_class(states[index])
                    blank_field_counts[column] = int(
                        states[index]["blank"]
                    )

            return {
                "archive": path.name,
                "partition": partition,
                "zip_members": members,
                "primary_csv_member": primary,
                "source_encoding": source_encoding,
                "delimiter": delimiter,
                "column_count": len(columns),
                "columns": columns,
                "timestamp_column": columns[timestamp_index],
                "rows": row_count,
                "first_timestamp": (
                    first_timestamp.isoformat()
                    if first_timestamp is not None
                    else None
                ),
                "last_timestamp": (
                    last_timestamp.isoformat()
                    if last_timestamp is not None
                    else None
                ),
                "timestamp_parse_errors": timestamp_parse_errors,
                "duplicate_timestamp_count": duplicate_timestamp_count,
                "non_monotonic_timestamp_count": (
                    non_monotonic_timestamp_count
                ),
                "timestamp_step_seconds_counts": dict(
                    sorted(
                        step_counts.items(),
                        key=lambda item: float(item[0]),
                    )
                ),
                "field_count_mismatch_rows": field_count_mismatch_rows,
                "structural_blank_rows": structural_blank_rows,
                "coarse_schema_classes": schema_classes,
                "blank_field_counts_structural_only": blank_field_counts,
            }


def _inspect_overview(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header_line = handle.readline()
        if not header_line:
            raise RuntimeError("overview_experiments.csv is empty.")
        delimiter = _delimiter_from_header(header_line)
        columns = next(csv.reader([header_line], delimiter=delimiter))
        rows = sum(1 for _ in csv.reader(handle, delimiter=delimiter))

    return {
        "file": path.name,
        "delimiter": delimiter,
        "columns": columns,
        "row_count": rows,
        "content_values_serialized": False,
    }


def main() -> None:
    config = _yaml(CONFIG_PATH)

    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            PROTOCOL_COMMIT,
            "HEAD",
        ],
        cwd=ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(
            "CoBra schema/chronology protocol is not an ancestor of HEAD."
        )

    if config["study"]["model_outcome_metrics_allowed"] is not False:
        raise RuntimeError("Model outcome metrics must remain forbidden.")
    if config["study"]["evaluation_sensor_exploration_allowed"] is not False:
        raise RuntimeError("EVALUATION sensor exploration must remain forbidden.")
    if not all(bool(value) for value in config["forbidden"].values()):
        raise RuntimeError("A forbidden-operation guard was disabled.")

    output = ROOT / config["output"]["manifest"]
    if output.exists():
        raise RuntimeError(
            f"Manifest already exists; refusing overwrite: {output}"
        )

    source_manifest = _json(
        ROOT / config["source"]["source_lock_manifest"]
    )
    source_dir = ROOT / config["source"]["directory"]

    if source_manifest["source_archives_unzipped"] is not False:
        raise RuntimeError("Source-lock archive provenance changed.")
    if source_manifest["sensor_values_opened"] is not False:
        raise RuntimeError("Source-lock sensor provenance changed.")
    if source_manifest["outcome_metrics_computed"] is not False:
        raise RuntimeError("Source-lock outcome provenance changed.")

    primary_files = source_manifest["primary_local_files"]
    expected_archives = sorted(
        name
        for name in primary_files
        if name.startswith("CoBra20") and name.endswith(".zip")
    )

    if len(expected_archives) != 23:
        raise RuntimeError(
            f"Expected 23 CoBra test-day archives, got {len(expected_archives)}."
        )

    hash_verification: dict[str, Any] = {}
    for name, record in primary_files.items():
        path = ROOT / record["path"]
        observed_sha256 = _sha256(path)
        if observed_sha256 != record["sha256"]:
            raise RuntimeError(
                f"Source SHA256 changed for {name}: "
                f"{observed_sha256} != {record['sha256']}"
            )
        hash_verification[name] = {
            "sha256": observed_sha256,
            "matches_source_lock": True,
        }

    archives: list[dict[str, Any]] = []
    for index, name in enumerate(expected_archives, start=1):
        day = _day_from_archive_name(name)
        partition = _partition_for_day(config, day)

        print(
            f"[{index}/23] inspecting schema/chronology for "
            f"{name} ({partition})",
            flush=True,
        )
        archives.append(
            _inspect_archive(
                source_dir / name,
                partition=partition,
            )
        )

    overview = _inspect_overview(
        source_dir / config["source"]["overview_file"]
    )

    all_parse_clean = all(
        item["timestamp_parse_errors"] == 0
        for item in archives
    )
    all_monotonic = all(
        item["non_monotonic_timestamp_count"] == 0
        for item in archives
    )
    all_field_counts_consistent = all(
        item["field_count_mismatch_rows"] == 0
        for item in archives
    )
    primary_csv_unambiguous = len(archives) == 23

    split = config["split"]
    payload = {
        "schema_version": "aeroxai.cobra_schema_chronology.v1",
        "study": config["study"]["name"],
        "evidence_role": config["study"]["evidence_role"],
        "protocol_commit": PROTOCOL_COMMIT,
        "source_lock_commit": config["study"]["parent_source_lock_commit"],
        "source_hash_verification": hash_verification,
        "frozen_split": {
            "unit": split["unit"],
            "assignment_basis": split["assignment_basis"],
            "train_days": split["train_days"],
            "calibration_days": split["calibration_days"],
            "evaluation_days": split["evaluation_days"],
            "may_change_after_schema_inspection": False,
            "may_change_after_model_results": False,
        },
        "archives": archives,
        "overview_experiments": overview,
        "structural_readiness": {
            "primary_csv_unambiguous": primary_csv_unambiguous,
            "all_timestamps_parse_cleanly": all_parse_clean,
            "all_archives_monotonic_non_decreasing": all_monotonic,
            "all_row_field_counts_consistent": all_field_counts_consistent,
            "ready_for_adaptation_protocol": (
                primary_csv_unambiguous
                and all_parse_clean
                and all_monotonic
                and all_field_counts_consistent
            ),
        },
        "inspection_boundary": {
            "sensor_descriptive_statistics_computed": False,
            "sensor_quantiles_computed": False,
            "sensor_correlations_computed": False,
            "sensor_plots_created": False,
            "pca_scores_computed": False,
            "ewma_scores_computed": False,
            "support_scores_computed": False,
            "tcn_scores_computed": False,
            "router_v1_scores_computed": False,
            "router_v2_scores_computed": False,
            "target_prevalence_computed": False,
            "coverage_metrics_computed": False,
            "anomaly_rate_computed": False,
            "energy_metrics_computed": False,
            "train_cal_eval_sensor_distributions_compared": False,
            "evaluation_used_for_feature_selection": False,
            "evaluation_used_for_hyperparameter_selection": False,
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    partition_rows = Counter()
    for item in archives:
        partition_rows[item["partition"]] += int(item["rows"])

    print(
        json.dumps(
            {
                "status": (
                    "COBRA_SCHEMA_CHRONOLOGY_PASS"
                    if payload["structural_readiness"][
                        "ready_for_adaptation_protocol"
                    ]
                    else "COBRA_SCHEMA_CHRONOLOGY_REVIEW_REQUIRED"
                ),
                "archives_inspected": len(archives),
                "train_days": len(split["train_days"]),
                "calibration_days": len(split["calibration_days"]),
                "evaluation_days": len(split["evaluation_days"]),
                "row_counts_by_partition": dict(partition_rows),
                "all_timestamps_parse_cleanly": all_parse_clean,
                "all_archives_monotonic_non_decreasing": all_monotonic,
                "all_row_field_counts_consistent": (
                    all_field_counts_consistent
                ),
                "ready_for_adaptation_protocol": payload[
                    "structural_readiness"
                ]["ready_for_adaptation_protocol"],
                "model_outcome_metrics_computed": False,
                "manifest": str(output.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
