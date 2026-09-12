from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from ml.data.schema import (
    ANALOG_COLUMNS,
    DIGITAL_COLUMNS,
    canonicalize_column_name,
    normalize_raw_frame,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "metropt.yaml"
REPORT_PATH = ROOT / "docs" / "data_audit.json"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["dataset"]


def as_float(value) -> float:
    return float(value)


def main() -> None:
    config = load_config()

    csv_path = ROOT / config["raw_csv"]

    print(f"Reading {csv_path} ...")

    raw = pd.read_csv(csv_path)

    canonical_raw = raw.rename(
        columns={
            column: canonicalize_column_name(column)
            for column in raw.columns
        }
    )

    raw_timestamps = pd.to_datetime(
        canonical_raw["timestamp"],
        errors="raise",
    )

    duplicate_timestamps = int(
        raw_timestamps.duplicated().sum()
    )

    frame = normalize_raw_frame(raw)

    timestamp_deltas = (
        frame["timestamp"]
        .diff()
        .dt.total_seconds()
        .dropna()
    )

    cadence_quantiles = (
        timestamp_deltas
        .quantile([0.01, 0.50, 0.95, 0.99])
        .to_dict()
    )

    cadence_top = {
        str(key): int(value)
        for key, value in (
            timestamp_deltas
            .value_counts()
            .head(10)
            .items()
        )
    }

    gaps = {
        "gt_15_seconds": int(
            (timestamp_deltas > 15).sum()
        ),
        "gt_30_seconds": int(
            (timestamp_deltas > 30).sum()
        ),
        "gt_60_seconds": int(
            (timestamp_deltas > 60).sum()
        ),
        "gt_300_seconds": int(
            (timestamp_deltas > 300).sum()
        ),
    }

    analog_summary = {}

    for column in ANALOG_COLUMNS:
        series = frame[column]

        analog_summary[column] = {
            "min": as_float(series.min()),
            "p01": as_float(series.quantile(0.01)),
            "median": as_float(series.median()),
            "mean": as_float(series.mean()),
            "p99": as_float(series.quantile(0.99)),
            "max": as_float(series.max()),
            "std": as_float(series.std()),
        }

    digital_values = {}

    for column in DIGITAL_COLUMNS:
        values = frame[column].dropna().unique()

        digital_values[column] = sorted(
            float(value) for value in values
        )[:50]

    incidents = []

    for incident in config["reported_incidents"]:
        start = pd.Timestamp(incident["start"])
        end = pd.Timestamp(incident["end"])

        mask = frame["timestamp"].between(
            start,
            end,
            inclusive="both",
        )

        incidents.append(
            {
                **incident,
                "rows_in_interval": int(mask.sum()),
            }
        )

    missing_values = {
        column: int(count)
        for column, count in (
            frame.isna().sum().items()
        )
    }

    report = {
        "dataset": config["name"],
        "rows_raw": len(raw),
        "rows_after_normalization": len(frame),
        "expected_rows": int(config["expected_rows"]),
        "raw_columns": list(raw.columns),
        "canonical_columns": list(frame.columns),
        "duplicate_timestamps_raw": duplicate_timestamps,
        "missing_values": missing_values,
        "time_start": str(frame["timestamp"].min()),
        "time_end": str(frame["timestamp"].max()),
        "cadence_seconds": {
            "min": as_float(timestamp_deltas.min()),
            "p01": as_float(cadence_quantiles[0.01]),
            "median": as_float(cadence_quantiles[0.50]),
            "p95": as_float(cadence_quantiles[0.95]),
            "p99": as_float(cadence_quantiles[0.99]),
            "max": as_float(timestamp_deltas.max()),
            "most_common": cadence_top,
        },
        "gaps": gaps,
        "analog_summary": analog_summary,
        "digital_unique_values": digital_values,
        "reported_incidents": incidents,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=== MetroPT-3 Audit ===")
    print(f"Rows:       {len(frame):,}")
    print(
        f"Time range: {frame['timestamp'].min()} "
        f"→ {frame['timestamp'].max()}"
    )
    print(
        "Median cadence: "
        f"{timestamp_deltas.median():.3f} s"
    )
    print(
        "Duplicate timestamps: "
        f"{duplicate_timestamps:,}"
    )

    print("\nIncident coverage:")

    for incident in incidents:
        print(
            f"  Incident {incident['id']}: "
            f"{incident['rows_in_interval']:,} rows"
        )

    print(
        f"\nAudit written to: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()