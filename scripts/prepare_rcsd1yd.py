from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ml.data.rcsd1yd import (
    count_gaps,
    frame_bounds,
    load_and_split_rcsd1yd,
    md5_file,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rcsd1yd_external_evaluation.yaml"
SOURCE_PATH = ROOT / "data/raw/rcsd1yd/Centrifugal compressor _2022.xlsx"
OUTPUT_DIR = ROOT / "data/processed/rcsd1yd"
MANIFEST_PATH = ROOT / "docs/research/rcsd1yd_ingestion_manifest.json"

PREREGISTRATION_COMMIT = "7dc7bd2d9a44758075dab286cc3d674949a876bd"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return payload


def main() -> None:
    config = _load_yaml(CONFIG_PATH)

    if config["study"]["branch"] != "research/rcsd1yd-external-evaluation":
        raise RuntimeError("Unexpected study branch in RCSD-1YD config.")

    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"RCSD-1YD source file not found at {SOURCE_PATH}. "
            "Download the frozen Zenodo file before running ingestion."
        )

    observed_md5 = md5_file(SOURCE_PATH)
    expected_md5 = str(config["dataset"]["expected_source_md5"])

    if observed_md5 != expected_md5:
        raise RuntimeError(
            "RCSD-1YD source MD5 mismatch: "
            f"expected {expected_md5}, observed {observed_md5}."
        )

    sensors = [str(value) for value in config["sensor_columns"]]

    result = load_and_split_rcsd1yd(
        SOURCE_PATH,
        sensors=sensors,
        expected_total_columns=int(config["dataset"]["expected_total_columns"]),
        expected_start=str(config["dataset"]["expected_start"]),
        expected_end=str(config["dataset"]["expected_end"]),
        train_cfg=config["chronological_split"]["train"],
        calibration_cfg=config["chronological_split"]["calibration"],
        evaluation_cfg=config["chronological_split"]["evaluation"],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.parquet"
    calibration_path = OUTPUT_DIR / "calibration.parquet"
    evaluation_path = OUTPUT_DIR / "evaluation.parquet"

    result.train.to_parquet(train_path)
    result.calibration.to_parquet(calibration_path)
    result.evaluation.to_parquet(evaluation_path)

    manifest = {
        "schema_version": "aeroxai.rcsd1yd_ingestion.v1",
        "study": config["study"]["name"],
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "source": {
            "file_name": config["dataset"]["source_file_name"],
            "md5": observed_md5,
            "dataset_doi": config["dataset"]["dataset_doi"],
            "timestamp_column": result.timestamp_column,
            "source_rows": result.source_rows,
            "sensor_count": len(sensors),
            "sensor_columns": sensors,
            "schema_validation_passed": True,
        },
        "data_use": {
            "evaluation_rows_written": True,
            "evaluation_sensor_summary_computed": False,
            "evaluation_anomaly_metrics_computed": False,
            "evaluation_router_metrics_computed": False,
            "evaluation_tcn_metrics_computed": False,
        },
        "train": {
            "path": str(train_path.relative_to(ROOT)),
            "rows": len(result.train),
            "bounds": frame_bounds(result.train),
            "dropped_nonfinite_rows": result.train_dropped_rows,
            "gaps_greater_than_30_minutes": count_gaps(
                result.train,
                greater_than_minutes=30,
            ),
            "sha256": sha256_file(train_path),
        },
        "calibration": {
            "path": str(calibration_path.relative_to(ROOT)),
            "rows": len(result.calibration),
            "bounds": frame_bounds(result.calibration),
            "dropped_nonfinite_rows": result.calibration_dropped_rows,
            "gaps_greater_than_30_minutes": count_gaps(
                result.calibration,
                greater_than_minutes=30,
            ),
            "sha256": sha256_file(calibration_path),
        },
        "evaluation": {
            "path": str(evaluation_path.relative_to(ROOT)),
            "rows": len(result.evaluation),
            "frozen_calendar_bounds": {
                "start": config["chronological_split"]["evaluation"]["start"],
                "end": config["chronological_split"]["evaluation"][
                    "end_inclusive"
                ],
            },
            "sha256": sha256_file(evaluation_path),
        },
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "source_md5": observed_md5,
                "schema_validation_passed": True,
                "train_rows": len(result.train),
                "calibration_rows": len(result.calibration),
                "evaluation_rows": len(result.evaluation),
                "train_dropped_rows": result.train_dropped_rows,
                "calibration_dropped_rows": result.calibration_dropped_rows,
                "train_gaps_gt_30min": manifest["train"][
                    "gaps_greater_than_30_minutes"
                ],
                "calibration_gaps_gt_30min": manifest["calibration"][
                    "gaps_greater_than_30_minutes"
                ],
                "evaluation_metrics_computed": False,
                "output": str(MANIFEST_PATH.relative_to(ROOT)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
