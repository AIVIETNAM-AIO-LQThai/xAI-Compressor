from __future__ import annotations

import json
import pickle
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from ml.data.cobra import (
    aggregate_five_minute,
    semantic_representation_sha256,
)
from ml.detection.common import transform_frame
from ml.energy.evidence_aware_router_v2 import (
    RouterV2Candidate,
    evidence_coverage,
    route_candidate,
)
from ml.temporal.detector import (
    TemporalDetector,
    score_temporal_detector,
)
from ml.temporal.tcn import (
    TCNForecaster,
    TemporalForecastConfig,
)
from scripts.build_cobra_train_representation import (
    _archive_metadata_by_day,
    _read_train_archive,
    _sha256,
)
from scripts.fit_cobra_train_models import (
    expected_sequence_count,
)

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs/cobra_router_v2_adaptation.yaml"
SCHEMA_CONFIG_PATH = ROOT / "configs/cobra_schema_chronology.yaml"
SCHEMA_MANIFEST_PATH = (
    ROOT / "docs/research/cobra_schema_chronology_manifest.json"
)
SOURCE_LOCK_PATH = (
    ROOT / "docs/research/cobra_source_lock_manifest.json"
)
FEATURE_PATH = (
    ROOT / "docs/research/cobra_router_v2_raw_feature_contract.json"
)
TRAIN_FIT_PATH = (
    ROOT / "docs/research/cobra_router_v2_train_fit.json"
)

SKLEARN_ARTIFACT = (
    ROOT / "artifacts/research/cobra_router_v2_train_models.pkl"
)
TCN_ARTIFACT = (
    ROOT / "artifacts/research/cobra_router_v2_tcn.pt"
)

CAL_DATA_PATH = (
    ROOT / "data/processed/cobra_router_v2/calibration_5min.csv"
)
CAL_SCORES_PATH = (
    ROOT / "data/processed/cobra_router_v2/calibration_scores.csv"
)
RESULT_PATH = (
    ROOT / "docs/research/cobra_router_v2_calibration.json"
)

REQUIRED_TRAIN_FIT_COMMIT = (
    "52da89292b9838c4bbc4ba82bb24c722f9f84c30"
)

EXPECTED_FEATURE_SHA = (
    "ba54ffa70cda5690ceaef4d0a58d6d14129dd36a0e03abbc95c20344e3e4a9aa"
)

IMPLEMENTATION_PATHS = [
    "scripts/calibrate_cobra_router_v2.py",
    "tests/test_cobra_router_v2_calibration.py",
]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON mapping in {path}.")
    return value


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping in {path}.")
    return value


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def causal_ewma(
    values: np.ndarray,
    segments: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    segments = np.asarray(segments)

    if values.ndim != 1 or segments.ndim != 1:
        raise ValueError("EWMA arrays must be one-dimensional.")

    if len(values) != len(segments):
        raise ValueError("EWMA value/segment lengths differ.")

    if not 0.0 < alpha <= 1.0:
        raise ValueError("EWMA alpha must lie in (0, 1].")

    if not np.isfinite(values).all():
        raise ValueError("EWMA input contains non-finite values.")

    result = np.empty(len(values), dtype=float)

    previous_segment: object | None = None
    previous_ewma = 0.0

    for index, (value, segment) in enumerate(
        zip(values, segments, strict=True)
    ):
        if index == 0 or segment != previous_segment:
            current = float(value)
        else:
            current = (
                alpha * float(value)
                + (1.0 - alpha) * previous_ewma
            )

        result[index] = current
        previous_ewma = current
        previous_segment = segment

    return result


def higher_quantile(
    values: np.ndarray,
    quantile: float,
) -> float:
    values = np.asarray(values, dtype=float)

    if values.ndim != 1 or values.size == 0:
        raise ValueError("Quantile input must be non-empty 1-D.")

    if not np.isfinite(values).all():
        raise ValueError("Quantile input contains non-finite values.")

    return float(
        np.quantile(
            values,
            quantile,
            method="higher",
        )
    )


def right_inclusive_percentile(
    values: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    reference = np.sort(
        np.asarray(reference, dtype=float)
    )

    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("Percentile reference must be non-empty.")

    if not np.isfinite(reference).all():
        raise ValueError("Percentile reference is non-finite.")

    return (
        np.searchsorted(
            reference,
            values,
            side="right",
        )
        / reference.size
    ).astype(float)


def recent_hit_fraction(
    values: np.ndarray,
    segments: np.ndarray,
    *,
    threshold: float,
    history_bins: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    segments = np.asarray(segments)

    if len(values) != len(segments):
        raise ValueError("Rolling value/segment lengths differ.")

    result = np.full(len(values), np.nan, dtype=float)

    start = 0
    while start < len(values):
        segment = segments[start]
        stop = start + 1

        while stop < len(values) and segments[stop] == segment:
            stop += 1

        hits = (
            values[start:stop]
            >= threshold
        ).astype(float)

        for local_index in range(
            history_bins - 1,
            len(hits),
        ):
            absolute = start + local_index
            window = hits[
                local_index - history_bins + 1:
                local_index + 1
            ]
            result[absolute] = float(window.mean())

        start = stop

    return result


def _verify_boundary() -> None:
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            REQUIRED_TRAIN_FIT_COMMIT,
            "HEAD",
        ],
        cwd=ROOT,
        check=False,
    )

    if ancestor.returncode != 0:
        raise RuntimeError(
            "Frozen TRAIN fit result is not an ancestor."
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
            "CAL implementation must be clean and committed."
        )

    for path in (
        CAL_DATA_PATH,
        CAL_SCORES_PATH,
        RESULT_PATH,
    ):
        if path.exists():
            raise RuntimeError(
                f"Refusing to overwrite existing CAL output: {path}"
            )


def _load_tcn() -> TCNForecaster:
    payload = torch.load(
        TCN_ARTIFACT,
        map_location="cpu",
        weights_only=False,
    )

    config = payload["model_config"]

    model_config = TemporalForecastConfig(
        input_dim=int(config["input_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        kernel_size=int(config["kernel_size"]),
        dilations=tuple(
            int(value)
            for value in config["dilations"]
        ),
        dropout=float(config["dropout"]),
    )

    model = TCNForecaster(model_config)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    return model


def main() -> None:
    _verify_boundary()

    config = _yaml(CONFIG_PATH)
    schema_config = _yaml(SCHEMA_CONFIG_PATH)
    schema_manifest = _json(SCHEMA_MANIFEST_PATH)
    source_lock = _json(SOURCE_LOCK_PATH)
    feature_contract = _json(FEATURE_PATH)
    train_fit = _json(TRAIN_FIT_PATH)

    features = [
        str(value)
        for value in feature_contract["final_raw_features"]
    ]

    if len(features) != 219:
        raise RuntimeError("Expected 219 frozen features.")

    if (
        feature_contract["final_raw_feature_sha256"]
        != EXPECTED_FEATURE_SHA
    ):
        raise RuntimeError("Frozen feature SHA changed.")

    if _sha256(SKLEARN_ARTIFACT) != (
        train_fit["artifacts"]["sklearn_sha256"]
    ):
        raise RuntimeError("Frozen sklearn artifact SHA mismatch.")

    if _sha256(TCN_ARTIFACT) != (
        train_fit["artifacts"]["tcn_sha256"]
    ):
        raise RuntimeError("Frozen TCN artifact SHA mismatch.")

    with SKLEARN_ARTIFACT.open("rb") as handle:
        trained = pickle.load(handle)

    if trained["features"] != features:
        raise RuntimeError("Frozen model feature order changed.")

    pca_scaler = trained["pca_scaler"]
    pca = trained["pca"]
    tcn_scaler = trained["tcn_scaler"]

    tcn_model = _load_tcn()

    split = schema_config["split"]

    train_days = {
        str(value)
        for value in split["train_days"]
    }
    calibration_days = {
        str(value)
        for value in split["calibration_days"]
    }
    evaluation_days = {
        str(value)
        for value in split["evaluation_days"]
    }

    ordered_calibration_days = [
        str(value)
        for value in split["calibration_days"]
    ]

    if len(ordered_calibration_days) != 4:
        raise RuntimeError("Expected four CALIBRATION days.")

    archive_by_day = _archive_metadata_by_day(
        schema_manifest
    )

    primary_files = source_lock["primary_local_files"]

    day_frames: list[pd.DataFrame] = []
    per_day: list[dict[str, Any]] = []
    opened_archives: list[str] = []

    global_segment_offset = 0

    for index, day in enumerate(
        ordered_calibration_days,
        start=1,
    ):
        # Explicitly reject TRAIN/EVAL access in this CAL stage.
        if day in train_days:
            raise RuntimeError(
                f"TRAIN day unexpectedly requested: {day}"
            )
        if day in evaluation_days:
            raise RuntimeError(
                f"EVALUATION raw access forbidden: {day}"
            )
        if day not in calibration_days:
            raise RuntimeError(
                f"Day is outside CALIBRATION: {day}"
            )

        metadata = archive_by_day[day]

        if metadata["partition"] != "CALIBRATION":
            raise RuntimeError(
                f"Manifest does not mark {day} as CALIBRATION."
            )

        archive_name = str(metadata["archive"])
        source_record = primary_files[archive_name]
        archive_path = ROOT / source_record["path"]

        observed_sha = _sha256(archive_path)

        if observed_sha != source_record["sha256"]:
            raise RuntimeError(
                f"Source SHA mismatch: {archive_name}"
            )

        print(
            f"[{index}/4] opening CALIBRATION {archive_name}",
            flush=True,
        )

        source_frame = _read_train_archive(
            archive_path=archive_path,
            archive_metadata=metadata,
            feature_names=features,
        )

        opened_archives.append(archive_name)

        five = aggregate_five_minute(
            source_frame,
            timestamp_column=str(
                metadata["timestamp_column"]
            ),
            feature_names=features,
        )

        frame = five.frame.copy()

        if not frame.empty:
            frame["segment_id"] = (
                frame["segment_id"].to_numpy(
                    dtype=np.int64
                )
                + global_segment_offset
            )

            global_segment_offset = (
                int(frame["segment_id"].max()) + 1
            )

        frame.insert(0, "day", day)
        day_frames.append(frame)

        per_day.append(
            {
                "day": day,
                "archive": archive_name,
                "source_rows": five.source_rows,
                "produced_5min_bins": five.produced_bins,
                "valid_5min_bins": five.valid_bins,
                "invalid_5min_bins": five.invalid_bins,
                "contiguous_segments": (
                    int(frame["segment_id"].nunique())
                    if not frame.empty
                    else 0
                ),
            }
        )

    calibration = pd.concat(
        day_frames,
        ignore_index=True,
    )

    calibration = calibration.loc[
        :,
        [
            "day",
            "timestamp",
            "segment_id",
            *features,
        ],
    ]

    if calibration.empty:
        raise RuntimeError(
            "CALIBRATION produced zero valid bins."
        )

    values = calibration[
        features
    ].to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        raise RuntimeError(
            "CALIBRATION representation is non-finite."
        )

    CAL_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    calibration.to_csv(
        CAL_DATA_PATH,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        date_format="%Y-%m-%d %H:%M:%S",
        float_format="%.17g",
    )

    cal_semantic_sha = semantic_representation_sha256(
        calibration,
        feature_names=features,
    )

    # ---------------------------------------------------------
    # Frozen PCA -> reconstruction error -> causal EWMA
    # ---------------------------------------------------------
    pca_scaled = transform_frame(
        calibration,
        features,
        pca_scaler,
    )

    latent = pca.transform(pca_scaled)
    reconstructed = pca.inverse_transform(latent)

    pca_raw = np.mean(
        (pca_scaled - reconstructed) ** 2,
        axis=1,
    )

    alpha = float(
        config["pca"]["ewma"]["alpha"]
    )

    pca_ewma = causal_ewma(
        pca_raw,
        calibration["segment_id"].to_numpy(),
        alpha=alpha,
    )

    pca_reference = np.sort(pca_ewma.copy())

    pca_q90 = higher_quantile(
        pca_reference,
        0.90,
    )
    pca_q95 = higher_quantile(
        pca_reference,
        0.95,
    )

    pca_percentile = right_inclusive_percentile(
        pca_ewma,
        pca_reference,
    )

    history_bins = int(
        config["router_v2"][
            "recent_q90_hit_fraction"
        ]["history_bins"]
    )

    recent_q90 = recent_hit_fraction(
        pca_ewma,
        calibration["segment_id"].to_numpy(),
        threshold=pca_q90,
        history_bins=history_bins,
    )

    recent_q95 = recent_hit_fraction(
        pca_ewma,
        calibration["segment_id"].to_numpy(),
        threshold=pca_q95,
        history_bins=history_bins,
    )

    # ---------------------------------------------------------
    # Frozen TCN teacher
    # ---------------------------------------------------------
    indexed = (
        calibration.sort_values("timestamp")
        .set_index("timestamp")
    )

    detector = TemporalDetector(
        features=features,
        scaler=tcn_scaler,
        model=tcn_model,
        sequence_length=int(
            config["temporal_model"][
                "sequence"
            ]["history_bins"]
        ),
        bin_minutes=int(
            config["temporal_model"][
                "sequence"
            ]["bin_minutes"]
        ),
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tcn_scores = score_temporal_detector(
        indexed,
        detector,
        device=device,
        batch_size=1024,
    )

    expected_targets = expected_sequence_count(
        calibration["segment_id"],
        history_bins=int(
            config["temporal_model"][
                "sequence"
            ]["history_bins"]
        ),
    )

    if len(tcn_scores) != expected_targets:
        raise RuntimeError(
            "CAL TCN target count differs from segment contract: "
            f"{len(tcn_scores)} != {expected_targets}"
        )

    if len(tcn_scores) == 0:
        raise RuntimeError(
            "CALIBRATION has zero eligible TCN targets."
        )

    tcn_values = tcn_scores.to_numpy(dtype=float)

    tcn_q90 = higher_quantile(
        tcn_values,
        float(
            config["calibration"][
                "tcn_teacher"
            ]["high_evidence_quantile"]
        ),
    )

    tcn_q995 = higher_quantile(
        tcn_values,
        float(
            config["calibration"][
                "tcn_teacher"
            ]["alert_evidence_quantile"]
        ),
    )

    # ---------------------------------------------------------
    # Router V2 at exactly the TCN target timestamps
    # ---------------------------------------------------------
    score_frame = pd.DataFrame(
        {
            "timestamp": calibration["timestamp"],
            "segment_id": calibration["segment_id"],
            "pca_raw": pca_raw,
            "pca_ewma": pca_ewma,
            "pca_ewma_percentile": pca_percentile,
            "recent_q90_hit_fraction": recent_q90,
            "recent_q95_hit_fraction": recent_q95,
        }
    ).set_index("timestamp")

    router_features = score_frame.loc[
        tcn_scores.index,
        [
            "pca_ewma_percentile",
            "recent_q90_hit_fraction",
            "recent_q95_hit_fraction",
        ],
    ]

    candidate = RouterV2Candidate(
        family="max3",
        threshold=0.40,
    )

    route, router_score, forced_missing = (
        route_candidate(
            router_features,
            candidate,
        )
    )

    high = tcn_values >= tcn_q90
    alert = tcn_values >= tcn_q995

    high_coverage = evidence_coverage(
        route,
        high,
    )
    alert_coverage = evidence_coverage(
        route,
        alert,
    )

    score_output = router_features.copy()
    score_output["router_score"] = router_score
    score_output["route_tcn"] = route
    score_output["tcn_score"] = tcn_values
    score_output["teacher_high"] = high
    score_output["teacher_alert"] = alert

    score_output.to_csv(
        CAL_SCORES_PATH,
        index=True,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.17g",
    )

    result = {
        "schema_version": (
            "aeroxai.cobra_router_v2_calibration.v1"
        ),
        "study": "cobra_router_v2_adaptation",
        "status": "CALIBRATION_FROZEN",
        "implementation_commit": _git_head(),
        "train_fit_commit": REQUIRED_TRAIN_FIT_COMMIT,
        "data_use": {
            "partition_opened": "CALIBRATION",
            "opened_archives": opened_archives,
            "calibration_day_count": 4,
            "evaluation_archives_opened": False,
            "evaluation_sensor_values_used": False,
        },
        "representation": {
            "valid_5min_bins": len(calibration),
            "invalid_5min_bins": sum(
                int(item["invalid_5min_bins"])
                for item in per_day
            ),
            "contiguous_segments": int(
                calibration["segment_id"].nunique()
            ),
            "semantic_sha256": cal_semantic_sha,
            "file_sha256": _sha256(CAL_DATA_PATH),
            "per_day": per_day,
        },
        "pca": {
            "ewma_alpha": alpha,
            "reference_count": len(pca_reference),
            "reference_sorted": [
                float(value)
                for value in pca_reference
            ],
            "q90": pca_q90,
            "q95": pca_q95,
            "percentile_definition": (
                "right_inclusive_empirical_percentile"
            ),
            "quantile_interpolation": "higher",
        },
        "tcn_teacher": {
            "eligible_targets": len(tcn_scores),
            "device": device,
            "high_quantile": 0.90,
            "alert_quantile": 0.995,
            "q90": tcn_q90,
            "q995": tcn_q995,
            "high_evidence_units": int(high.sum()),
            "alert_evidence_units": int(alert.sum()),
            "quantile_interpolation": "higher",
        },
        "router_v2": {
            "family": "max3",
            "threshold": 0.40,
            "eligible_targets": len(router_features),
            "forced_missing_input_routes": forced_missing,
            "tcn_invocation_fraction": float(
                route.mean()
            ),
            "high_evidence_coverage": high_coverage,
            "alert_evidence_coverage": alert_coverage,
        },
        "local_outputs": {
            "calibration_data_path": str(
                CAL_DATA_PATH.relative_to(ROOT)
            ),
            "calibration_scores_path": str(
                CAL_SCORES_PATH.relative_to(ROOT)
            ),
            "scores_sha256": _sha256(CAL_SCORES_PATH),
        },
        "firewall": {
            "evaluation_opened": False,
            "evaluation_used_for_tuning": False,
            "router_policy_changed": False,
            "feature_contract_changed": False,
            "model_refit_on_calibration": False,
        },
    }

    RESULT_PATH.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": result["status"],
                "valid_5min_bins": len(calibration),
                "segments": result[
                    "representation"
                ]["contiguous_segments"],
                "tcn_targets": len(tcn_scores),
                "pca_q90": pca_q90,
                "pca_q95": pca_q95,
                "tcn_q90": tcn_q90,
                "tcn_q995": tcn_q995,
                "high_units": int(high.sum()),
                "alert_units": int(alert.sum()),
                "router_invocation": float(
                    route.mean()
                ),
                "high_coverage": high_coverage,
                "alert_coverage": alert_coverage,
                "forced_missing_routes": (
                    forced_missing
                ),
                "evaluation_opened": False,
                "result": str(
                    RESULT_PATH.relative_to(ROOT)
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
