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
from scripts.calibrate_cobra_router_v2 import (
    causal_ewma,
    recent_hit_fraction,
    right_inclusive_percentile,
)
from scripts.fit_cobra_train_models import (
    expected_sequence_count,
)

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT / "configs/cobra_router_v2_adaptation.yaml"
)
SCHEMA_CONFIG_PATH = (
    ROOT / "configs/cobra_schema_chronology.yaml"
)
SCHEMA_MANIFEST_PATH = (
    ROOT
    / "docs/research/cobra_schema_chronology_manifest.json"
)
SOURCE_LOCK_PATH = (
    ROOT
    / "docs/research/cobra_source_lock_manifest.json"
)
FEATURE_PATH = (
    ROOT
    / "docs/research/cobra_router_v2_raw_feature_contract.json"
)
TRAIN_FIT_PATH = (
    ROOT
    / "docs/research/cobra_router_v2_train_fit.json"
)
CALIBRATION_PATH = (
    ROOT
    / "docs/research/cobra_router_v2_calibration.json"
)

SKLEARN_ARTIFACT = (
    ROOT
    / "artifacts/research/cobra_router_v2_train_models.pkl"
)
TCN_ARTIFACT = (
    ROOT
    / "artifacts/research/cobra_router_v2_tcn.pt"
)

EVAL_DATA_PATH = (
    ROOT
    / "data/processed/cobra_router_v2/evaluation_5min.csv"
)
EVAL_SCORES_PATH = (
    ROOT
    / "data/processed/cobra_router_v2/evaluation_scores.csv"
)
RESULT_PATH = (
    ROOT
    / "docs/research/cobra_router_v2_evaluation.json"
)

REQUIRED_CALIBRATION_COMMIT = (
    "af4e8a17a2bbce5dfd938fa11d6a4d681d9846c8"
)

EXPECTED_FEATURE_SHA = (
    "ba54ffa70cda5690ceaef4d0a58d6d14129dd36a0e03abbc95c20344e3e4a9aa"
)

IMPLEMENTATION_PATHS = [
    "scripts/evaluate_cobra_router_v2.py",
    "tests/test_cobra_router_v2_evaluation.py",
]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected JSON mapping in {path}."
        )

    return value


def _yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected YAML mapping in {path}."
        )

    return value


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def criterion_status(
    *,
    evidence_count: int,
    minimum_count: int,
    coverage: float | None,
    minimum_coverage: float,
) -> str:
    if evidence_count < minimum_count:
        return "INSUFFICIENT_EVIDENCE"

    if coverage is None:
        raise RuntimeError(
            "Adequate evidence count has no coverage."
        )

    if coverage >= minimum_coverage:
        return "PASS"

    return "FAIL"


def _verify_boundary() -> None:
    ancestor = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            REQUIRED_CALIBRATION_COMMIT,
            "HEAD",
        ],
        cwd=ROOT,
        check=False,
    )

    if ancestor.returncode != 0:
        raise RuntimeError(
            "Frozen CALIBRATION result is not "
            "an ancestor of HEAD."
        )

    paths_to_check = [
        *IMPLEMENTATION_PATHS,
        str(CALIBRATION_PATH.relative_to(ROOT)),
    ]

    status = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--",
            *paths_to_check,
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    if status.stdout.strip():
        raise RuntimeError(
            "EVAL implementation or frozen "
            "CALIBRATION result is dirty."
        )

    for path in (
        EVAL_DATA_PATH,
        EVAL_SCORES_PATH,
        RESULT_PATH,
    ):
        if path.exists():
            raise RuntimeError(
                "Refusing to overwrite existing "
                f"EVAL output: {path}"
            )


def _load_tcn() -> TCNForecaster:
    payload = torch.load(
        TCN_ARTIFACT,
        map_location="cpu",
        weights_only=False,
    )

    model_payload = payload["model_config"]

    model_config = TemporalForecastConfig(
        input_dim=int(
            model_payload["input_dim"]
        ),
        hidden_dim=int(
            model_payload["hidden_dim"]
        ),
        kernel_size=int(
            model_payload["kernel_size"]
        ),
        dilations=tuple(
            int(value)
            for value in model_payload[
                "dilations"
            ]
        ),
        dropout=float(
            model_payload["dropout"]
        ),
    )

    model = TCNForecaster(
        model_config
    )

    model.load_state_dict(
        payload["state_dict"]
    )

    model.eval()

    return model


def _per_day_metrics(
    *,
    days: np.ndarray,
    route: np.ndarray,
    high: np.ndarray,
    alert: np.ndarray,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []

    for day in dict.fromkeys(
        str(value)
        for value in days.tolist()
    ):
        mask = days == day
        count = int(mask.sum())

        high_count = int(
            high[mask].sum()
        )
        alert_count = int(
            alert[mask].sum()
        )

        reports.append(
            {
                "day": day,
                "eligible_targets": count,
                "tcn_invocation_fraction": (
                    float(route[mask].mean())
                    if count
                    else None
                ),
                "high_evidence_units": (
                    high_count
                ),
                "high_evidence_coverage": (
                    evidence_coverage(
                        route[mask],
                        high[mask],
                    )
                    if high_count
                    else None
                ),
                "alert_evidence_units": (
                    alert_count
                ),
                "alert_evidence_coverage": (
                    evidence_coverage(
                        route[mask],
                        alert[mask],
                    )
                    if alert_count
                    else None
                ),
            }
        )

    return reports


def main() -> None:
    _verify_boundary()

    config = _yaml(
        CONFIG_PATH
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
        FEATURE_PATH
    )
    train_fit = _json(
        TRAIN_FIT_PATH
    )
    calibration = _json(
        CALIBRATION_PATH
    )

    if calibration["status"] != (
        "CALIBRATION_FROZEN"
    ):
        raise RuntimeError(
            "CALIBRATION result status changed."
        )

    if calibration["data_use"][
        "evaluation_sensor_values_used"
    ]:
        raise RuntimeError(
            "Frozen CALIBRATION result already "
            "claims EVAL sensor use."
        )

    features = [
        str(value)
        for value in feature_contract[
            "final_raw_features"
        ]
    ]

    if len(features) != 219:
        raise RuntimeError(
            "Expected 219 frozen features."
        )

    if feature_contract[
        "final_raw_feature_sha256"
    ] != EXPECTED_FEATURE_SHA:
        raise RuntimeError(
            "Frozen feature SHA changed."
        )

    if _sha256(
        SKLEARN_ARTIFACT
    ) != train_fit[
        "artifacts"
    ][
        "sklearn_sha256"
    ]:
        raise RuntimeError(
            "Frozen sklearn artifact SHA mismatch."
        )

    if _sha256(
        TCN_ARTIFACT
    ) != train_fit[
        "artifacts"
    ][
        "tcn_sha256"
    ]:
        raise RuntimeError(
            "Frozen TCN artifact SHA mismatch."
        )

    with SKLEARN_ARTIFACT.open(
        "rb"
    ) as handle:
        trained = pickle.load(
            handle
        )

    if trained["features"] != features:
        raise RuntimeError(
            "Frozen model feature order changed."
        )

    pca_scaler = trained[
        "pca_scaler"
    ]
    pca = trained["pca"]
    tcn_scaler = trained[
        "tcn_scaler"
    ]

    tcn_model = _load_tcn()

    frozen_pca_reference = np.asarray(
        calibration["pca"][
            "reference_sorted"
        ],
        dtype=float,
    )

    pca_q90 = float(
        calibration["pca"]["q90"]
    )
    pca_q95 = float(
        calibration["pca"]["q95"]
    )

    tcn_q90 = float(
        calibration[
            "tcn_teacher"
        ]["q90"]
    )
    tcn_q995 = float(
        calibration[
            "tcn_teacher"
        ]["q995"]
    )

    if calibration[
        "router_v2"
    ]["family"] != "max3":
        raise RuntimeError(
            "Frozen Router V2 family changed."
        )

    if float(
        calibration[
            "router_v2"
        ]["threshold"]
    ) != 0.40:
        raise RuntimeError(
            "Frozen Router V2 threshold changed."
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

    ordered_evaluation_days = [
        str(value)
        for value in split[
            "evaluation_days"
        ]
    ]

    evaluation_days = set(
        ordered_evaluation_days
    )

    if len(
        ordered_evaluation_days
    ) != 5:
        raise RuntimeError(
            "Expected five EVALUATION days."
        )

    if (
        evaluation_days & train_days
        or evaluation_days
        & calibration_days
    ):
        raise RuntimeError(
            "EVALUATION split overlaps earlier "
            "partitions."
        )

    archive_by_day = (
        _archive_metadata_by_day(
            schema_manifest
        )
    )

    primary_files = source_lock[
        "primary_local_files"
    ]

    day_frames: list[
        pd.DataFrame
    ] = []

    per_day_representation: list[
        dict[str, Any]
    ] = []

    opened_archives: list[
        str
    ] = []

    global_segment_offset = 0

    for index, day in enumerate(
        ordered_evaluation_days,
        start=1,
    ):
        if day not in evaluation_days:
            raise RuntimeError(
                f"Non-EVAL day requested: {day}"
            )

        metadata = archive_by_day[
            day
        ]

        if metadata[
            "partition"
        ] != "EVALUATION":
            raise RuntimeError(
                f"Manifest does not mark "
                f"{day} as EVALUATION."
            )

        archive_name = str(
            metadata["archive"]
        )

        source_record = (
            primary_files[
                archive_name
            ]
        )

        archive_path = (
            ROOT
            / source_record["path"]
        )

        observed_sha = _sha256(
            archive_path
        )

        if observed_sha != (
            source_record[
                "sha256"
            ]
        ):
            raise RuntimeError(
                "Source SHA mismatch: "
                f"{archive_name}"
            )

        print(
            f"[{index}/5] opening EVALUATION "
            f"{archive_name}",
            flush=True,
        )

        source_frame = (
            _read_train_archive(
                archive_path=archive_path,
                archive_metadata=metadata,
                feature_names=features,
            )
        )

        opened_archives.append(
            archive_name
        )

        five = (
            aggregate_five_minute(
                source_frame,
                timestamp_column=str(
                    metadata[
                        "timestamp_column"
                    ]
                ),
                feature_names=features,
            )
        )

        frame = five.frame.copy()

        if not frame.empty:
            frame[
                "segment_id"
            ] = (
                frame[
                    "segment_id"
                ].to_numpy(
                    dtype=np.int64
                )
                + global_segment_offset
            )

            global_segment_offset = (
                int(
                    frame[
                        "segment_id"
                    ].max()
                )
                + 1
            )

        frame.insert(
            0,
            "day",
            day,
        )

        day_frames.append(
            frame
        )

        per_day_representation.append(
            {
                "day": day,
                "archive": archive_name,
                "source_rows": (
                    five.source_rows
                ),
                "produced_5min_bins": (
                    five.produced_bins
                ),
                "valid_5min_bins": (
                    five.valid_bins
                ),
                "invalid_5min_bins": (
                    five.invalid_bins
                ),
                "contiguous_segments": (
                    int(
                        frame[
                            "segment_id"
                        ].nunique()
                    )
                    if not frame.empty
                    else 0
                ),
            }
        )

    expected_archives = [
        "CoBra"
        + day.replace(
            "-",
            "",
        )
        + ".zip"
        for day in (
            ordered_evaluation_days
        )
    ]

    if opened_archives != (
        expected_archives
    ):
        raise RuntimeError(
            "EVALUATION archive sequence "
            "changed."
        )

    evaluation = pd.concat(
        day_frames,
        ignore_index=True,
    )

    evaluation = evaluation.loc[
        :,
        [
            "day",
            "timestamp",
            "segment_id",
            *features,
        ],
    ]

    if evaluation.empty:
        raise RuntimeError(
            "EVALUATION produced zero valid bins."
        )

    values = evaluation[
        features
    ].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(
        values
    ).all():
        raise RuntimeError(
            "EVALUATION representation "
            "contains non-finite values."
        )

    EVAL_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation.to_csv(
        EVAL_DATA_PATH,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        date_format=(
            "%Y-%m-%d %H:%M:%S"
        ),
        float_format="%.17g",
    )

    semantic_sha = (
        semantic_representation_sha256(
            evaluation,
            feature_names=features,
        )
    )

    # ------------------------------------------------------
    # Frozen PCA path
    # ------------------------------------------------------
    pca_scaled = transform_frame(
        evaluation,
        features,
        pca_scaler,
    )

    latent = pca.transform(
        pca_scaled
    )

    reconstructed = (
        pca.inverse_transform(
            latent
        )
    )

    pca_raw = np.mean(
        (
            pca_scaled
            - reconstructed
        )
        ** 2,
        axis=1,
    )

    pca_ewma = causal_ewma(
        pca_raw,
        evaluation[
            "segment_id"
        ].to_numpy(),
        alpha=float(
            config[
                "pca"
            ][
                "ewma"
            ][
                "alpha"
            ]
        ),
    )

    pca_percentile = (
        right_inclusive_percentile(
            pca_ewma,
            frozen_pca_reference,
        )
    )

    history_bins = int(
        config[
            "router_v2"
        ][
            "recent_q90_hit_fraction"
        ][
            "history_bins"
        ]
    )

    recent_q90 = (
        recent_hit_fraction(
            pca_ewma,
            evaluation[
                "segment_id"
            ].to_numpy(),
            threshold=pca_q90,
            history_bins=history_bins,
        )
    )

    recent_q95 = (
        recent_hit_fraction(
            pca_ewma,
            evaluation[
                "segment_id"
            ].to_numpy(),
            threshold=pca_q95,
            history_bins=history_bins,
        )
    )

    # ------------------------------------------------------
    # Frozen TCN teacher
    # ------------------------------------------------------
    indexed = (
        evaluation
        .sort_values(
            "timestamp"
        )
        .set_index(
            "timestamp"
        )
    )

    sequence_config = config[
        "temporal_model"
    ][
        "sequence"
    ]

    expected_targets = (
        expected_sequence_count(
            evaluation[
                "segment_id"
            ],
            history_bins=int(
                sequence_config[
                    "history_bins"
                ]
            ),
        )
    )

    if expected_targets <= 0:
        raise RuntimeError(
            "EVALUATION has zero eligible "
            "TCN targets."
        )

    detector = TemporalDetector(
        features=features,
        scaler=tcn_scaler,
        model=tcn_model,
        sequence_length=int(
            sequence_config[
                "history_bins"
            ]
        ),
        bin_minutes=int(
            sequence_config[
                "bin_minutes"
            ]
        ),
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    tcn_scores = (
        score_temporal_detector(
            indexed,
            detector,
            device=device,
            batch_size=1024,
        )
    )

    if len(
        tcn_scores
    ) != expected_targets:
        raise RuntimeError(
            "EVALUATION TCN target count "
            "differs from frozen segments: "
            f"{len(tcn_scores)} != "
            f"{expected_targets}"
        )

    tcn_values = (
        tcn_scores.to_numpy(
            dtype=float
        )
    )

    # These labels use CAL-frozen thresholds.
    # No EVAL quantile is calculated.
    high = (
        tcn_values
        >= tcn_q90
    )

    alert = (
        tcn_values
        >= tcn_q995
    )

    # ------------------------------------------------------
    # Frozen Router V2
    # ------------------------------------------------------
    score_frame = pd.DataFrame(
        {
            "timestamp": (
                evaluation[
                    "timestamp"
                ]
            ),
            "day": (
                evaluation[
                    "day"
                ]
            ),
            "segment_id": (
                evaluation[
                    "segment_id"
                ]
            ),
            "pca_raw": pca_raw,
            "pca_ewma": pca_ewma,
            "pca_ewma_percentile": (
                pca_percentile
            ),
            "recent_q90_hit_fraction": (
                recent_q90
            ),
            "recent_q95_hit_fraction": (
                recent_q95
            ),
        }
    ).set_index(
        "timestamp"
    )

    target_rows = score_frame.loc[
        tcn_scores.index
    ]

    router_features = target_rows[
        [
            "pca_ewma_percentile",
            "recent_q90_hit_fraction",
            "recent_q95_hit_fraction",
        ]
    ]

    candidate = RouterV2Candidate(
        family="max3",
        threshold=0.40,
    )

    (
        route,
        router_score,
        forced_missing,
    ) = route_candidate(
        router_features,
        candidate,
    )

    high_count = int(
        high.sum()
    )
    alert_count = int(
        alert.sum()
    )

    high_coverage = (
        evidence_coverage(
            route,
            high,
        )
    )

    alert_coverage = (
        evidence_coverage(
            route,
            alert,
        )
    )

    criteria = config[
        "success_criteria"
    ]

    minimum_high_units = int(
        criteria[
            "minimum_high_evidence_units"
        ]
    )

    minimum_alert_units = int(
        criteria[
            "minimum_alert_evidence_units"
        ]
    )

    minimum_high_coverage = float(
        criteria[
            "minimum_high_evidence_coverage"
        ]
    )

    minimum_alert_coverage = float(
        criteria[
            "minimum_alert_evidence_coverage"
        ]
    )

    high_status = (
        criterion_status(
            evidence_count=high_count,
            minimum_count=minimum_high_units,
            coverage=high_coverage,
            minimum_coverage=(
                minimum_high_coverage
            ),
        )
    )

    alert_status = (
        criterion_status(
            evidence_count=alert_count,
            minimum_count=minimum_alert_units,
            coverage=alert_coverage,
            minimum_coverage=(
                minimum_alert_coverage
            ),
        )
    )

    target_days = target_rows[
        "day"
    ].to_numpy(
        dtype=str
    )

    per_day = _per_day_metrics(
        days=target_days,
        route=route,
        high=high,
        alert=alert,
    )

    score_output = (
        router_features.copy()
    )

    score_output.insert(
        0,
        "day",
        target_days,
    )

    score_output[
        "router_score"
    ] = router_score

    score_output[
        "route_tcn"
    ] = route

    score_output[
        "tcn_score"
    ] = tcn_values

    score_output[
        "teacher_high"
    ] = high

    score_output[
        "teacher_alert"
    ] = alert

    score_output.to_csv(
        EVAL_SCORES_PATH,
        index=True,
        encoding="utf-8",
        lineterminator="\n",
        float_format="%.17g",
    )

    invocation = float(
        route.mean()
    )

    result = {
        "schema_version": (
            "aeroxai.cobra_router_v2_evaluation.v1"
        ),
        "study": (
            "cobra_router_v2_adaptation"
        ),
        "status": (
            "EVALUATION_FUNCTIONAL_COMPLETE"
        ),
        "implementation_commit": (
            _git_head()
        ),
        "calibration_commit": (
            REQUIRED_CALIBRATION_COMMIT
        ),
        "evaluation_role": (
            "genuinely_independent_future_partition_"
            "after_preregistered_within_domain_adaptation"
        ),
        "zero_shot_claim": False,
        "data_use": {
            "partition_opened": (
                "EVALUATION"
            ),
            "opened_archives": (
                opened_archives
            ),
            "evaluation_day_count": 5,
            "model_refit": False,
            "threshold_refit": False,
            "feature_selection": False,
            "router_policy_changed": False,
        },
        "frozen_inputs": {
            "feature_count": 219,
            "feature_sha256": (
                EXPECTED_FEATURE_SHA
            ),
            "pca_q90": pca_q90,
            "pca_q95": pca_q95,
            "tcn_q90": tcn_q90,
            "tcn_q995": tcn_q995,
            "router_family": "max3",
            "router_threshold": 0.40,
        },
        "representation": {
            "source_rows": sum(
                int(
                    item[
                        "source_rows"
                    ]
                )
                for item in (
                    per_day_representation
                )
            ),
            "valid_5min_bins": (
                len(evaluation)
            ),
            "invalid_5min_bins": sum(
                int(
                    item[
                        "invalid_5min_bins"
                    ]
                )
                for item in (
                    per_day_representation
                )
            ),
            "contiguous_segments": int(
                evaluation[
                    "segment_id"
                ].nunique()
            ),
            "semantic_sha256": (
                semantic_sha
            ),
            "file_sha256": (
                _sha256(
                    EVAL_DATA_PATH
                )
            ),
            "per_day": (
                per_day_representation
            ),
        },
        "teacher_evidence": {
            "eligible_targets": (
                len(tcn_scores)
            ),
            "device": device,
            "high_threshold": (
                tcn_q90
            ),
            "alert_threshold": (
                tcn_q995
            ),
            "high_evidence_units": (
                high_count
            ),
            "alert_evidence_units": (
                alert_count
            ),
        },
        "router_v2": {
            "family": "max3",
            "threshold": 0.40,
            "eligible_targets": (
                len(route)
            ),
            "forced_missing_input_routes": (
                forced_missing
            ),
            "tcn_invocation_fraction": (
                invocation
            ),
            "tcn_call_reduction_fraction": (
                1.0 - invocation
            ),
            "high_evidence_coverage": (
                high_coverage
            ),
            "alert_evidence_coverage": (
                alert_coverage
            ),
        },
        "success_criteria": {
            "high": {
                "minimum_units": (
                    minimum_high_units
                ),
                "observed_units": (
                    high_count
                ),
                "minimum_coverage": (
                    minimum_high_coverage
                ),
                "observed_coverage": (
                    high_coverage
                ),
                "status": (
                    high_status
                ),
            },
            "alert": {
                "minimum_units": (
                    minimum_alert_units
                ),
                "observed_units": (
                    alert_count
                ),
                "minimum_coverage": (
                    minimum_alert_coverage
                ),
                "observed_coverage": (
                    alert_coverage
                ),
                "status": (
                    alert_status
                ),
            },
            "gpu_energy": {
                "status": (
                    "NOT_YET_EVALUATED"
                )
            },
            "overall": (
                "PENDING_GPU_ENERGY_MEASUREMENT"
            ),
        },
        "per_day_routing": (
            per_day
        ),
        "local_outputs": {
            "evaluation_data_path": str(
                EVAL_DATA_PATH.relative_to(
                    ROOT
                )
            ),
            "evaluation_scores_path": str(
                EVAL_SCORES_PATH.relative_to(
                    ROOT
                )
            ),
            "scores_sha256": (
                _sha256(
                    EVAL_SCORES_PATH
                )
            ),
        },
        "claim_boundaries": {
            "causal_fault_diagnosis": False,
            "autonomous_control": False,
            "plant_energy_savings": False,
            "carbon_savings": False,
            "zero_shot_generalization": False,
            "gpu_energy_savings_claimed": False,
        },
    }

    RESULT_PATH.write_text(
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
                "valid_5min_bins": (
                    len(evaluation)
                ),
                "segments": result[
                    "representation"
                ][
                    "contiguous_segments"
                ],
                "tcn_targets": (
                    len(tcn_scores)
                ),
                "high_units": (
                    high_count
                ),
                "alert_units": (
                    alert_count
                ),
                "router_invocation": (
                    invocation
                ),
                "tcn_call_reduction": (
                    1.0 - invocation
                ),
                "high_coverage": (
                    high_coverage
                ),
                "high_status": (
                    high_status
                ),
                "alert_coverage": (
                    alert_coverage
                ),
                "alert_status": (
                    alert_status
                ),
                "energy_status": (
                    "NOT_YET_EVALUATED"
                ),
                "result": str(
                    RESULT_PATH.relative_to(
                        ROOT
                    )
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
