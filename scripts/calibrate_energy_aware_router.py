from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.data.features import model_feature_columns
from ml.detection.alerts import causal_ewma
from ml.detection.pca_detector import (
    fit_pca_detector,
    score_pca_detector,
)
from ml.energy.hierarchical_inference import (
    higher_quantile,
    routing_mask,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY_CONFIG = (
    ROOT
    / "configs"
    / "energy_aware_hierarchical_intelligence.yaml"
)
BENCHMARK_CONFIG = (
    ROOT
    / "configs"
    / "energy_aware_inference_benchmark.yaml"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise TypeError(
            f"Expected mapping in {path}."
        )
    return payload


def _load_partition(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = pd.DatetimeIndex(
        frame.index,
        name="timestamp",
    )
    return frame.sort_index()


def main() -> None:
    study = _load_yaml(STUDY_CONFIG)
    benchmark = _load_yaml(BENCHMARK_CONFIG)

    if not bool(
        benchmark["benchmark"][
            "frozen_before_primary_test"
        ]
    ):
        raise ValueError(
            "Benchmark must be frozen before TEST."
        )

    primary = study["datasets"]["primary"]
    frozen_pca = study["frozen_pca"]
    router_config = benchmark["router"]
    tcn_config = benchmark["tcn"]
    outputs = benchmark["outputs"]

    # This script intentionally never loads primary["test_file"].
    train = _load_partition(
        ROOT / primary["train_file"]
    )
    calibration = _load_partition(
        ROOT / primary["calibration_file"]
    )

    features = model_feature_columns(
        train.columns
    )

    if len(features) != 63:
        raise RuntimeError(
            "Expected frozen 63-feature representation, "
            f"got {len(features)}."
        )

    if router_config["scaler"] != "robust":
        raise ValueError(
            "Router PCA must remain RobustScaler-based."
        )

    if float(
        router_config["variance_retained"]
    ) != float(
        frozen_pca["variance_retained"]
    ):
        raise ValueError(
            "Router PCA variance retention does not "
            "match the frozen PCA."
        )

    if float(
        router_config["ewma_alpha"]
    ) != float(
        frozen_pca["alerting"]["ewma_alpha"]
    ):
        raise ValueError(
            "Router EWMA alpha does not match "
            "the frozen PCA."
        )

    if int(
        router_config["reset_gap_minutes"]
    ) != int(
        frozen_pca["alerting"][
            "reset_gap_minutes"
        ]
    ):
        raise ValueError(
            "Router reset gap does not match "
            "the frozen PCA."
        )

    if (
        router_config["threshold_source"]
        != "official_calibration_split_only"
    ):
        raise ValueError(
            "Router thresholds must come from "
            "official CALIBRATION only."
        )

    if (
        router_config["quantile_interpolation"]
        != "higher"
    ):
        raise ValueError(
            "Router quantile interpolation must "
            "remain 'higher'."
        )

    checkpoint_path = (
        ROOT / tcn_config["checkpoint_file"]
    )
    observed_checkpoint_sha = sha256_file(
        checkpoint_path
    )
    expected_checkpoint_sha = str(
        tcn_config["checkpoint_sha256"]
    )

    if (
        observed_checkpoint_sha
        != expected_checkpoint_sha
    ):
        raise RuntimeError(
            "TCN checkpoint SHA256 mismatch. "
            f"Expected {expected_checkpoint_sha}, "
            f"got {observed_checkpoint_sha}."
        )

    detector = fit_pca_detector(
        train,
        features,
        variance_retained=float(
            router_config["variance_retained"]
        ),
        scaler_name=str(
            router_config["scaler"]
        ),
    )

    calibration_raw = score_pca_detector(
        calibration,
        detector,
    )

    calibration_smoothed = causal_ewma(
        calibration_raw,
        alpha=float(
            router_config["ewma_alpha"]
        ),
        reset_gap_minutes=int(
            router_config["reset_gap_minutes"]
        ),
    )

    operating_points: dict[
        str,
        dict[str, float | int],
    ] = {}

    for (
        operating_id,
        quantile_value,
    ) in router_config[
        "operating_points"
    ].items():
        quantile = float(quantile_value)
        threshold = higher_quantile(
            calibration_smoothed,
            quantile,
        )
        mask = routing_mask(
            calibration_smoothed,
            threshold,
        )
        routed = int(mask.sum())
        total = len(mask)

        operating_points[
            str(operating_id)
        ] = {
            "calibration_quantile": quantile,
            "threshold": threshold,
            "calibration_rows": total,
            "routed_rows": routed,
            "routed_fraction": (
                routed / total
                if total
                else 0.0
            ),
        }

    report = {
        "schema_version": (
            "aeroxai.energy_aware_router_calibration.v1"
        ),
        "study": study["study"]["name"],
        "evidence_class": (
            study["study"]["evidence_class"]
        ),
        "causal_claim": False,
        "purpose": (
            "Freeze PCA routing thresholds using TRAIN "
            "and official CALIBRATION only."
        ),
        "data_use": {
            "train_file": primary["train_file"],
            "calibration_file": (
                primary["calibration_file"]
            ),
            "test_file_opened": False,
        },
        "tcn_checkpoint": {
            "file": tcn_config[
                "checkpoint_file"
            ],
            "sha256": (
                observed_checkpoint_sha
            ),
        },
        "router": {
            "source_model": (
                router_config["source_model"]
            ),
            "feature_count": len(features),
            "scaler": (
                router_config["scaler"]
            ),
            "variance_retained": float(
                router_config[
                    "variance_retained"
                ]
            ),
            "ewma_alpha": float(
                router_config["ewma_alpha"]
            ),
            "reset_gap_minutes": int(
                router_config[
                    "reset_gap_minutes"
                ]
            ),
            "threshold_source": (
                router_config[
                    "threshold_source"
                ]
            ),
            "quantile_interpolation": (
                router_config[
                    "quantile_interpolation"
                ]
            ),
            "comparison_rule": (
                router_config[
                    "comparison_rule"
                ]
            ),
            "operating_points": (
                operating_points
            ),
        },
        "calibration_score_summary": {
            "rows": len(calibration_smoothed),
            "minimum": float(
                calibration_smoothed.min()
            ),
            "median": float(
                calibration_smoothed.median()
            ),
            "mean": float(
                calibration_smoothed.mean()
            ),
            "maximum": float(
                calibration_smoothed.max()
            ),
        },
    }

    output_path = (
        ROOT
        / outputs[
            "router_calibration_file"
        ]
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            report,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "test_file_opened": False,
                "tcn_checkpoint_sha256": (
                    observed_checkpoint_sha
                ),
                "operating_points": (
                    operating_points
                ),
                "output": str(
                    output_path.relative_to(ROOT)
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
