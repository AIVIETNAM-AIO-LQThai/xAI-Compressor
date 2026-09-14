from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ml.data.features import model_feature_columns
from ml.detection.common import fit_scaler, transform_frame
from ml.representation.manifold_geometry import (
    chronological_split,
    fit_global_pca,
    local_pca_audit,
    reconstruct_with_pca,
    reconstruction_mse,
    relationship_shape_audit,
    residual_structure_audit,
    two_nearest_neighbor_dimension,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "manifold_audit.yaml"


def _load_config() -> dict[str, Any]:
    payload = yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            "manifold_audit.yaml must contain a mapping."
        )

    return payload


def main() -> None:
    config = _load_config()

    train_path = ROOT / config[
        "dataset"
    ][
        "train_file"
    ]

    print(
        "Loading TRAIN ONLY:",
        train_path,
    )

    frame = pd.read_parquet(
        train_path
    ).sort_index()

    features = model_feature_columns(
        frame.columns
    )

    fit_frame, audit_frame = (
        chronological_split(
            frame,
            fit_fraction=float(
                config[
                    "split"
                ][
                    "fit_fraction"
                ]
            ),
        )
    )

    scaler = fit_scaler(
        fit_frame,
        features,
        method=str(
            config[
                "scaling"
            ][
                "method"
            ]
        ),
    )

    x_fit = transform_frame(
        fit_frame,
        features,
        scaler,
    )

    x_audit = transform_frame(
        audit_frame,
        features,
        scaler,
    )

    variance_targets = [
        float(value)
        for value in config[
            "global_pca"
        ][
            "variance_targets"
        ]
    ]

    global_result = fit_global_pca(
        x_fit,
        variance_targets=variance_targets,
    )

    components_95 = int(
        global_result.components_for_targets[
            f"{0.95:.4f}"
        ]
    )

    global_reconstruction = (
        reconstruct_with_pca(
            global_result.model,
            x_audit,
            component_count=(
                components_95
            ),
        )
    )

    global_error = reconstruction_mse(
        x_audit,
        global_reconstruction,
    )

    pc_scores = (
        global_result.model.transform(
            x_audit
        )
    )

    neighbor_counts = [
        int(value)
        for value in config[
            "local_geometry"
        ][
            "neighbor_counts"
        ]
    ]

    local_points, local_summary = (
        local_pca_audit(
            x_fit,
            x_audit,
            global_model=(
                global_result.model
            ),
            latent_dimension=(
                components_95
            ),
            neighbor_counts=(
                neighbor_counts
            ),
        )
    )

    point_table = pd.DataFrame(
        index=audit_frame.index
    )

    pc_count = int(
        config[
            "residual_structure"
        ][
            "pc_count"
        ]
    )

    for index in range(
        min(
            pc_count,
            pc_scores.shape[1],
        )
    ):
        point_table[
            f"PC{index + 1}"
        ] = pc_scores[
            :,
            index,
        ]

    for column in local_points.columns:
        point_table[
            column
        ] = local_points[
            column
        ].to_numpy()

    for feature in config[
        "residual_structure"
    ][
        "regime_features"
    ]:
        if feature in audit_frame.columns:
            point_table[
                feature
            ] = audit_frame[
                feature
            ]

    residual_structure = (
        residual_structure_audit(
            global_error=global_error,
            pc_scores=pc_scores,
            audit_frame=audit_frame,
            regime_features=list(
                config[
                    "residual_structure"
                ][
                    "regime_features"
                ]
            ),
            quantile_bins=int(
                config[
                    "residual_structure"
                ][
                    "quantile_bins"
                ]
            ),
        )
    )

    relationship_results = (
        relationship_shape_audit(
            fit_frame,
            audit_frame,
            pairs=list(
                config[
                    "relationship_shapes"
                ][
                    "pairs"
                ]
            ),
            spline_knots=int(
                config[
                    "relationship_shapes"
                ][
                    "spline_knots"
                ]
            ),
        )
    )

    two_nn = (
        two_nearest_neighbor_dimension(
            x_fit
        )
    )

    spectrum_rows = []

    for index, (
        eigenvalue,
        ratio,
        cumulative,
    ) in enumerate(
        zip(
            global_result.eigenvalues,
            global_result
            .explained_variance_ratio,
            global_result
            .cumulative_variance,
            strict=True,
        ),
        start=1,
    ):
        spectrum_rows.append(
            {
                "component": int(index),
                "eigenvalue": float(
                    eigenvalue
                ),
                "explained_variance_ratio": float(
                    ratio
                ),
                "cumulative_variance": float(
                    cumulative
                ),
            }
        )

    summary = {
        "schema_version": (
            "aeroxai.healthy_manifold_geometry.v1"
        ),
        "dataset": "MetroPT-3",
        "evidence_class": config[
            "scope"
        ][
            "evidence_class"
        ],
        "analysis_role": config[
            "scope"
        ][
            "analysis_role"
        ],
        "data_boundary": {
            "loaded_files": [
                str(
                    config[
                        "dataset"
                    ][
                        "train_file"
                    ]
                )
            ],
            "calibration_loaded": False,
            "test_loaded": False,
            "incident_labels_used": False,
        },
        "training_input": {
            "rows": len(frame),
            "feature_count": len(features),
            "start": str(
                frame.index.min()
            ),
            "end": str(
                frame.index.max()
            ),
        },
        "internal_split": {
            "fit_fraction": float(
                config[
                    "split"
                ][
                    "fit_fraction"
                ]
            ),
            "fit_rows": len(fit_frame),
            "audit_rows": len(audit_frame),
            "fit_start": str(
                fit_frame.index.min()
            ),
            "fit_end": str(
                fit_frame.index.max()
            ),
            "audit_start": str(
                audit_frame.index.min()
            ),
            "audit_end": str(
                audit_frame.index.max()
            ),
        },
        "global_pca": {
            "scaler": str(
                config[
                    "scaling"
                ][
                    "method"
                ]
            ),
            "participation_ratio_dimension": float(
                global_result
                .participation_ratio
            ),
            "components_for_variance_targets": (
                global_result
                .components_for_targets
            ),
            "latent_dimension_for_local_audit": (
                components_95
            ),
            "spectrum": spectrum_rows,
        },
        "two_nearest_neighbor_dimension": (
            two_nn
        ),
        "local_pca": (
            local_summary
        ),
        "residual_structure": (
            residual_structure
        ),
        "relationship_shapes": (
            relationship_results
            .replace(
                {
                    np.nan: None
                }
            )
            .to_dict(
                orient="records"
            )
        ),
        "guardrails": config[
            "guardrails"
        ],
    }

    outputs = config[
        "outputs"
    ]

    summary_path = (
        ROOT
        / outputs[
            "summary_json"
        ]
    )

    points_path = (
        ROOT
        / outputs[
            "point_table"
        ]
    )

    relationships_path = (
        ROOT
        / outputs[
            "relationship_table"
        ]
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    points_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    relationships_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    point_table.to_parquet(
        points_path
    )

    relationship_results.to_parquet(
        relationships_path,
        index=False,
    )

    compact = {
        "data_boundary": (
            summary[
                "data_boundary"
            ]
        ),
        "internal_split": (
            summary[
                "internal_split"
            ]
        ),
        "participation_ratio_dimension": (
            summary[
                "global_pca"
            ][
                "participation_ratio_dimension"
            ]
        ),
        "components_for_variance_targets": (
            summary[
                "global_pca"
            ][
                "components_for_variance_targets"
            ]
        ),
        "two_nearest_neighbor_dimension": (
            two_nn
        ),
        "local_pca": (
            local_summary
        ),
        "summary_json": str(
            summary_path
        ),
    }

    print(
        json.dumps(
            compact,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
