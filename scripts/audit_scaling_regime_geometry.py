from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ml.data.features import model_feature_columns
from ml.detection.common import (
    fit_scaler,
    transform_frame,
)
from ml.representation.manifold_geometry import (
    chronological_split,
)
from ml.representation.regime_geometry import (
    define_current_pressure_regimes,
    feature_scaling_diagnostics,
    fit_pca_95,
    neighbor_regime_purity,
    pca_feature_diagnostics,
    regime_conditioned_pca,
    regime_subspace_audit,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "regime_geometry_audit.yaml"
)


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
            "regime_geometry_audit.yaml must contain a mapping."
        )

    return payload


def _records(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    cleaned = frame.replace(
        {
            np.nan: None
        }
    )

    return cleaned.to_dict(
        orient="records"
    )


def main() -> None:
    config = _load_config()

    train_path = (
        ROOT
        / config[
            "dataset"
        ][
            "train_file"
        ]
    )

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

    regimes = config[
        "regimes"
    ]

    (
        regime_thresholds,
        fit_regime,
        audit_regime,
    ) = define_current_pressure_regimes(
        fit_frame=fit_frame,
        audit_frame=audit_frame,
        current_feature=str(
            regimes[
                "current_feature"
            ]
        ),
        pressure_feature=str(
            regimes[
                "pressure_feature"
            ]
        ),
        high_quantile=float(
            regimes[
                "high_quantile"
            ]
        ),
        label_map=dict(
            regimes[
                "labels"
            ]
        ),
    )

    q_low, q_high = [
        float(value)
        for value in config[
            "distribution_shift"
        ][
            "reference_quantiles"
        ]
    ]

    variance_target = float(
        config[
            "pca"
        ][
            "variance_target"
        ]
    )

    report_pc_loadings = int(
        config[
            "pca"
        ][
            "report_pc_loadings"
        ]
    )

    report_top = int(
        config[
            "pca"
        ][
            "report_top_features"
        ]
    )

    all_feature_tables = []
    all_regime_tables = []
    scaler_summaries: dict[
        str,
        Any,
    ] = {}

    for scaler_name in config[
        "scaling"
    ][
        "methods"
    ]:
        scaler_name = str(
            scaler_name
        )

        print(
            "Auditing scaler:",
            scaler_name,
        )

        scaler = fit_scaler(
            fit_frame,
            features,
            method=scaler_name,
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

        scaling_table = (
            feature_scaling_diagnostics(
                fit_frame=fit_frame,
                audit_frame=audit_frame,
                features=features,
                scaler=scaler,
                fit_scaled=x_fit,
                audit_scaled=x_audit,
                scaler_name=scaler_name,
                q_low=q_low,
                q_high=q_high,
            )
        )

        pca_fit = fit_pca_95(
            x_fit,
            variance_target=(
                variance_target
            ),
        )

        pca_table = (
            pca_feature_diagnostics(
                pca_fit=pca_fit,
                audit_scaled=x_audit,
                features=features,
                report_pc_loadings=(
                    report_pc_loadings
                ),
            )
        )

        feature_table = (
            scaling_table.merge(
                pca_table,
                on="feature",
                how="left",
                validate="one_to_one",
            )
        )

        feature_table[
            "scaler"
        ] = scaler_name

        all_feature_tables.append(
            feature_table
        )

        regime_table = (
            regime_conditioned_pca(
                fit_scaled=x_fit,
                audit_scaled=x_audit,
                fit_regime=fit_regime,
                audit_regime=audit_regime,
                global_fit=pca_fit,
                variance_target=(
                    variance_target
                ),
                minimum_fit_rows=int(
                    regimes[
                        "minimum_fit_rows"
                    ]
                ),
                minimum_audit_rows=int(
                    regimes[
                        "minimum_audit_rows"
                    ]
                ),
            )
        )

        regime_table[
            "scaler"
        ] = scaler_name

        all_regime_tables.append(
            regime_table
        )

        top_scaled_variance = (
            feature_table.sort_values(
                "fit_scaled_variance",
                ascending=False,
            )
            .head(
                report_top
            )
        )

        top_audit_residual = (
            feature_table.sort_values(
                "mean_audit_squared_residual",
                ascending=False,
            )
            .head(
                report_top
            )
        )

        top_shift = (
            feature_table.sort_values(
                "wasserstein_scaled",
                ascending=False,
            )
            .head(
                report_top
            )
        )

        top_loadings: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        for pc_index in range(
            report_pc_loadings
        ):
            column = (
                f"PC{pc_index + 1}_abs_loading"
            )

            if column not in (
                feature_table.columns
            ):
                continue

            top_loadings[
                f"PC{pc_index + 1}"
            ] = _records(
                feature_table.sort_values(
                    column,
                    ascending=False,
                )
                .loc[
                    :,
                    [
                        "feature",
                        column,
                    ],
                ]
                .head(
                    report_top
                )
            )

        scaler_summaries[
            scaler_name
        ] = {
            "global_pca": {
                "component_count_95": int(
                    pca_fit.component_count_95
                ),
                "participation_ratio": float(
                    pca_fit.participation_ratio
                ),
            },
            "top_fit_scaled_variance": (
                _records(
                    top_scaled_variance.loc[
                        :,
                        [
                            "feature",
                            "fit_iqr",
                            "fit_std",
                            "scaler_scale",
                            "fit_scaled_variance",
                            "fit_max_abs_scaled",
                        ],
                    ]
                )
            ),
            "top_chronological_shift": (
                _records(
                    top_shift.loc[
                        :,
                        [
                            "feature",
                            "wasserstein_scaled",
                            "ks_statistic",
                            "audit_outside_fit_p01_p99_fraction",
                        ],
                    ]
                )
            ),
            "top_audit_residual_contributors": (
                _records(
                    top_audit_residual.loc[
                        :,
                        [
                            "feature",
                            "mean_audit_squared_residual",
                            "median_audit_squared_residual",
                        ],
                    ]
                )
            ),
            "top_pc_loadings": (
                top_loadings
            ),
            "regime_conditioned_pca": (
                _records(
                    regime_table
                )
            ),
            "neighbor_regime_purity": (
                neighbor_regime_purity(
                    fit_scaled=x_fit,
                    audit_scaled=x_audit,
                    fit_regime=fit_regime,
                    audit_regime=audit_regime,
                    k=int(
                        config[
                            "neighbors"
                        ][
                            "k"
                        ]
                    ),
                )
            ),
            "subspace": (
                regime_subspace_audit(
                    fit_scaled=x_fit,
                    fit_regime=fit_regime,
                    component_count=(
                        pca_fit.component_count_95
                    ),
                    minimum_half_rows=int(
                        config[
                            "subspace"
                        ][
                            "minimum_half_rows"
                        ]
                    ),
                )
            ),
        }

    feature_output = pd.concat(
        all_feature_tables,
        ignore_index=True,
    )

    regime_output = pd.concat(
        all_regime_tables,
        ignore_index=True,
    )

    summary = {
        "schema_version": (
            "aeroxai.scaling_regime_geometry.v1"
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
        "prior_result_informed": True,
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
        "internal_split": {
            "fit_rows": len(
                    fit_frame
                ),
            "audit_rows": len(
                    audit_frame
                ),
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
        "regime_definition": {
            **regime_thresholds,
            "fit_counts": (
                fit_regime
                .value_counts()
                .sort_index()
                .to_dict()
            ),
            "audit_counts": (
                audit_regime
                .value_counts()
                .sort_index()
                .to_dict()
            ),
        },
        "scalers": scaler_summaries,
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

    feature_path = (
        ROOT
        / outputs[
            "feature_table"
        ]
    )

    regime_path = (
        ROOT
        / outputs[
            "regime_table"
        ]
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    regime_path.parent.mkdir(
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

    feature_output.to_parquet(
        feature_path,
        index=False,
    )

    regime_output.to_parquet(
        regime_path,
        index=False,
    )

    print(
        json.dumps(
            {
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
                "regime_definition": (
                    summary[
                        "regime_definition"
                    ]
                ),
                "scaler_global_pca": {
                    key: value[
                        "global_pca"
                    ]
                    for key, value
                    in scaler_summaries.items()
                },
                "summary_json": str(
                    summary_path
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
