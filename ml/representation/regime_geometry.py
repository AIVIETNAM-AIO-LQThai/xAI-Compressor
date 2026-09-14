from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from ml.detection.common import Scaler


@dataclass(frozen=True)
class PCAFit:
    model: PCA
    component_count_95: int
    participation_ratio: float


def fit_pca_95(
    values: np.ndarray,
    *,
    variance_target: float,
) -> PCAFit:
    x = np.asarray(values, dtype=float)

    model = PCA(svd_solver="full")
    model.fit(x)

    cumulative = np.cumsum(
        model.explained_variance_ratio_
    )

    component_count = int(
        np.searchsorted(
            cumulative,
            variance_target,
            side="left",
        )
        + 1
    )

    eigenvalues = model.explained_variance_
    participation_ratio = float(
        eigenvalues.sum() ** 2
        / np.square(eigenvalues).sum()
    )

    return PCAFit(
        model=model,
        component_count_95=component_count,
        participation_ratio=participation_ratio,
    )


def reconstruct_fixed_dimension(
    model: PCA,
    values: np.ndarray,
    *,
    component_count: int,
) -> np.ndarray:
    x = np.asarray(values, dtype=float)

    components = model.components_[
        :component_count
    ]

    centered = x - model.mean_

    return (
        (centered @ components.T)
        @ components
        + model.mean_
    )


def row_mse(
    observed: np.ndarray,
    reconstructed: np.ndarray,
) -> np.ndarray:
    return np.mean(
        np.square(
            np.asarray(observed, dtype=float)
            - np.asarray(reconstructed, dtype=float)
        ),
        axis=1,
    )


def feature_scaling_diagnostics(
    *,
    fit_frame: pd.DataFrame,
    audit_frame: pd.DataFrame,
    features: list[str],
    scaler: Scaler,
    fit_scaled: np.ndarray,
    audit_scaled: np.ndarray,
    scaler_name: str,
    q_low: float,
    q_high: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    scale = np.asarray(
        scaler.scale_,
        dtype=float,
    )

    for index, feature in enumerate(features):
        fit_raw = fit_frame[
            feature
        ].to_numpy(dtype=float)

        audit_raw = audit_frame[
            feature
        ].to_numpy(dtype=float)

        fit_values = fit_scaled[
            :,
            index,
        ]

        audit_values = audit_scaled[
            :,
            index,
        ]

        low = float(
            np.quantile(
                fit_raw,
                q_low,
            )
        )

        high = float(
            np.quantile(
                fit_raw,
                q_high,
            )
        )

        outside = (
            (audit_raw < low)
            | (audit_raw > high)
        )

        q25, q75 = np.quantile(
            fit_raw,
            [
                0.25,
                0.75,
            ],
        )

        rows.append(
            {
                "scaler": scaler_name,
                "feature": feature,
                "fit_median": float(
                    np.median(fit_raw)
                ),
                "fit_iqr": float(
                    q75 - q25
                ),
                "fit_std": float(
                    np.std(
                        fit_raw,
                        ddof=0,
                    )
                ),
                "scaler_scale": float(
                    scale[index]
                ),
                "fit_scaled_variance": float(
                    np.var(
                        fit_values,
                        ddof=0,
                    )
                ),
                "audit_scaled_variance": float(
                    np.var(
                        audit_values,
                        ddof=0,
                    )
                ),
                "fit_max_abs_scaled": float(
                    np.max(
                        np.abs(
                            fit_values
                        )
                    )
                ),
                "audit_max_abs_scaled": float(
                    np.max(
                        np.abs(
                            audit_values
                        )
                    )
                ),
                "wasserstein_scaled": float(
                    wasserstein_distance(
                        fit_values,
                        audit_values,
                    )
                ),
                "ks_statistic": float(
                    ks_2samp(
                        fit_values,
                        audit_values,
                        method="auto",
                    ).statistic
                ),
                "audit_outside_fit_p01_p99_fraction": float(
                    np.mean(outside)
                ),
            }
        )

    return pd.DataFrame(rows)


def pca_feature_diagnostics(
    *,
    pca_fit: PCAFit,
    audit_scaled: np.ndarray,
    features: list[str],
    report_pc_loadings: int,
) -> pd.DataFrame:
    reconstructed = reconstruct_fixed_dimension(
        pca_fit.model,
        audit_scaled,
        component_count=(
            pca_fit.component_count_95
        ),
    )

    residual_squared = np.square(
        audit_scaled
        - reconstructed
    )

    rows: list[dict[str, Any]] = []

    for index, feature in enumerate(features):
        row: dict[str, Any] = {
            "feature": feature,
            "median_audit_squared_residual": float(
                np.median(
                    residual_squared[
                        :,
                        index,
                    ]
                )
            ),
            "mean_audit_squared_residual": float(
                np.mean(
                    residual_squared[
                        :,
                        index,
                    ]
                )
            ),
        }

        for pc_index in range(
            min(
                report_pc_loadings,
                pca_fit.model.components_.shape[0],
            )
        ):
            row[
                f"PC{pc_index + 1}_abs_loading"
            ] = float(
                abs(
                    pca_fit.model.components_[
                        pc_index,
                        index,
                    ]
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


def define_current_pressure_regimes(
    *,
    fit_frame: pd.DataFrame,
    audit_frame: pd.DataFrame,
    current_feature: str,
    pressure_feature: str,
    high_quantile: float,
    label_map: dict[str, str],
) -> tuple[dict[str, float], pd.Series, pd.Series]:
    current_threshold = float(
        fit_frame[
            current_feature
        ].quantile(
            high_quantile
        )
    )

    pressure_threshold = float(
        fit_frame[
            pressure_feature
        ].quantile(
            high_quantile
        )
    )

    def classify(
        frame: pd.DataFrame,
    ) -> pd.Series:
        high_current = (
            frame[
                current_feature
            ]
            >= current_threshold
        )

        high_pressure = (
            frame[
                pressure_feature
            ]
            >= pressure_threshold
        )

        values = np.full(
            len(frame),
            label_map[
                "low_low"
            ],
            dtype=object,
        )

        values[
            high_current
            & ~high_pressure
        ] = label_map[
            "high_current"
        ]

        values[
            ~high_current
            & high_pressure
        ] = label_map[
            "high_pressure"
        ]

        values[
            high_current
            & high_pressure
        ] = label_map[
            "high_both"
        ]

        return pd.Series(
            values,
            index=frame.index,
            name="operating_regime",
        )

    thresholds = {
        "current_threshold": (
            current_threshold
        ),
        "pressure_threshold": (
            pressure_threshold
        ),
        "quantile": float(
            high_quantile
        ),
    }

    return (
        thresholds,
        classify(fit_frame),
        classify(audit_frame),
    )


def regime_conditioned_pca(
    *,
    fit_scaled: np.ndarray,
    audit_scaled: np.ndarray,
    fit_regime: pd.Series,
    audit_regime: pd.Series,
    global_fit: PCAFit,
    variance_target: float,
    minimum_fit_rows: int,
    minimum_audit_rows: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    global_reconstructed = (
        reconstruct_fixed_dimension(
            global_fit.model,
            audit_scaled,
            component_count=(
                global_fit.component_count_95
            ),
        )
    )

    global_error = row_mse(
        audit_scaled,
        global_reconstructed,
    )

    fit_labels = (
        fit_regime.to_numpy()
    )

    audit_labels = (
        audit_regime.to_numpy()
    )

    for regime in sorted(
        set(fit_labels)
        | set(audit_labels)
    ):
        fit_mask = (
            fit_labels == regime
        )

        audit_mask = (
            audit_labels == regime
        )

        fit_rows = int(
            fit_mask.sum()
        )

        audit_rows = int(
            audit_mask.sum()
        )

        eligible = (
            fit_rows
            >= minimum_fit_rows
            and audit_rows
            >= minimum_audit_rows
        )

        row: dict[str, Any] = {
            "regime": regime,
            "fit_rows": fit_rows,
            "audit_rows": audit_rows,
            "eligible": eligible,
            "global_component_count": int(
                global_fit.component_count_95
            ),
        }

        if not eligible:
            rows.append(row)
            continue

        regime_model = PCA(
            svd_solver="full"
        ).fit(
            fit_scaled[
                fit_mask
            ]
        )

        cumulative = np.cumsum(
            regime_model
            .explained_variance_ratio_
        )

        own_dim = int(
            np.searchsorted(
                cumulative,
                variance_target,
                side="left",
            )
            + 1
        )

        fixed_dim = min(
            global_fit.component_count_95,
            regime_model.components_.shape[0],
        )

        regime_reconstructed = (
            reconstruct_fixed_dimension(
                regime_model,
                audit_scaled[
                    audit_mask
                ],
                component_count=fixed_dim,
            )
        )

        regime_error = row_mse(
            audit_scaled[
                audit_mask
            ],
            regime_reconstructed,
        )

        global_regime_error = (
            global_error[
                audit_mask
            ]
        )

        ratio = np.divide(
            regime_error,
            global_regime_error,
            out=np.full_like(
                regime_error,
                np.nan,
            ),
            where=(
                global_regime_error > 0
            ),
        )

        row.update(
            {
                "regime_own_component_count_95": (
                    own_dim
                ),
                "median_global_mse": float(
                    np.median(
                        global_regime_error
                    )
                ),
                "median_regime_mse": float(
                    np.median(
                        regime_error
                    )
                ),
                "median_regime_global_ratio": float(
                    np.nanmedian(
                        ratio
                    )
                ),
                "fraction_regime_better": float(
                    np.mean(
                        regime_error
                        < global_regime_error
                    )
                ),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def neighbor_regime_purity(
    *,
    fit_scaled: np.ndarray,
    audit_scaled: np.ndarray,
    fit_regime: pd.Series,
    audit_regime: pd.Series,
    k: int,
) -> dict[str, float | int]:
    model = NearestNeighbors(
        n_neighbors=k,
        metric="euclidean",
    )

    model.fit(
        fit_scaled
    )

    _, indices = model.kneighbors(
        audit_scaled
    )

    fit_labels = (
        fit_regime.to_numpy()
    )

    audit_labels = (
        audit_regime.to_numpy()
    )

    purity = np.mean(
        fit_labels[
            indices
        ]
        == audit_labels[
            :,
            None,
        ],
        axis=1,
    )

    return {
        "k": int(k),
        "mean_same_regime_fraction": float(
            np.mean(purity)
        ),
        "median_same_regime_fraction": float(
            np.median(purity)
        ),
        "fraction_at_least_80pct_same_regime": float(
            np.mean(
                purity >= 0.80
            )
        ),
    }


def _max_principal_angle_deg(
    basis_a: np.ndarray,
    basis_b: np.ndarray,
) -> float:
    angles = subspace_angles(
        basis_a.T,
        basis_b.T,
    )

    return float(
        np.degrees(
            angles
        ).max()
    )


def regime_subspace_audit(
    *,
    fit_scaled: np.ndarray,
    fit_regime: pd.Series,
    component_count: int,
    minimum_half_rows: int,
) -> dict[str, Any]:
    labels = fit_regime.to_numpy()
    full_bases: dict[
        str,
        np.ndarray,
    ] = {}

    within_rows: list[
        dict[str, Any]
    ] = []

    for regime in sorted(
        set(labels)
    ):
        mask = (
            labels == regime
        )

        values = fit_scaled[
            mask
        ]

        if (
            len(values)
            < max(
                component_count + 1,
                2 * minimum_half_rows,
            )
        ):
            within_rows.append(
                {
                    "regime": regime,
                    "eligible": False,
                    "rows": len(values),
                }
            )
            continue

        full_model = PCA(
            n_components=component_count,
            svd_solver="full",
        ).fit(values)

        full_bases[
            regime
        ] = (
            full_model.components_
        )

        cut = (
            len(values)
            // 2
        )

        early = values[
            :cut
        ]

        late = values[
            cut:
        ]

        early_model = PCA(
            n_components=component_count,
            svd_solver="full",
        ).fit(early)

        late_model = PCA(
            n_components=component_count,
            svd_solver="full",
        ).fit(late)

        within_rows.append(
            {
                "regime": regime,
                "eligible": True,
                "rows": len(values),
                "early_rows": len(early),
                "late_rows": len(late),
                "early_late_max_principal_angle_deg": (
                    _max_principal_angle_deg(
                        early_model.components_,
                        late_model.components_,
                    )
                ),
            }
        )

    between_rows: list[
        dict[str, Any]
    ] = []

    for regime_a, regime_b in combinations(
        sorted(
            full_bases
        ),
        2,
    ):
        between_rows.append(
            {
                "regime_a": regime_a,
                "regime_b": regime_b,
                "max_principal_angle_deg": (
                    _max_principal_angle_deg(
                        full_bases[
                            regime_a
                        ],
                        full_bases[
                            regime_b
                        ],
                    )
                ),
            }
        )

    return {
        "within_regime": within_rows,
        "between_regime": between_rows,
    }
