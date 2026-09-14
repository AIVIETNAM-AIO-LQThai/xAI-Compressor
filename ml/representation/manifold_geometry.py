from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer


@dataclass(frozen=True)
class GlobalPCAResult:
    model: PCA
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    participation_ratio: float
    components_for_targets: dict[str, int]


def chronological_split(
    frame: pd.DataFrame,
    *,
    fit_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("fit_fraction must be in (0, 1).")

    ordered = frame.sort_index()
    cut = int(np.floor(len(ordered) * fit_fraction))

    if cut <= 0 or cut >= len(ordered):
        raise ValueError("Chronological split produced an empty subset.")

    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def fit_global_pca(
    x_fit: np.ndarray,
    *,
    variance_targets: list[float],
) -> GlobalPCAResult:
    values = np.asarray(x_fit, dtype=np.float64)

    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("x_fit must be a non-trivial 2D matrix.")

    model = PCA(svd_solver="full")
    model.fit(values)

    eigenvalues = model.explained_variance_.astype(np.float64, copy=True)
    ratios = model.explained_variance_ratio_.astype(np.float64, copy=True)
    cumulative = np.cumsum(ratios)

    total = float(eigenvalues.sum())
    squared_total = float(np.square(eigenvalues).sum())
    participation_ratio = float(total * total / squared_total)

    target_components: dict[str, int] = {}

    for target in variance_targets:
        if not 0.0 < target <= 1.0:
            raise ValueError("Variance targets must be in (0, 1].")

        target_components[f"{target:.4f}"] = int(
            np.searchsorted(cumulative, target, side="left") + 1
        )

    return GlobalPCAResult(
        model=model,
        eigenvalues=eigenvalues,
        explained_variance_ratio=ratios,
        cumulative_variance=cumulative,
        participation_ratio=participation_ratio,
        components_for_targets=target_components,
    )


def two_nearest_neighbor_dimension(
    x_fit: np.ndarray,
) -> dict[str, float | int]:
    values = np.asarray(x_fit, dtype=np.float64)

    neighbors = NearestNeighbors(
        n_neighbors=3,
        metric="euclidean",
    )
    neighbors.fit(values)
    distances, _ = neighbors.kneighbors(values)

    r1 = distances[:, 1]
    r2 = distances[:, 2]

    valid = (
        np.isfinite(r1)
        & np.isfinite(r2)
        & (r1 > 0.0)
        & (r2 > r1)
    )

    mu = r2[valid] / r1[valid]
    log_mu = np.log(mu)

    if (
        log_mu.size == 0
        or not np.isfinite(log_mu).all()
        or float(log_mu.mean()) <= 0.0
    ):
        raise ValueError("TwoNN intrinsic dimension is undefined.")

    return {
        "estimate": float(1.0 / log_mu.mean()),
        "sample_count": int(values.shape[0]),
        "valid_count": int(valid.sum()),
        "excluded_count": int((~valid).sum()),
    }


def reconstruct_with_pca(
    model: PCA,
    values: np.ndarray,
    *,
    component_count: int,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)

    if not 1 <= component_count <= model.components_.shape[0]:
        raise ValueError("component_count is outside fitted PCA range.")

    components = model.components_[:component_count]
    centered = x - model.mean_
    encoded = centered @ components.T

    return encoded @ components + model.mean_


def reconstruction_mse(
    observed: np.ndarray,
    reconstructed: np.ndarray,
) -> np.ndarray:
    x = np.asarray(observed, dtype=np.float64)
    x_hat = np.asarray(reconstructed, dtype=np.float64)

    if x.shape != x_hat.shape:
        raise ValueError("Observed and reconstructed shapes differ.")

    return np.mean(np.square(x - x_hat), axis=1)


def _subspace_summary(
    reference_basis: np.ndarray,
    comparison_basis: np.ndarray,
) -> tuple[float, float]:
    angles = subspace_angles(
        reference_basis.T,
        comparison_basis.T,
    )
    degrees = np.degrees(angles)
    rms_sine = float(
        np.sqrt(np.mean(np.square(np.sin(angles))))
    )

    return float(degrees.max()), rms_sine


def local_pca_audit(
    x_reference: np.ndarray,
    x_audit: np.ndarray,
    *,
    global_model: PCA,
    latent_dimension: int,
    neighbor_counts: list[int],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reference = np.asarray(x_reference, dtype=np.float64)
    audit = np.asarray(x_audit, dtype=np.float64)

    if latent_dimension < 1:
        raise ValueError("latent_dimension must be positive.")

    max_neighbors = max(neighbor_counts)

    if max_neighbors > len(reference):
        raise ValueError("Neighborhood larger than reference set.")

    neighbor_model = NearestNeighbors(
        n_neighbors=max_neighbors,
        metric="euclidean",
    )
    neighbor_model.fit(reference)
    _, neighbor_indices = neighbor_model.kneighbors(audit)

    global_reconstruction = reconstruct_with_pca(
        global_model,
        audit,
        component_count=latent_dimension,
    )
    global_error = reconstruction_mse(
        audit,
        global_reconstruction,
    )
    global_basis = global_model.components_[:latent_dimension]

    point_columns: dict[str, np.ndarray] = {
        "global_pca_mse": global_error,
    }
    summary: dict[str, Any] = {}

    for neighbor_count in neighbor_counts:
        local_error = np.empty(len(audit), dtype=np.float64)
        max_angle = np.empty(len(audit), dtype=np.float64)
        rms_sine = np.empty(len(audit), dtype=np.float64)
        local_bases: list[np.ndarray] = []

        for position in range(len(audit)):
            indices = neighbor_indices[
                position,
                :neighbor_count,
            ]
            neighborhood = reference[indices]

            local_model = PCA(
                n_components=latent_dimension,
                svd_solver="full",
            )
            local_model.fit(neighborhood)

            observed = audit[position : position + 1]
            reconstructed = local_model.inverse_transform(
                local_model.transform(observed)
            )

            local_error[position] = float(
                reconstruction_mse(
                    observed,
                    reconstructed,
                )[0]
            )

            basis = local_model.components_
            local_bases.append(basis)

            (
                max_angle[position],
                rms_sine[position],
            ) = _subspace_summary(
                global_basis,
                basis,
            )

        consecutive_max_angle = np.full(
            len(audit),
            np.nan,
            dtype=np.float64,
        )
        consecutive_rms_sine = np.full(
            len(audit),
            np.nan,
            dtype=np.float64,
        )

        for position in range(1, len(local_bases)):
            (
                consecutive_max_angle[position],
                consecutive_rms_sine[position],
            ) = _subspace_summary(
                local_bases[position - 1],
                local_bases[position],
            )

        ratio = np.divide(
            local_error,
            global_error,
            out=np.full_like(local_error, np.nan),
            where=(global_error > 0.0),
        )

        key = str(neighbor_count)

        point_columns[f"local_pca_mse_k{key}"] = local_error
        point_columns[f"local_global_ratio_k{key}"] = ratio
        point_columns[
            f"local_global_max_angle_deg_k{key}"
        ] = max_angle
        point_columns[
            f"local_global_rms_sine_k{key}"
        ] = rms_sine
        point_columns[
            f"consecutive_local_max_angle_deg_k{key}"
        ] = consecutive_max_angle
        point_columns[
            f"consecutive_local_rms_sine_k{key}"
        ] = consecutive_rms_sine

        finite_ratio = ratio[np.isfinite(ratio)]

        summary[key] = {
            "neighbor_count": int(neighbor_count),
            "median_global_mse": float(
                np.median(global_error)
            ),
            "median_local_mse": float(
                np.median(local_error)
            ),
            "median_local_global_ratio": float(
                np.median(finite_ratio)
            ),
            "fraction_local_better": float(
                np.mean(local_error < global_error)
            ),
            "median_local_global_max_angle_deg": float(
                np.median(max_angle)
            ),
            "median_local_global_rms_sine": float(
                np.median(rms_sine)
            ),
            "median_consecutive_local_max_angle_deg": float(
                np.nanmedian(consecutive_max_angle)
            ),
            "median_consecutive_local_rms_sine": float(
                np.nanmedian(consecutive_rms_sine)
            ),
        }

    return pd.DataFrame(point_columns), summary


def _spearman(
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, float | None]:
    result = spearmanr(
        x,
        y,
        nan_policy="omit",
    )
    statistic = float(result.statistic)
    pvalue = float(result.pvalue)

    return {
        "rho": statistic if np.isfinite(statistic) else None,
        "pvalue": pvalue if np.isfinite(pvalue) else None,
    }


def residual_structure_audit(
    *,
    global_error: np.ndarray,
    pc_scores: np.ndarray,
    audit_frame: pd.DataFrame,
    regime_features: list[str],
    quantile_bins: int,
) -> dict[str, Any]:
    errors = np.asarray(global_error, dtype=np.float64)

    result: dict[str, Any] = {
        "pc_correlations": {},
        "regime_features": {},
    }

    for index in range(min(3, pc_scores.shape[1])):
        result["pc_correlations"][
            f"PC{index + 1}"
        ] = _spearman(
            pc_scores[:, index],
            errors,
        )

    for feature in regime_features:
        if feature not in audit_frame.columns:
            result["regime_features"][feature] = {
                "available": False,
            }
            continue

        values = audit_frame[
            feature
        ].to_numpy(dtype=np.float64)

        quantiles = pd.qcut(
            audit_frame[feature],
            q=quantile_bins,
            duplicates="drop",
        )

        grouped = pd.DataFrame(
            {
                "bin": quantiles,
                "error": errors,
            },
            index=audit_frame.index,
        ).groupby(
            "bin",
            observed=True,
        )["error"]

        medians = grouped.median()
        positive = medians[medians > 0.0]

        ratio = (
            None
            if positive.empty
            else float(
                positive.max()
                / positive.min()
            )
        )

        result["regime_features"][feature] = {
            "available": True,
            "spearman": _spearman(
                values,
                errors,
            ),
            "quantile_bin_median_mse": [
                {
                    "bin": str(interval),
                    "median_mse": float(value),
                }
                for interval, value
                in medians.items()
            ],
            "max_min_positive_median_ratio": ratio,
        }

    return result


def relationship_shape_audit(
    fit_frame: pd.DataFrame,
    audit_frame: pd.DataFrame,
    *,
    pairs: list[dict[str, str]],
    spline_knots: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    model_builders = {
        "linear": lambda: LinearRegression(),
        "quadratic": lambda: make_pipeline(
            PolynomialFeatures(
                degree=2,
                include_bias=False,
            ),
            LinearRegression(),
        ),
        "cubic_spline": lambda: make_pipeline(
            SplineTransformer(
                n_knots=spline_knots,
                degree=3,
                include_bias=False,
            ),
            LinearRegression(),
        ),
    }

    for pair in pairs:
        x_name = pair["x"]
        y_name = pair["y"]

        available = (
            x_name in fit_frame.columns
            and y_name in fit_frame.columns
            and x_name in audit_frame.columns
            and y_name in audit_frame.columns
        )

        if not available:
            rows.append(
                {
                    "x": x_name,
                    "y": y_name,
                    "model": None,
                    "available": False,
                    "mse": np.nan,
                    "r2": np.nan,
                }
            )
            continue

        x_fit = fit_frame[
            [x_name]
        ].to_numpy(dtype=np.float64)
        y_fit = fit_frame[
            y_name
        ].to_numpy(dtype=np.float64)
        x_audit = audit_frame[
            [x_name]
        ].to_numpy(dtype=np.float64)
        y_audit = audit_frame[
            y_name
        ].to_numpy(dtype=np.float64)

        for model_name, builder in model_builders.items():
            model = builder()
            model.fit(x_fit, y_fit)
            prediction = model.predict(x_audit)

            rows.append(
                {
                    "x": x_name,
                    "y": y_name,
                    "model": model_name,
                    "available": True,
                    "mse": float(
                        mean_squared_error(
                            y_audit,
                            prediction,
                        )
                    ),
                    "r2": float(
                        r2_score(
                            y_audit,
                            prediction,
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)
