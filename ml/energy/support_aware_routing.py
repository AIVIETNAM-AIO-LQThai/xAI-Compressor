from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.detection.common import transform_frame
from ml.detection.pca_detector import PCADetector


@dataclass(frozen=True)
class SupportPolicy:
    mid_support_percentile: float
    high_support_percentile: float

    def __post_init__(self) -> None:
        if not (
            0.0
            <= self.mid_support_percentile
            < self.high_support_percentile
            <= 1.0
        ):
            raise ValueError(
                "Support policy requires "
                "0 <= mid < high <= 1."
            )


def empirical_percentile(
    reference: np.ndarray | pd.Series,
    values: np.ndarray | pd.Series,
) -> np.ndarray:
    reference_array = np.asarray(
        reference,
        dtype=float,
    ).reshape(-1)
    value_array = np.asarray(
        values,
        dtype=float,
    )

    if reference_array.size == 0:
        raise ValueError(
            "Empirical-percentile reference "
            "cannot be empty."
        )

    if not np.isfinite(reference_array).all():
        raise ValueError(
            "Reference contains non-finite values."
        )
    if not np.isfinite(value_array).all():
        raise ValueError(
            "Values contain non-finite values."
        )

    ordered = np.sort(reference_array)

    # Right-inclusive empirical CDF:
    # P_CAL(S <= s) = count(S_CAL <= s) / N_CAL.
    ranks = np.searchsorted(
        ordered,
        value_array,
        side="right",
    )

    return ranks.astype(float) / ordered.size


def feature_support_statistic(
    frame: pd.DataFrame,
    *,
    features: list[str],
    scaler: StandardScaler,
    quantile: float = 0.95,
) -> pd.Series:
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(
            "quantile must be in [0, 1]."
        )

    scaled = transform_frame(
        frame,
        features,
        scaler,
    )

    values = np.quantile(
        np.abs(scaled),
        quantile,
        axis=1,
        method="linear",
    )

    return pd.Series(
        values,
        index=frame.index,
        name="feature_support_raw",
        dtype=float,
    )


def pca_latent_support_statistic(
    frame: pd.DataFrame,
    *,
    detector: PCADetector,
) -> pd.Series:
    scaled = transform_frame(
        frame,
        detector.features,
        detector.scaler,
    )
    latent = detector.model.transform(
        scaled
    )

    variance = np.asarray(
        detector.model.explained_variance_,
        dtype=float,
    )

    if variance.ndim != 1:
        raise ValueError(
            "PCA explained variance must be 1-D."
        )
    if np.any(variance <= 0.0):
        raise ValueError(
            "PCA latent support requires strictly "
            "positive retained-component variance."
        )

    values = np.sqrt(
        np.sum(
            np.square(latent)
            / variance.reshape(1, -1),
            axis=1,
        )
    )

    return pd.Series(
        values,
        index=frame.index,
        name="pca_latent_support_raw",
        dtype=float,
    )


def calibration_support_percentiles(
    *,
    feature_support: pd.Series,
    latent_support: pd.Series,
) -> pd.DataFrame:
    if not feature_support.index.equals(
        latent_support.index
    ):
        raise ValueError(
            "Support components must have "
            "identical indexes."
        )

    feature_percentile = (
        empirical_percentile(
            feature_support.to_numpy(),
            feature_support.to_numpy(),
        )
    )
    latent_percentile = (
        empirical_percentile(
            latent_support.to_numpy(),
            latent_support.to_numpy(),
        )
    )

    combined = np.maximum(
        feature_percentile,
        latent_percentile,
    )

    return pd.DataFrame(
        {
            "feature_support_percentile": (
                feature_percentile
            ),
            "latent_support_percentile": (
                latent_percentile
            ),
            "combined_support_percentile": (
                combined
            ),
        },
        index=feature_support.index,
    )


def support_percentiles_from_reference(
    *,
    feature_reference: pd.Series,
    latent_reference: pd.Series,
    feature_values: pd.Series,
    latent_values: pd.Series,
) -> pd.DataFrame:
    if not feature_values.index.equals(
        latent_values.index
    ):
        raise ValueError(
            "Support value components must have "
            "identical indexes."
        )

    feature_percentile = (
        empirical_percentile(
            feature_reference.to_numpy(),
            feature_values.to_numpy(),
        )
    )
    latent_percentile = (
        empirical_percentile(
            latent_reference.to_numpy(),
            latent_values.to_numpy(),
        )
    )

    return pd.DataFrame(
        {
            "feature_support_percentile": (
                feature_percentile
            ),
            "latent_support_percentile": (
                latent_percentile
            ),
            "combined_support_percentile": (
                np.maximum(
                    feature_percentile,
                    latent_percentile,
                )
            ),
        },
        index=feature_values.index,
    )


def support_aware_thresholds(
    support_percentile: pd.Series,
    *,
    policy: SupportPolicy,
    router_thresholds: dict[str, float],
) -> pd.Series:
    required = {
        "route_q90",
        "route_q95",
        "route_q99",
    }

    if set(router_thresholds) != required:
        raise ValueError(
            "router_thresholds must contain "
            "exactly route_q90/q95/q99."
        )

    values = support_percentile.astype(float)
    if (
        (values < 0.0).any()
        or (values > 1.0).any()
    ):
        raise ValueError(
            "Support percentiles must lie "
            "within [0, 1]."
        )

    selected = np.full(
        len(values),
        float(router_thresholds["route_q99"]),
        dtype=float,
    )

    mid = (
        values.to_numpy()
        >= policy.mid_support_percentile
    )
    high = (
        values.to_numpy()
        >= policy.high_support_percentile
    )

    selected[mid] = float(
        router_thresholds["route_q95"]
    )
    selected[high] = float(
        router_thresholds["route_q90"]
    )

    return pd.Series(
        selected,
        index=values.index,
        name="selected_pca_threshold",
        dtype=float,
    )


def support_aware_route(
    pca_scores: pd.Series,
    support_percentile: pd.Series,
    *,
    policy: SupportPolicy,
    router_thresholds: dict[str, float],
) -> pd.Series:
    if not pca_scores.index.equals(
        support_percentile.index
    ):
        raise ValueError(
            "PCA and support scores must have "
            "identical indexes."
        )

    thresholds = support_aware_thresholds(
        support_percentile,
        policy=policy,
        router_thresholds=router_thresholds,
    )

    return (
        pca_scores.astype(float)
        >= thresholds
    ).rename("route_to_tcn")


def fixed_route(
    pca_scores: pd.Series,
    *,
    threshold: float,
) -> pd.Series:
    return (
        pca_scores.astype(float)
        >= float(threshold)
    ).rename("route_to_tcn")


def evidence_coverage(
    route: pd.Series,
    evidence: pd.Series,
) -> float | None:
    evidence_bool = evidence.reindex(
        route.index,
        fill_value=False,
    ).astype(bool)

    denominator = int(
        evidence_bool.sum()
    )

    if denominator == 0:
        return None

    numerator = int(
        (
            route.astype(bool)
            & evidence_bool
        ).sum()
    )

    return numerator / denominator


def candidate_grid(
    *,
    mid_values: list[float],
    high_values: list[float],
) -> list[SupportPolicy]:
    policies = [
        SupportPolicy(
            mid_support_percentile=float(mid),
            high_support_percentile=float(high),
        )
        for mid, high in product(
            mid_values,
            high_values,
        )
        if float(mid) < float(high)
    ]

    if not policies:
        raise ValueError(
            "Candidate grid is empty."
        )

    return policies


def select_policy(
    candidate_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    eligible = [
        report
        for report in candidate_reports
        if bool(report["eligible"])
    ]

    if not eligible:
        raise RuntimeError(
            "No support-aware candidate satisfies "
            "the preregistered development "
            "evidence constraints."
        )

    def key(
        report: dict[str, Any],
    ) -> tuple[float, float, float, float]:
        return (
            float(
                report[
                    "mean_tcn_invocation_fraction"
                ]
            ),
            -float(
                report[
                    "worst_dataset_alert_coverage"
                ]
            ),
            -float(
                report[
                    "high_support_percentile"
                ]
            ),
            -float(
                report[
                    "mid_support_percentile"
                ]
            ),
        )

    return min(
        eligible,
        key=key,
    )
