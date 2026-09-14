from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml.energy.support_aware_routing import (
    empirical_percentile,
)

ROUTER_FEATURE_NAMES = [
    "pca_ewma_percentile",
    "pca_ewma_delta_robust_z",
    "pca_ewma_volatility_percentile",
    "feature_support_percentile",
    "pca_latent_support_percentile",
    "combined_support_percentile",
    "recent_q90_hit_fraction",
    "recent_q95_hit_fraction",
]


@dataclass(frozen=True)
class ChronologicalSplit:
    fit_index: pd.DatetimeIndex
    validation_index: pd.DatetimeIndex


@dataclass(frozen=True)
class LogisticCandidate:
    C: float
    class_weight: str | None
    probability_threshold: float


def chronological_split(
    index: pd.DatetimeIndex,
    *,
    fit_fraction: float,
) -> ChronologicalSplit:
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError(
            "fit_fraction must be in (0, 1)."
        )

    ordered = pd.DatetimeIndex(index)

    if not ordered.is_monotonic_increasing:
        raise ValueError(
            "Chronological split requires a "
            "monotonic increasing index."
        )

    split = int(
        np.floor(
            fit_fraction * len(ordered)
        )
    )

    if split <= 0 or split >= len(ordered):
        raise ValueError(
            "Chronological split produced an "
            "empty fit or validation subset."
        )

    return ChronologicalSplit(
        fit_index=ordered[:split],
        validation_index=ordered[split:],
    )


def robust_center_scale(
    fit_values: pd.Series,
) -> tuple[float, float]:
    values = fit_values.astype(float)

    if values.empty:
        raise ValueError(
            "fit_values cannot be empty."
        )

    median = float(
        values.median()
    )
    q25 = float(
        values.quantile(
            0.25,
            interpolation="linear",
        )
    )
    q75 = float(
        values.quantile(
            0.75,
            interpolation="linear",
        )
    )

    scale = q75 - q25

    if not np.isfinite(scale):
        raise ValueError(
            "Robust scale is non-finite."
        )

    if scale == 0.0:
        scale = 1.0

    return median, scale


def build_router_features(
    *,
    pca_ewma: pd.Series,
    feature_support_raw: pd.Series,
    latent_support_raw: pd.Series,
    target_index: pd.DatetimeIndex,
    fit_index: pd.DatetimeIndex,
    rolling_history_bins: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if rolling_history_bins <= 1:
        raise ValueError(
            "rolling_history_bins must be > 1."
        )

    target_index = pd.DatetimeIndex(
        target_index
    )
    fit_index = pd.DatetimeIndex(
        fit_index
    )

    if not target_index.is_monotonic_increasing:
        raise ValueError(
            "target_index must be chronological."
        )

    if not fit_index.isin(
        target_index
    ).all():
        raise ValueError(
            "fit_index must be a subset of "
            "target_index."
        )

    if not (
        pca_ewma.index.equals(
            feature_support_raw.index
        )
        and pca_ewma.index.equals(
            latent_support_raw.index
        )
    ):
        raise ValueError(
            "Raw router signals must have "
            "identical indexes."
        )

    pca_ewma = pca_ewma.astype(float)
    feature_support_raw = (
        feature_support_raw.astype(float)
    )
    latent_support_raw = (
        latent_support_raw.astype(float)
    )

    delta = pca_ewma.diff()

    volatility = (
        pca_ewma
        .rolling(
            window=rolling_history_bins,
            min_periods=rolling_history_bins,
        )
        .std(ddof=0)
    )

    pca_fit = pca_ewma.reindex(
        fit_index
    )
    delta_fit = delta.reindex(
        fit_index
    )
    volatility_fit = volatility.reindex(
        fit_index
    )
    feature_support_fit = (
        feature_support_raw.reindex(
            fit_index
        )
    )
    latent_support_fit = (
        latent_support_raw.reindex(
            fit_index
        )
    )

    reference_series = {
        "pca_ewma": pca_fit,
        "pca_ewma_delta": delta_fit,
        "pca_ewma_volatility": (
            volatility_fit
        ),
        "feature_support": (
            feature_support_fit
        ),
        "latent_support": (
            latent_support_fit
        ),
    }

    for name, series in (
        reference_series.items()
    ):
        if (
            series.isna().any()
            or not np.isfinite(
                series.to_numpy(
                    dtype=float
                )
            ).all()
        ):
            raise ValueError(
                f"{name} router-fit reference "
                "contains missing/non-finite "
                "values."
            )

    pca_q90 = float(
        pca_fit.quantile(
            0.90,
            interpolation="higher",
        )
    )
    pca_q95 = float(
        pca_fit.quantile(
            0.95,
            interpolation="higher",
        )
    )

    delta_center, delta_scale = (
        robust_center_scale(
            delta_fit
        )
    )

    pca_target = pca_ewma.reindex(
        target_index
    )
    delta_target = delta.reindex(
        target_index
    )
    volatility_target = (
        volatility.reindex(
            target_index
        )
    )
    feature_support_target = (
        feature_support_raw.reindex(
            target_index
        )
    )
    latent_support_target = (
        latent_support_raw.reindex(
            target_index
        )
    )

    pca_percentile = (
        empirical_percentile(
            pca_fit.to_numpy(),
            pca_target.to_numpy(),
        )
    )
    volatility_percentile = (
        empirical_percentile(
            volatility_fit.to_numpy(),
            volatility_target.to_numpy(),
        )
    )
    feature_percentile = (
        empirical_percentile(
            feature_support_fit.to_numpy(),
            feature_support_target.to_numpy(),
        )
    )
    latent_percentile = (
        empirical_percentile(
            latent_support_fit.to_numpy(),
            latent_support_target.to_numpy(),
        )
    )

    combined_support = np.maximum(
        feature_percentile,
        latent_percentile,
    )

    q90_hits_full = (
        pca_ewma >= pca_q90
    ).astype(float)
    q95_hits_full = (
        pca_ewma >= pca_q95
    ).astype(float)

    q90_hit_fraction = (
        q90_hits_full
        .rolling(
            window=rolling_history_bins,
            min_periods=rolling_history_bins,
        )
        .mean()
        .reindex(target_index)
    )
    q95_hit_fraction = (
        q95_hits_full
        .rolling(
            window=rolling_history_bins,
            min_periods=rolling_history_bins,
        )
        .mean()
        .reindex(target_index)
    )

    frame = pd.DataFrame(
        {
            "pca_ewma_percentile": (
                pca_percentile
            ),
            "pca_ewma_delta_robust_z": (
                (
                    delta_target.to_numpy(
                        dtype=float
                    )
                    - delta_center
                )
                / delta_scale
            ),
            "pca_ewma_volatility_percentile": (
                volatility_percentile
            ),
            "feature_support_percentile": (
                feature_percentile
            ),
            "pca_latent_support_percentile": (
                latent_percentile
            ),
            "combined_support_percentile": (
                combined_support
            ),
            "recent_q90_hit_fraction": (
                q90_hit_fraction.to_numpy(
                    dtype=float
                )
            ),
            "recent_q95_hit_fraction": (
                q95_hit_fraction.to_numpy(
                    dtype=float
                )
            ),
        },
        index=target_index,
    )

    if frame.columns.tolist() != ROUTER_FEATURE_NAMES:
        raise RuntimeError(
            "Router feature order changed."
        )

    if (
        frame.isna().any().any()
        or not np.isfinite(
            frame.to_numpy(dtype=float)
        ).all()
    ):
        raise ValueError(
            "Router feature frame contains "
            "missing/non-finite values."
        )

    references = {
        "pca_q90_threshold": pca_q90,
        "pca_q95_threshold": pca_q95,
        "delta_center_median": (
            delta_center
        ),
        "delta_scale_iqr": (
            delta_scale
        ),
        "pca_ewma_reference": (
            np.sort(
                pca_fit.to_numpy(
                    dtype=float
                )
            ).tolist()
        ),
        "volatility_reference": (
            np.sort(
                volatility_fit.to_numpy(
                    dtype=float
                )
            ).tolist()
        ),
        "feature_support_reference": (
            np.sort(
                feature_support_fit.to_numpy(
                    dtype=float
                )
            ).tolist()
        ),
        "latent_support_reference": (
            np.sort(
                latent_support_fit.to_numpy(
                    dtype=float
                )
            ).tolist()
        ),
    }

    return frame, references


def logistic_candidate_grid(
    *,
    C_values: list[float],
    class_weights: list[str | None],
    probability_thresholds: list[float],
) -> list[LogisticCandidate]:
    candidates = [
        LogisticCandidate(
            C=float(C),
            class_weight=class_weight,
            probability_threshold=float(
                threshold
            ),
        )
        for C, class_weight, threshold
        in product(
            C_values,
            class_weights,
            probability_thresholds,
        )
    ]

    if not candidates:
        raise ValueError(
            "Logistic candidate grid is empty."
        )

    return candidates


def fit_logistic_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    C: float,
    class_weight: str | None,
    solver: str,
    max_iter: int,
    random_state: int,
) -> LogisticRegression:
    y = np.asarray(
        y,
        dtype=int,
    )

    if np.unique(y).size != 2:
        raise ValueError(
            "Logistic router requires both "
            "target classes in router-fit data."
        )

    model = LogisticRegression(
        C=float(C),
        class_weight=class_weight,
        solver=solver,
        max_iter=int(max_iter),
        random_state=int(
            random_state
        ),
    )
    model.fit(x, y)

    return model


def route_probabilities(
    model: LogisticRegression,
    x: np.ndarray,
) -> np.ndarray:
    classes = list(
        model.classes_
    )

    if 1 not in classes:
        raise RuntimeError(
            "Logistic model has no positive class."
        )

    positive_index = classes.index(1)

    return model.predict_proba(x)[
        :,
        positive_index,
    ]


def route_from_probability(
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Probability threshold must lie "
            "within [0, 1]."
        )

    values = np.asarray(
        probabilities,
        dtype=float,
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Router probabilities contain "
            "non-finite values."
        )

    return values >= threshold


def evidence_coverage(
    route: np.ndarray,
    evidence: np.ndarray,
) -> float | None:
    route = np.asarray(
        route,
        dtype=bool,
    )
    evidence = np.asarray(
        evidence,
        dtype=bool,
    )

    if route.shape != evidence.shape:
        raise ValueError(
            "route and evidence shapes differ."
        )

    denominator = int(
        evidence.sum()
    )

    if denominator == 0:
        return None

    numerator = int(
        (route & evidence).sum()
    )

    return numerator / denominator


def select_candidate(
    candidate_reports: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    eligible = [
        report
        for report in candidate_reports
        if bool(report["eligible"])
    ]

    if not eligible:
        raise RuntimeError(
            "No evidence-aware logistic "
            "candidate satisfies the frozen "
            "dual-dataset validation "
            "constraints."
        )

    def key(
        report: dict[str, Any],
    ) -> tuple[
        float,
        float,
        float,
        float,
        float,
        int,
    ]:
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
                    "worst_dataset_high_evidence_coverage"
                ]
            ),
            float(
                report["C"]
            ),
            -float(
                report[
                    "probability_threshold"
                ]
            ),
            (
                0
                if report[
                    "class_weight"
                ] is None
                else 1
            ),
        )

    return min(
        eligible,
        key=key,
    )


def serialize_standard_scaler(
    scaler: StandardScaler,
) -> dict[str, list[float]]:
    return {
        "mean": (
            np.asarray(
                scaler.mean_,
                dtype=float,
            ).tolist()
        ),
        "scale": (
            np.asarray(
                scaler.scale_,
                dtype=float,
            ).tolist()
        ),
    }


def serialize_logistic_model(
    model: LogisticRegression,
) -> dict[str, Any]:
    return {
        "classes": (
            np.asarray(
                model.classes_,
                dtype=int,
            ).tolist()
        ),
        "coef": (
            np.asarray(
                model.coef_,
                dtype=float,
            ).tolist()
        ),
        "intercept": (
            np.asarray(
                model.intercept_,
                dtype=float,
            ).tolist()
        ),
        "n_iter": (
            np.asarray(
                model.n_iter_,
                dtype=int,
            ).tolist()
        ),
    }
