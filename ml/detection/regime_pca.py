from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ml.detection.common import (
    Scaler,
    fit_scaler,
    transform_frame,
)

REGIME_LOW_LOW = "low_current_low_pressure"
REGIME_HIGH_CURRENT = "high_current_only"
REGIME_HIGH_PRESSURE = "high_pressure_only"
REGIME_HIGH_BOTH = "high_current_high_pressure"

REGIME_NAMES = (
    REGIME_LOW_LOW,
    REGIME_HIGH_CURRENT,
    REGIME_HIGH_PRESSURE,
    REGIME_HIGH_BOTH,
)


@dataclass(frozen=True)
class RegimePCADetector:
    features: list[str]
    scaler: Scaler
    scaler_name: str
    current_feature: str
    pressure_feature: str
    current_threshold: float
    pressure_threshold: float
    high_quantile: float
    component_count: int
    global_explained_variance_ratio: float
    models: dict[str, PCA]
    train_regime_counts: dict[str, int]


@dataclass(frozen=True)
class RegimePCAExplanation:
    score: pd.Series
    contributions: pd.DataFrame
    normalized_contributions: pd.DataFrame
    regime: pd.Series


def assign_regimes(
    frame: pd.DataFrame,
    *,
    current_feature: str,
    pressure_feature: str,
    current_threshold: float,
    pressure_threshold: float,
) -> pd.Series:
    for feature in (
        current_feature,
        pressure_feature,
    ):
        if feature not in frame.columns:
            raise ValueError(
                f"Missing router feature: {feature!r}."
            )

    high_current = (
        frame[current_feature].astype(float)
        >= current_threshold
    )
    high_pressure = (
        frame[pressure_feature].astype(float)
        >= pressure_threshold
    )

    values = np.full(
        len(frame),
        REGIME_LOW_LOW,
        dtype=object,
    )

    values[
        high_current.to_numpy()
        & ~high_pressure.to_numpy()
    ] = REGIME_HIGH_CURRENT

    values[
        ~high_current.to_numpy()
        & high_pressure.to_numpy()
    ] = REGIME_HIGH_PRESSURE

    values[
        high_current.to_numpy()
        & high_pressure.to_numpy()
    ] = REGIME_HIGH_BOTH

    return pd.Series(
        values,
        index=frame.index,
        name="operating_regime",
        dtype="object",
    )


def fit_regime_pca_detector(
    frame: pd.DataFrame,
    features: list[str],
    *,
    current_feature: str,
    pressure_feature: str,
    high_quantile: float,
    variance_retained: float,
) -> RegimePCADetector:
    if not 0.0 < high_quantile < 1.0:
        raise ValueError(
            "high_quantile must be in (0, 1)."
        )

    if not 0.0 < variance_retained <= 1.0:
        raise ValueError(
            "variance_retained must be in (0, 1]."
        )

    scaler = fit_scaler(
        frame,
        features,
        method="standard",
    )

    x = transform_frame(
        frame,
        features,
        scaler,
    )

    global_model = PCA(
        n_components=variance_retained,
        svd_solver="full",
    )
    global_model.fit(x)

    component_count = int(
        global_model.n_components_
    )

    current_threshold = float(
        frame[current_feature]
        .astype(float)
        .quantile(high_quantile)
    )

    pressure_threshold = float(
        frame[pressure_feature]
        .astype(float)
        .quantile(high_quantile)
    )

    regimes = assign_regimes(
        frame,
        current_feature=current_feature,
        pressure_feature=pressure_feature,
        current_threshold=current_threshold,
        pressure_threshold=pressure_threshold,
    )

    labels = regimes.to_numpy()
    models: dict[str, PCA] = {}
    counts: dict[str, int] = {}

    for regime in REGIME_NAMES:
        mask = labels == regime
        count = int(mask.sum())
        counts[regime] = count

        if count <= component_count:
            raise ValueError(
                "Insufficient training rows for regime "
                f"{regime!r}: {count} rows for "
                f"{component_count} components."
            )

        model = PCA(
            n_components=component_count,
            svd_solver="full",
        )
        model.fit(x[mask])
        models[regime] = model

    return RegimePCADetector(
        features=list(features),
        scaler=scaler,
        scaler_name="standard",
        current_feature=current_feature,
        pressure_feature=pressure_feature,
        current_threshold=current_threshold,
        pressure_threshold=pressure_threshold,
        high_quantile=high_quantile,
        component_count=component_count,
        global_explained_variance_ratio=float(
            global_model.explained_variance_ratio_.sum()
        ),
        models=models,
        train_regime_counts=counts,
    )


def _reconstruct(
    frame: pd.DataFrame,
    detector: RegimePCADetector,
) -> tuple[np.ndarray, pd.Series]:
    x = transform_frame(
        frame,
        detector.features,
        detector.scaler,
    )

    regimes = assign_regimes(
        frame,
        current_feature=detector.current_feature,
        pressure_feature=detector.pressure_feature,
        current_threshold=detector.current_threshold,
        pressure_threshold=detector.pressure_threshold,
    )

    labels = regimes.to_numpy()
    reconstructed = np.empty_like(
        x,
        dtype=float,
    )

    for regime in REGIME_NAMES:
        mask = labels == regime

        if not mask.any():
            continue

        model = detector.models[regime]
        encoded = model.transform(x[mask])
        reconstructed[mask] = (
            model.inverse_transform(encoded)
        )

    return reconstructed, regimes


def score_regime_pca_detector(
    frame: pd.DataFrame,
    detector: RegimePCADetector,
) -> pd.Series:
    x = transform_frame(
        frame,
        detector.features,
        detector.scaler,
    )

    reconstructed, _ = _reconstruct(
        frame,
        detector,
    )

    residual = x - reconstructed

    return pd.Series(
        np.mean(
            residual**2,
            axis=1,
        ),
        index=frame.index,
        name="raw_score",
    )


def explain_regime_pca_detector(
    frame: pd.DataFrame,
    detector: RegimePCADetector,
) -> RegimePCAExplanation:
    x = transform_frame(
        frame,
        detector.features,
        detector.scaler,
    )

    reconstructed, regimes = _reconstruct(
        frame,
        detector,
    )

    residual_squared = (
        x - reconstructed
    ) ** 2

    feature_count = len(
        detector.features
    )

    contribution_values = (
        residual_squared
        / feature_count
    )

    contributions = pd.DataFrame(
        contribution_values,
        index=frame.index,
        columns=detector.features,
    )

    score = contributions.sum(
        axis=1
    ).rename("raw_score")

    residual_total = residual_squared.sum(
        axis=1
    )

    normalized_values = np.divide(
        residual_squared,
        residual_total[:, None],
        out=np.zeros_like(
            residual_squared,
            dtype=float,
        ),
        where=(
            residual_total[:, None] > 0.0
        ),
    )

    normalized = pd.DataFrame(
        normalized_values,
        index=frame.index,
        columns=detector.features,
    )

    return RegimePCAExplanation(
        score=score,
        contributions=contributions,
        normalized_contributions=normalized,
        regime=regimes,
    )
