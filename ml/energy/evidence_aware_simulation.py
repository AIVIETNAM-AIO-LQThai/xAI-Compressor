from __future__ import annotations

import time
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from ml.energy.support_aware_routing import empirical_percentile


def frozen_router_probability(
    router_features: pd.DataFrame,
    *,
    feature_names: list[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    coef: np.ndarray,
    intercept: float,
) -> np.ndarray:
    x = router_features.loc[:, feature_names].to_numpy(dtype=float)

    mean = np.asarray(scaler_mean, dtype=float).reshape(1, -1)
    scale = np.asarray(scaler_scale, dtype=float).reshape(1, -1)
    weights = np.asarray(coef, dtype=float).reshape(-1)

    if x.shape[1] != mean.shape[1]:
        raise ValueError("Router feature/scaler dimension mismatch.")
    if x.shape[1] != weights.shape[0]:
        raise ValueError("Router feature/coefficient dimension mismatch.")
    if np.any(scale <= 0.0):
        raise ValueError("Frozen router scaler contains non-positive scale.")

    z = (x - mean) / scale
    logits = z @ weights + float(intercept)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))

    if not np.isfinite(probabilities).all():
        raise ValueError("Frozen router probabilities are non-finite.")

    return probabilities


def route_summary(
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    values = np.asarray(probabilities, dtype=float)
    route = values >= float(threshold)

    return {
        "rows": len(values),
        "tcn_invocations": int(route.sum()),
        "tcn_invocation_fraction": float(route.mean()),
        "probability_mean": float(values.mean()),
        "probability_median": float(np.median(values)),
        "probability_p90": float(np.quantile(values, 0.90)),
        "probability_p99": float(np.quantile(values, 0.99)),
    }


def fixed_reference_router_features(
    *,
    pca_ewma: pd.Series,
    feature_support_raw: pd.Series,
    latent_support_raw: pd.Series,
    target_index: pd.DatetimeIndex,
    references: dict[str, Any],
    rolling_history_bins: int,
) -> pd.DataFrame:
    if rolling_history_bins <= 1:
        raise ValueError("rolling_history_bins must be > 1.")

    if not (
        pca_ewma.index.equals(feature_support_raw.index)
        and pca_ewma.index.equals(latent_support_raw.index)
    ):
        raise ValueError("Raw router signal indexes differ.")

    delta = pca_ewma.astype(float).diff()
    volatility = (
        pca_ewma.astype(float)
        .rolling(
            window=rolling_history_bins,
            min_periods=rolling_history_bins,
        )
        .std(ddof=0)
    )

    pca_target = pca_ewma.reindex(target_index).astype(float)
    delta_target = delta.reindex(target_index).astype(float)
    volatility_target = volatility.reindex(target_index).astype(float)
    feature_target = feature_support_raw.reindex(target_index).astype(float)
    latent_target = latent_support_raw.reindex(target_index).astype(float)

    components = {
        "pca": pca_target,
        "delta": delta_target,
        "volatility": volatility_target,
        "feature": feature_target,
        "latent": latent_target,
    }
    for name, series in components.items():
        if series.isna().any():
            raise ValueError(f"{name} target contains missing values.")

    pca_percentile = empirical_percentile(
        np.asarray(references["pca_ewma_reference"], dtype=float),
        pca_target.to_numpy(),
    )
    volatility_percentile = empirical_percentile(
        np.asarray(references["volatility_reference"], dtype=float),
        volatility_target.to_numpy(),
    )
    feature_percentile = empirical_percentile(
        np.asarray(references["feature_support_reference"], dtype=float),
        feature_target.to_numpy(),
    )
    latent_percentile = empirical_percentile(
        np.asarray(references["latent_support_reference"], dtype=float),
        latent_target.to_numpy(),
    )

    q90_hits = (pca_ewma >= float(references["pca_q90_threshold"])).astype(float)
    q95_hits = (pca_ewma >= float(references["pca_q95_threshold"])).astype(float)

    q90_fraction = (
        q90_hits
        .rolling(
            window=rolling_history_bins,
            min_periods=rolling_history_bins,
        )
        .mean()
        .reindex(target_index)
    )
    q95_fraction = (
        q95_hits
        .rolling(
            window=rolling_history_bins,
            min_periods=rolling_history_bins,
        )
        .mean()
        .reindex(target_index)
    )

    frame = pd.DataFrame(
        {
            "pca_ewma_percentile": pca_percentile,
            "pca_ewma_delta_robust_z": (
                delta_target.to_numpy()
                - float(references["delta_center_median"])
            )
            / float(references["delta_scale_iqr"]),
            "pca_ewma_volatility_percentile": volatility_percentile,
            "feature_support_percentile": feature_percentile,
            "pca_latent_support_percentile": latent_percentile,
            "combined_support_percentile": np.maximum(
                feature_percentile,
                latent_percentile,
            ),
            "recent_q90_hit_fraction": q90_fraction.to_numpy(dtype=float),
            "recent_q95_hit_fraction": q95_fraction.to_numpy(dtype=float),
        },
        index=target_index,
    )

    if frame.isna().any().any():
        raise ValueError("Frozen router features contain missing values.")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Frozen router features contain non-finite values.")

    return frame


def _float_working_copy(
    frame: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    shifted = frame.copy()

    for feature in features:
        shifted[feature] = (
            shifted[feature]
            .astype("float64")
        )

    return shifted


def location_shift(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    *,
    features: list[str],
    severity_sigma: float,
) -> pd.DataFrame:
    shifted = _float_working_copy(
        frame,
        features,
    )
    train_std = (
        train.loc[:, features]
        .astype(float)
        .std(ddof=0)
    )

    shifted.loc[:, features] = (
        shifted.loc[:, features]
        + float(severity_sigma) * train_std
    )
    return shifted


def scale_shift(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    *,
    features: list[str],
    factor: float,
) -> pd.DataFrame:
    shifted = _float_working_copy(
        frame,
        features,
    )
    mean = (
        train.loc[:, features]
        .astype(float)
        .mean()
    )

    shifted.loc[:, features] = (
        mean
        + float(factor)
        * (shifted.loc[:, features] - mean)
    )
    return shifted


def tail_inflation(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    *,
    features: list[str],
    tail_mask: pd.Series,
    factor: float,
) -> pd.DataFrame:
    shifted = _float_working_copy(
        frame,
        features,
    )
    mean = (
        train.loc[:, features]
        .astype(float)
        .mean()
    )
    mask = (
        tail_mask
        .reindex(
            frame.index,
            fill_value=False,
        )
        .astype(bool)
    )

    current = shifted.loc[
        mask,
        features,
    ]

    shifted.loc[
        mask,
        features,
    ] = (
        mean
        + float(factor)
        * (current - mean)
    )

    return shifted


def monotonic_nondecreasing(values: list[float], *, tolerance: float = 1e-12) -> bool:
    return all(
        later + tolerance >= earlier
        for earlier, later in pairwise(values)
    )


def operating_regimes(
    motor_current: pd.Series,
    *,
    low_upper: float,
    high_lower: float,
) -> pd.Series:
    values = motor_current.astype(float)

    regime = pd.Series(
        "middle",
        index=values.index,
        dtype="object",
    )
    regime.loc[values <= float(low_upper)] = "low"
    regime.loc[values >= float(high_lower)] = "high"
    return regime


def counterfactual_mixture(
    *,
    route: pd.Series,
    regimes: pd.Series,
    weights: dict[str, float],
) -> dict[str, Any]:
    required = {"low", "middle", "high"}
    if set(weights) != required:
        raise ValueError("Mixture weights must contain low/middle/high.")

    total_weight = float(sum(weights.values()))
    if not np.isclose(total_weight, 1.0, atol=1e-12):
        raise ValueError("Mixture weights must sum to 1.")

    aligned_regimes = regimes.reindex(route.index)
    if aligned_regimes.isna().any():
        raise ValueError("Operating regime alignment failed.")

    conditional = {}
    expected = 0.0

    for name in ("low", "middle", "high"):
        mask = aligned_regimes == name
        rows = int(mask.sum())
        if rows == 0:
            raise ValueError(f"Operating regime {name} has zero rows.")

        rate = float(route.loc[mask].mean())
        conditional[name] = {
            "rows": rows,
            "invocation_fraction": rate,
        }
        expected += float(weights[name]) * rate

    return {
        "weights": {key: float(value) for key, value in weights.items()},
        "conditional_regime_rates": conditional,
        "counterfactual_invocation_fraction": float(expected),
    }


def benchmark_frozen_decision(
    router_features: pd.DataFrame,
    *,
    feature_names: list[str],
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    coef: np.ndarray,
    intercept: float,
    probability_threshold: float,
    repetitions: int,
    minimum_seconds: float,
) -> dict[str, Any]:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")
    if minimum_seconds <= 0.0:
        raise ValueError("minimum_seconds must be positive.")

    timings = []
    passes = []
    checksum = 0

    for _ in range(repetitions):
        count = 0
        start = time.perf_counter()

        while True:
            probabilities = frozen_router_probability(
                router_features,
                feature_names=feature_names,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                coef=coef,
                intercept=intercept,
            )
            route = probabilities >= probability_threshold
            checksum += int(route.sum())
            count += 1

            elapsed = time.perf_counter() - start
            if elapsed >= minimum_seconds:
                break

        timings.append(float(elapsed))
        passes.append(int(count))

    rows = len(router_features)
    per_row_us = [
        (duration / (count * rows)) * 1.0e6
        for duration, count in zip(timings, passes)
    ]

    return {
        "scope": (
            "final_8_feature_standardization_plus_frozen_logistic_"
            "probability_and_threshold_decision"
        ),
        "rows_per_logical_pass": int(rows),
        "repetitions": int(repetitions),
        "duration_seconds": timings,
        "logical_passes": passes,
        "microseconds_per_row": per_row_us,
        "mean_microseconds_per_row": float(np.mean(per_row_us)),
        "sample_sd_microseconds_per_row": float(np.std(per_row_us, ddof=1)),
        "checksum": int(checksum),
        "cpu_energy": "UNKNOWN",
    }
