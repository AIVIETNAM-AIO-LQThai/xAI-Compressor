from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from ml.detection.alerts import causal_ewma
from ml.energy.support_aware_routing import empirical_percentile
from ml.temporal.sequences import CausalSequenceBatch, build_causal_sequences
from ml.temporal.tcn import TCNForecaster, TemporalForecastConfig

ROUTE_NAMES = (
    "always_on_tcn",
    "fixed_pca_route_q90",
    "fixed_pca_route_q95",
    "fixed_pca_route_q99",
    "frozen_evidence_aware_router",
)


@dataclass(frozen=True)
class FrozenEvaluationContext:
    temporal_batch: CausalSequenceBatch
    pca_ewma: pd.Series
    feature_support_raw: pd.Series
    latent_support_raw: pd.Series
    router_features: pd.DataFrame
    routes: dict[str, np.ndarray]


def build_tcn_model(
    checkpoint: dict[str, Any],
    *,
    device: torch.device,
) -> TCNForecaster:
    model_cfg = checkpoint["model_config"]
    model = TCNForecaster(
        TemporalForecastConfig(
            input_dim=int(model_cfg["input_dim"]),
            hidden_dim=int(model_cfg["hidden_dim"]),
            kernel_size=int(model_cfg["kernel_size"]),
            dilations=tuple(
                int(value)
                for value in model_cfg["dilations"]
            ),
            dropout=float(model_cfg["dropout"]),
        )
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()


def build_temporal_batch_from_checkpoint(
    frame: pd.DataFrame,
    checkpoint: dict[str, Any],
) -> CausalSequenceBatch:
    features = [str(value) for value in checkpoint["features"]]
    if frame.columns.tolist() != features:
        raise RuntimeError("EVALUATION schema differs from frozen TCN schema.")

    values = frame.loc[:, features].to_numpy(dtype=float)
    mean = np.asarray(checkpoint["scaler_location"], dtype=float)
    scale = np.asarray(checkpoint["scaler_scale"], dtype=float)

    if mean.shape != (len(features),) or scale.shape != (len(features),):
        raise RuntimeError("Frozen TCN scaler dimension mismatch.")
    if np.any(scale <= 0.0):
        raise RuntimeError("Frozen TCN scaler contains non-positive scale.")

    scaled = (values - mean.reshape(1, -1)) / scale.reshape(1, -1)

    if not np.isfinite(scaled).all():
        raise RuntimeError("Frozen TCN transform produced non-finite values.")

    sequence = checkpoint["sequence"]
    return build_causal_sequences(
        scaled,
        pd.DatetimeIndex(frame.index),
        sequence_length=int(sequence["history_bins"]),
        expected_step=pd.Timedelta(
            minutes=int(sequence["bin_minutes"])
        ),
    )


def score_tcn_batch(
    model: TCNForecaster,
    batch: CausalSequenceBatch,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    scores: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, len(batch.inputs), batch_size):
            stop = min(len(batch.inputs), start + batch_size)
            inputs = torch.from_numpy(batch.inputs[start:stop]).to(device)
            targets = torch.from_numpy(batch.targets[start:stop]).to(device)
            predictions = model(inputs)
            values = torch.mean(
                (targets - predictions).square(),
                dim=1,
            )
            scores.append(values.detach().cpu().numpy())

    result = np.concatenate(scores)
    if not np.isfinite(result).all():
        raise RuntimeError("EVALUATION TCN scores contain non-finite values.")
    return result.astype(float, copy=False)


def frozen_pca_support_signals(
    frame: pd.DataFrame,
    adaptation_freeze: dict[str, Any],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    pca = adaptation_freeze["pca"]
    support_scaler = adaptation_freeze["support_scaler"]
    features = [str(value) for value in pca["features"]]

    if frame.columns.tolist() != features:
        raise RuntimeError("EVALUATION schema differs from frozen PCA schema.")

    raw = frame.loc[:, features].to_numpy(dtype=float)

    center = np.asarray(pca["scaler"]["center"], dtype=float)
    robust_scale = np.asarray(pca["scaler"]["scale"], dtype=float)
    if np.any(robust_scale <= 0.0):
        raise RuntimeError("Frozen RobustScaler contains non-positive scale.")

    scaled = (
        raw - center.reshape(1, -1)
    ) / robust_scale.reshape(1, -1)

    components = np.asarray(pca["components"], dtype=float)
    pca_mean = np.asarray(pca["mean"], dtype=float)
    variance = np.asarray(pca["explained_variance"], dtype=float)

    latent = (scaled - pca_mean.reshape(1, -1)) @ components.T
    reconstructed = latent @ components + pca_mean.reshape(1, -1)
    residual = scaled - reconstructed
    pca_raw = np.mean(np.square(residual), axis=1)

    if np.any(variance <= 0.0):
        raise RuntimeError("Frozen PCA explained variance is non-positive.")
    latent_support = np.sqrt(
        np.sum(
            np.square(latent) / variance.reshape(1, -1),
            axis=1,
        )
    )

    standard_mean = np.asarray(support_scaler["mean"], dtype=float)
    standard_scale = np.asarray(support_scaler["scale"], dtype=float)
    if np.any(standard_scale <= 0.0):
        raise RuntimeError("Frozen support scaler contains non-positive scale.")

    support_scaled = (
        raw - standard_mean.reshape(1, -1)
    ) / standard_scale.reshape(1, -1)
    feature_support = np.quantile(
        np.abs(support_scaled),
        0.95,
        axis=1,
        method="linear",
    )

    index = frame.index
    return (
        pd.Series(pca_raw, index=index, name="pca_raw_score", dtype=float),
        pd.Series(
            feature_support,
            index=index,
            name="feature_support_raw",
            dtype=float,
        ),
        pd.Series(
            latent_support,
            index=index,
            name="pca_latent_support_raw",
            dtype=float,
        ),
    )


def _segment_ids(
    index: pd.DatetimeIndex,
    *,
    reset_gap_minutes: int,
) -> np.ndarray:
    if reset_gap_minutes <= 0:
        raise ValueError("reset_gap_minutes must be positive.")

    deltas = pd.Series(index, index=index).diff()
    boundaries = (
        deltas > pd.Timedelta(minutes=reset_gap_minutes)
    ).fillna(False)
    return boundaries.cumsum().to_numpy(dtype=int)


def _gap_aware_rolling(
    values: pd.Series,
    *,
    window: int,
    reset_gap_minutes: int,
    operation: str,
) -> pd.Series:
    if window <= 1:
        raise ValueError("window must be > 1.")

    result = pd.Series(np.nan, index=values.index, dtype=float)
    segment_ids = _segment_ids(
        pd.DatetimeIndex(values.index),
        reset_gap_minutes=reset_gap_minutes,
    )

    for segment_id in np.unique(segment_ids):
        mask = segment_ids == segment_id
        segment = values.iloc[np.flatnonzero(mask)].astype(float)
        rolling = segment.rolling(window=window, min_periods=window)

        if operation == "mean":
            output = rolling.mean()
        elif operation == "std":
            output = rolling.std(ddof=0)
        else:
            raise ValueError(f"Unknown rolling operation: {operation}.")

        result.loc[segment.index] = output.to_numpy(dtype=float)

    return result


def _gap_aware_delta(
    values: pd.Series,
    *,
    reset_gap_minutes: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    segment_ids = _segment_ids(
        pd.DatetimeIndex(values.index),
        reset_gap_minutes=reset_gap_minutes,
    )

    for segment_id in np.unique(segment_ids):
        mask = segment_ids == segment_id
        segment = values.iloc[np.flatnonzero(mask)].astype(float)
        result.loc[segment.index] = segment.diff().to_numpy(dtype=float)

    return result


def gap_aware_router_features(
    *,
    pca_ewma: pd.Series,
    feature_support_raw: pd.Series,
    latent_support_raw: pd.Series,
    target_index: pd.DatetimeIndex,
    references: dict[str, Any],
    rolling_history_bins: int,
    reset_gap_minutes: int,
) -> pd.DataFrame:
    if not (
        pca_ewma.index.equals(feature_support_raw.index)
        and pca_ewma.index.equals(latent_support_raw.index)
    ):
        raise RuntimeError("Frozen router raw signal indexes differ.")

    delta = _gap_aware_delta(
        pca_ewma,
        reset_gap_minutes=reset_gap_minutes,
    )
    volatility = _gap_aware_rolling(
        pca_ewma,
        window=rolling_history_bins,
        reset_gap_minutes=reset_gap_minutes,
        operation="std",
    )

    pca_target = pca_ewma.reindex(target_index)
    delta_target = delta.reindex(target_index)
    volatility_target = volatility.reindex(target_index)
    feature_target = feature_support_raw.reindex(target_index)
    latent_target = latent_support_raw.reindex(target_index)

    pca_percentile = empirical_percentile(
        np.asarray(references["pca_ewma_reference"], dtype=float),
        pca_target.to_numpy(dtype=float),
    )
    volatility_percentile = empirical_percentile(
        np.asarray(references["volatility_reference"], dtype=float),
        volatility_target.to_numpy(dtype=float),
    )
    feature_percentile = empirical_percentile(
        np.asarray(references["feature_support_reference"], dtype=float),
        feature_target.to_numpy(dtype=float),
    )
    latent_percentile = empirical_percentile(
        np.asarray(references["latent_support_reference"], dtype=float),
        latent_target.to_numpy(dtype=float),
    )

    q90_hits = (
        pca_ewma >= float(references["pca_q90_threshold"])
    ).astype(float)
    q95_hits = (
        pca_ewma >= float(references["pca_q95_threshold"])
    ).astype(float)

    q90_fraction = _gap_aware_rolling(
        q90_hits,
        window=rolling_history_bins,
        reset_gap_minutes=reset_gap_minutes,
        operation="mean",
    ).reindex(target_index)
    q95_fraction = _gap_aware_rolling(
        q95_hits,
        window=rolling_history_bins,
        reset_gap_minutes=reset_gap_minutes,
        operation="mean",
    ).reindex(target_index)

    frame = pd.DataFrame(
        {
            "pca_ewma_percentile": pca_percentile,
            "pca_ewma_delta_robust_z": (
                delta_target.to_numpy(dtype=float)
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
        raise RuntimeError(
            "Gap-aware frozen router features contain missing values."
        )
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise RuntimeError(
            "Gap-aware frozen router features contain non-finite values."
        )

    return frame


def frozen_router_probability(
    router_features: pd.DataFrame,
    frozen_router: dict[str, Any],
) -> np.ndarray:
    feature_names = [str(value) for value in frozen_router["feature_names"]]
    if router_features.columns.tolist() != feature_names:
        raise RuntimeError("Frozen router feature identity/order mismatch.")

    x = router_features.to_numpy(dtype=float)
    mean = np.asarray(
        frozen_router["feature_scaler"]["mean"],
        dtype=float,
    )
    scale = np.asarray(
        frozen_router["feature_scaler"]["scale"],
        dtype=float,
    )
    coef = np.asarray(
        frozen_router["logistic"]["coef"][0],
        dtype=float,
    )
    intercept = float(frozen_router["logistic"]["intercept"][0])

    if np.any(scale <= 0.0):
        raise RuntimeError("Frozen final router scaler is invalid.")

    standardized = (x - mean.reshape(1, -1)) / scale.reshape(1, -1)
    logits = standardized @ coef + intercept
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))

    if not np.isfinite(probabilities).all():
        raise RuntimeError("Frozen router probabilities are non-finite.")

    return probabilities


def build_frozen_routes(
    *,
    frame: pd.DataFrame,
    temporal_batch: CausalSequenceBatch,
    adaptation_freeze: dict[str, Any],
) -> FrozenEvaluationContext:
    pca_raw, feature_support, latent_support = frozen_pca_support_signals(
        frame,
        adaptation_freeze,
    )

    reset_gap_minutes = 30
    pca_ewma = causal_ewma(
        pca_raw,
        alpha=0.20,
        reset_gap_minutes=reset_gap_minutes,
    )

    target_index = pd.DatetimeIndex(temporal_batch.target_index)
    references = adaptation_freeze["calibration_references"]

    router_features = gap_aware_router_features(
        pca_ewma=pca_ewma,
        feature_support_raw=feature_support,
        latent_support_raw=latent_support,
        target_index=target_index,
        references=references,
        rolling_history_bins=12,
        reset_gap_minutes=reset_gap_minutes,
    )

    pca_target = pca_ewma.reindex(target_index)
    if pca_target.isna().any():
        raise RuntimeError("PCA-EWMA alignment to TCN windows failed.")

    probability = frozen_router_probability(
        router_features,
        adaptation_freeze["frozen_router"],
    )
    router_threshold = float(
        adaptation_freeze["frozen_router"]["probability_threshold"]
    )

    routes = {
        "always_on_tcn": np.ones(len(target_index), dtype=bool),
        "fixed_pca_route_q90": (
            pca_target.to_numpy(dtype=float)
            >= float(references["pca_q90_threshold"])
        ),
        "fixed_pca_route_q95": (
            pca_target.to_numpy(dtype=float)
            >= float(references["pca_q95_threshold"])
        ),
        "fixed_pca_route_q99": (
            pca_target.to_numpy(dtype=float)
            >= float(references["pca_q99_threshold"])
        ),
        "frozen_evidence_aware_router": probability >= router_threshold,
    }

    if tuple(routes) != ROUTE_NAMES:
        raise RuntimeError("Frozen route-system order changed.")

    return FrozenEvaluationContext(
        temporal_batch=temporal_batch,
        pca_ewma=pca_ewma,
        feature_support_raw=feature_support,
        latent_support_raw=latent_support,
        router_features=router_features,
        routes=routes,
    )


def route_evidence_metrics(
    route: np.ndarray,
    *,
    high_evidence: np.ndarray,
    alert_evidence: np.ndarray,
) -> dict[str, int | float | None]:
    route = np.asarray(route, dtype=bool)
    high = np.asarray(high_evidence, dtype=bool)
    alert = np.asarray(alert_evidence, dtype=bool)

    if route.shape != high.shape or route.shape != alert.shape:
        raise ValueError("Route and evidence shapes differ.")

    high_total = int(high.sum())
    alert_total = int(alert.sum())
    high_covered = int((route & high).sum())
    alert_covered = int((route & alert).sum())

    return {
        "valid_tcn_rows": int(route.size),
        "tcn_invocations": int(route.sum()),
        "tcn_invocation_fraction": float(route.mean()),
        "high_evidence_bins": high_total,
        "covered_high_evidence_bins": high_covered,
        "high_evidence_coverage": (
            high_covered / high_total
            if high_total
            else None
        ),
        "tcn_alert_bins": alert_total,
        "covered_tcn_alert_bins": alert_covered,
        "tcn_alert_bin_coverage": (
            alert_covered / alert_total
            if alert_total
            else None
        ),
    }


def evidence_stage_status(
    evidence_aware_metrics: dict[str, Any],
    *,
    minimum_high_coverage: float,
    minimum_alert_coverage: float,
) -> str:
    if int(evidence_aware_metrics["high_evidence_bins"]) == 0:
        return "INCONCLUSIVE_EXTERNAL_EVALUATION"
    if int(evidence_aware_metrics["tcn_alert_bins"]) == 0:
        return "INCONCLUSIVE_EXTERNAL_EVALUATION"

    high = float(evidence_aware_metrics["high_evidence_coverage"])
    alert = float(evidence_aware_metrics["tcn_alert_bin_coverage"])

    if high >= minimum_high_coverage and alert >= minimum_alert_coverage:
        return "EVIDENCE_COVERAGE_PASS"

    return "EVIDENCE_COVERAGE_FAIL"


def final_external_status(
    *,
    evidence_status: str,
    evidence_aware_metrics: dict[str, Any],
    mean_gross_energy_reduction: float,
    gross_reductions: list[float],
    success_criteria: dict[str, Any],
) -> str:
    if evidence_status == "INCONCLUSIVE_EXTERNAL_EVALUATION":
        return evidence_status

    high = evidence_aware_metrics["high_evidence_coverage"]
    alert = evidence_aware_metrics["tcn_alert_bin_coverage"]

    if high is None or alert is None:
        return "INCONCLUSIVE_EXTERNAL_EVALUATION"

    pass_all = (
        float(high)
        >= float(success_criteria["minimum_high_tcn_evidence_bin_coverage"])
        and float(alert)
        >= float(success_criteria["minimum_tcn_alert_bin_coverage"])
        and mean_gross_energy_reduction
        >= float(success_criteria["minimum_mean_gross_gpu_energy_reduction"])
        and (
            not bool(
                success_criteria[
                    "gross_energy_reduction_positive_each_repetition"
                ]
            )
            or all(value > 0.0 for value in gross_reductions)
        )
    )

    return (
        "EXTERNAL_TRANSPORT_PASS"
        if pass_all
        else "EXTERNAL_TRANSPORT_FAIL"
    )
