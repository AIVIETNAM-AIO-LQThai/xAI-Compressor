from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PERCENTILE_FEATURES = {
    "pca_ewma_percentile",
    "pca_ewma_volatility_percentile",
    "feature_support_percentile",
    "pca_latent_support_percentile",
    "combined_support_percentile",
}


def descriptive_summary(
    values: np.ndarray | pd.Series,
    *,
    quantiles: list[float],
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float).reshape(-1)

    if array.size == 0:
        raise ValueError("Cannot summarize an empty array.")
    if not np.isfinite(array).all():
        raise ValueError("Summary values contain non-finite entries.")

    result: dict[str, Any] = {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
        "quantiles": {},
    }

    for quantile in quantiles:
        result["quantiles"][str(quantile)] = float(
            np.quantile(
                array,
                quantile,
                method="linear",
            )
        )

    return result


def median_iqr(
    values: np.ndarray | pd.Series,
) -> dict[str, float]:
    array = np.asarray(values, dtype=float).reshape(-1)

    if array.size == 0:
        raise ValueError("Cannot compute median/IQR for an empty array.")
    if not np.isfinite(array).all():
        raise ValueError("Median/IQR values contain non-finite entries.")

    q25 = float(np.quantile(array, 0.25, method="linear"))
    median = float(np.quantile(array, 0.50, method="linear"))
    q75 = float(np.quantile(array, 0.75, method="linear"))

    return {
        "q25": q25,
        "median": median,
        "q75": q75,
        "iqr": q75 - q25,
    }


def raw_auc_ap(
    values: np.ndarray | pd.Series,
    target: np.ndarray | pd.Series,
) -> dict[str, float | None]:
    scores = np.asarray(values, dtype=float).reshape(-1)
    labels = np.asarray(target, dtype=bool).reshape(-1)

    if scores.shape != labels.shape:
        raise ValueError("scores and target shapes differ.")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain non-finite values.")

    if np.unique(labels).size < 2:
        return {
            "roc_auc": None,
            "average_precision": None,
        }

    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(
            average_precision_score(labels, scores)
        ),
    }


def spearman_rank_correlation(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
) -> float | None:
    x_values = np.asarray(x, dtype=float).reshape(-1)
    y_values = np.asarray(y, dtype=float).reshape(-1)

    if x_values.shape != y_values.shape:
        raise ValueError("Spearman inputs have different shapes.")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("Spearman inputs contain non-finite values.")

    x_rank = pd.Series(x_values).rank(method="average").to_numpy(dtype=float)
    y_rank = pd.Series(y_values).rank(method="average").to_numpy(dtype=float)

    if np.std(x_rank) == 0.0 or np.std(y_rank) == 0.0:
        return None

    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def threshold_diagnostics(
    probabilities: np.ndarray,
    high_evidence: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int | None]:
    probability = np.asarray(probabilities, dtype=float).reshape(-1)
    target = np.asarray(high_evidence, dtype=bool).reshape(-1)

    if probability.shape != target.shape:
        raise ValueError("probability and target shapes differ.")
    if not np.isfinite(probability).all():
        raise ValueError("probabilities contain non-finite values.")

    routed = probability >= float(threshold)
    routed_count = int(routed.sum())
    positive_count = int(target.sum())

    precision = (
        float(target[routed].mean())
        if routed_count
        else None
    )
    recall = (
        float((routed & target).sum() / positive_count)
        if positive_count
        else None
    )
    prevalence = float(target.mean())

    enrichment = (
        float(precision / prevalence)
        if (
            precision is not None
            and prevalence > 0.0
        )
        else None
    )

    metrics = raw_auc_ap(probability, target)

    return {
        "rows": int(target.size),
        "high_evidence_bins": positive_count,
        "routed_rows": routed_count,
        "invocation_fraction": float(routed.mean()),
        "high_evidence_prevalence": prevalence,
        "precision": precision,
        "recall": recall,
        "high_evidence_enrichment": enrichment,
        "roc_auc": metrics["roc_auc"],
        "average_precision": metrics["average_precision"],
    }


def calibration_boundaries(
    probabilities: np.ndarray,
    *,
    quantiles: list[float],
) -> list[float]:
    values = np.asarray(probabilities, dtype=float).reshape(-1)

    if values.size == 0:
        raise ValueError("Calibration probabilities are empty.")
    if not np.isfinite(values).all():
        raise ValueError("Calibration probabilities are non-finite.")

    return [
        float(np.quantile(values, q, method="linear"))
        for q in quantiles
    ]


def probability_calibration_table(
    probabilities: np.ndarray,
    high_evidence: np.ndarray,
    *,
    boundaries: list[float],
) -> list[dict[str, float | int | None]]:
    probability = np.asarray(probabilities, dtype=float).reshape(-1)
    target = np.asarray(high_evidence, dtype=bool).reshape(-1)

    if probability.shape != target.shape:
        raise ValueError("Calibration-table shapes differ.")

    edges = np.asarray(boundaries, dtype=float)
    bin_index = np.searchsorted(
        edges,
        probability,
        side="right",
    )

    table: list[dict[str, float | int | None]] = []

    for bin_id in range(len(boundaries) + 1):
        mask = bin_index == bin_id
        count = int(mask.sum())

        table.append(
            {
                "bin": bin_id,
                "lower_exclusive": (
                    None
                    if bin_id == 0
                    else float(edges[bin_id - 1])
                ),
                "upper_inclusive": (
                    None
                    if bin_id == len(boundaries)
                    else float(edges[bin_id])
                ),
                "rows": count,
                "mean_probability": (
                    float(probability[mask].mean())
                    if count
                    else None
                ),
                "observed_high_evidence_fraction": (
                    float(target[mask].mean())
                    if count
                    else None
                ),
            }
        )

    return table


def group_summary(
    *,
    mask: np.ndarray,
    router_probability: np.ndarray,
    router_features: pd.DataFrame,
    pca_ewma: pd.Series,
    raw_tcn_score: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)

    if selected.shape != (len(router_features),):
        raise ValueError("Group mask length mismatch.")

    count = int(selected.sum())
    if count == 0:
        return {
            "count": 0,
            "router_probability": None,
            "features": {
                column: None
                for column in router_features.columns
            },
            "pca_ewma": None,
            "raw_tcn_score": None,
        }

    result = {
        "count": count,
        "router_probability": median_iqr(
            np.asarray(router_probability, dtype=float)[selected]
        ),
        "features": {},
        "pca_ewma": median_iqr(
            pca_ewma.to_numpy(dtype=float)[selected]
        ),
        "raw_tcn_score": median_iqr(
            np.asarray(raw_tcn_score, dtype=float)[selected]
        ),
    }

    for column in router_features.columns:
        result["features"][column] = median_iqr(
            router_features[column].to_numpy(dtype=float)[selected]
        )

    return result


def monthly_route_summary(
    *,
    target_index: pd.DatetimeIndex,
    high_evidence: np.ndarray,
    evidence_route: np.ndarray,
    q90_route: np.ndarray,
    months: list[str],
) -> dict[str, Any]:
    high = np.asarray(high_evidence, dtype=bool)
    evidence = np.asarray(evidence_route, dtype=bool)
    q90 = np.asarray(q90_route, dtype=bool)

    if not (
        len(target_index)
        == high.size
        == evidence.size
        == q90.size
    ):
        raise ValueError("Monthly summary arrays are misaligned.")

    period = pd.DatetimeIndex(target_index).to_period("M")
    result: dict[str, Any] = {}

    for month in months:
        mask = np.asarray(
            period == pd.Period(month, freq="M"),
            dtype=bool,
        )
        rows = int(mask.sum())
        high_count = int((mask & high).sum())

        evidence_high_covered = int(
            (mask & high & evidence).sum()
        )
        q90_high_covered = int(
            (mask & high & q90).sum()
        )

        result[month] = {
            "valid_tcn_rows": rows,
            "high_evidence_bins": high_count,
            "high_evidence_prevalence": (
                high_count / rows
                if rows
                else None
            ),
            "evidence_aware_invocation_fraction": (
                float(evidence[mask].mean())
                if rows
                else None
            ),
            "evidence_aware_high_coverage": (
                evidence_high_covered / high_count
                if high_count
                else None
            ),
            "evidence_aware_false_negative_count": (
                high_count - evidence_high_covered
            ),
            "q90_invocation_fraction": (
                float(q90[mask].mean())
                if rows
                else None
            ),
            "q90_high_coverage": (
                q90_high_covered / high_count
                if high_count
                else None
            ),
        }

    return result
