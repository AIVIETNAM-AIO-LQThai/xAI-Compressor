from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.detection.common import Scaler, transform_frame
from ml.detection.regime_pca import REGIME_NAMES


def score_summary(
    scores: pd.Series,
    *,
    threshold: float,
) -> dict[str, Any]:
    values = scores.astype(float).dropna()
    if values.empty:
        return {"count": 0}

    return {
        "count": len(values),
        "median": float(values.median()),
        "q90": float(values.quantile(0.90, interpolation="higher")),
        "q95": float(values.quantile(0.95, interpolation="higher")),
        "q99": float(values.quantile(0.99, interpolation="higher")),
        "q995": float(values.quantile(0.995, interpolation="higher")),
        "max": float(values.max()),
        "threshold_exceedance_fraction": float(
            (values >= threshold).mean()
        ),
    }


def transport_summary(
    calibration_scores: pd.Series,
    test_scores: pd.Series,
    *,
    threshold_quantile: float,
) -> dict[str, Any]:
    threshold = float(
        calibration_scores.quantile(
            threshold_quantile,
            interpolation="higher",
        )
    )
    calibration = score_summary(
        calibration_scores,
        threshold=threshold,
    )
    test = score_summary(
        test_scores,
        threshold=threshold,
    )

    return {
        "calibration_raw_threshold": threshold,
        "calibration": calibration,
        "test": test,
        "test_to_calibration_q99_ratio": (
            float(test["q99"] / calibration["q99"])
            if calibration["q99"] != 0.0
            else None
        ),
        "test_to_calibration_q995_ratio": (
            float(test["q995"] / calibration["q995"])
            if calibration["q995"] != 0.0
            else None
        ),
    }


def regime_transport_summary(
    calibration_scores: pd.Series,
    calibration_regimes: pd.Series,
    test_scores: pd.Series,
    test_regimes: pd.Series,
    *,
    threshold_quantile: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for regime in REGIME_NAMES:
        result[regime] = transport_summary(
            calibration_scores.loc[
                calibration_regimes == regime
            ],
            test_scores.loc[
                test_regimes == regime
            ],
            threshold_quantile=threshold_quantile,
        )

    return result


def incident_window_mask(
    index: pd.DatetimeIndex,
    incidents: list[dict[str, Any]],
    *,
    hours_before: int,
) -> pd.Series:
    mask = pd.Series(False, index=index)
    early = pd.Timedelta(hours=hours_before)

    for incident in incidents:
        start = pd.Timestamp(incident["start"])
        end = pd.Timestamp(incident["end"])
        mask.loc[
            (mask.index >= start - early)
            & (mask.index <= end)
        ] = True

    return mask


def feature_drift_summary(
    calibration: pd.DataFrame,
    background_test: pd.DataFrame,
    *,
    features: list[str],
    scaler: Scaler,
    abs_z_thresholds: list[float],
) -> dict[str, Any]:
    calibration_z = transform_frame(
        calibration,
        features,
        scaler,
    )
    background_z = transform_frame(
        background_test,
        features,
        scaler,
    )

    rows: list[dict[str, Any]] = []

    for position, feature in enumerate(features):
        cal = np.abs(calibration_z[:, position])
        test = np.abs(background_z[:, position])

        cal_q99 = float(np.quantile(cal, 0.99, method="higher"))
        test_q99 = float(np.quantile(test, 0.99, method="higher"))

        row: dict[str, Any] = {
            "feature": feature,
            "calibration_median_abs_z": float(np.median(cal)),
            "calibration_q99_abs_z": cal_q99,
            "background_test_median_abs_z": float(np.median(test)),
            "background_test_q99_abs_z": test_q99,
            "test_to_calibration_q99_abs_z_ratio": (
                float(test_q99 / cal_q99)
                if cal_q99 != 0.0
                else None
            ),
        }

        for threshold in abs_z_thresholds:
            key = str(threshold).replace(".", "_")
            row[
                f"calibration_fraction_abs_z_ge_{key}"
            ] = float(np.mean(cal >= threshold))
            row[
                f"background_test_fraction_abs_z_ge_{key}"
            ] = float(np.mean(test >= threshold))

        rows.append(row)

    by_feature = {
        row["feature"]: {
            key: value
            for key, value in row.items()
            if key != "feature"
        }
        for row in rows
    }

    top_by_test_q99 = sorted(
        rows,
        key=lambda row: row["background_test_q99_abs_z"],
        reverse=True,
    )
    top_by_ratio = sorted(
        rows,
        key=lambda row: (
            row["test_to_calibration_q99_abs_z_ratio"]
            if row["test_to_calibration_q99_abs_z_ratio"] is not None
            else -np.inf
        ),
        reverse=True,
    )

    return {
        "by_feature": by_feature,
        "ranked_by_background_test_q99_abs_z": [
            row["feature"] for row in top_by_test_q99
        ],
        "ranked_by_q99_abs_z_ratio": [
            row["feature"] for row in top_by_ratio
        ],
    }


def contribution_share_summary(
    contributions: pd.DataFrame,
    scores: pd.Series,
    *,
    threshold: float,
) -> dict[str, Any]:
    contributions, scores = contributions.align(
        scores,
        join="inner",
        axis=0,
    )

    def one(mask: pd.Series) -> dict[str, Any]:
        frame = contributions.loc[mask]
        if frame.empty:
            return {
                "row_count": 0,
                "group_share": {},
            }

        totals = frame.sum(axis=0).astype(float)
        grand_total = float(totals.sum())
        shares = (
            totals / grand_total
            if grand_total > 0.0
            else totals * 0.0
        )
        ordered = shares.sort_values(ascending=False)

        return {
            "row_count": len(frame),
            "group_share": {
                str(group): float(value)
                for group, value in ordered.items()
            },
        }

    all_mask = pd.Series(True, index=scores.index)
    high_mask = scores >= threshold

    return {
        "all_rows": one(all_mask),
        "above_calibration_raw_q995": one(high_mask),
    }


def high_score_run_summary(
    scores: pd.Series,
    regimes: pd.Series,
    *,
    threshold: float,
    gap_minutes: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    scores, regimes = scores.align(regimes, join="inner")
    scores = scores.sort_index()
    regimes = regimes.reindex(scores.index)

    mask = scores >= threshold
    gap = pd.Timedelta(minutes=gap_minutes)

    runs: list[dict[str, Any]] = []
    positions = np.flatnonzero(mask.to_numpy(dtype=bool))

    if len(positions):
        start_position = int(positions[0])
        previous_position = int(positions[0])

        def close_run(start: int, end: int) -> None:
            start_time = scores.index[start]
            end_time = scores.index[end]
            rows = end - start + 1
            duration = float(
                (end_time - start_time) / pd.Timedelta(minutes=1)
            )
            runs.append(
                {
                    "start": start_time,
                    "end": end_time,
                    "rows": int(rows),
                    "duration_minutes": duration,
                    "start_regime": str(regimes.iloc[start]),
                    "max_score": float(
                        scores.iloc[start : end + 1].max()
                    ),
                }
            )

        for position in positions[1:]:
            position = int(position)
            contiguous_position = position == previous_position + 1
            contiguous_time = (
                scores.index[position]
                - scores.index[previous_position]
                <= gap
            )
            if contiguous_position and contiguous_time:
                previous_position = position
                continue

            close_run(start_position, previous_position)
            start_position = position
            previous_position = position

        close_run(start_position, previous_position)

    table = pd.DataFrame(runs)

    if table.empty:
        return {"run_count": 0}, table

    def quantiles(series: pd.Series) -> dict[str, Any]:
        values = series.astype(float)
        return {
            "median": float(values.median()),
            "q90": float(values.quantile(0.90, interpolation="higher")),
            "q95": float(values.quantile(0.95, interpolation="higher")),
            "q99": float(values.quantile(0.99, interpolation="higher")),
            "max": float(values.max()),
        }

    return {
        "run_count": len(table),
        "rows": quantiles(table["rows"]),
        "duration_minutes": quantiles(table["duration_minutes"]),
        "start_regime_fraction": {
            regime: float(
                (table["start_regime"] == regime).mean()
            )
            for regime in REGIME_NAMES
        },
    }, table
