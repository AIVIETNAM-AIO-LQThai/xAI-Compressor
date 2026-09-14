from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.detection.common import Scaler, transform_frame


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


def raw_quantile_summary(
    series: pd.Series,
) -> dict[str, Any]:
    values = series.astype(float).dropna()
    if values.empty:
        return {"count": 0}

    return {
        "count": len(values),
        "min": float(values.min()),
        "q01": float(values.quantile(0.01)),
        "median": float(values.median()),
        "q99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def support_and_scaler_summary(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    background_test: pd.DataFrame,
    *,
    features: list[str],
    all_model_features: list[str],
    scaler: Scaler,
    abs_z_thresholds: list[float],
) -> dict[str, Any]:
    calibration_z = transform_frame(
        calibration,
        all_model_features,
        scaler,
    )
    background_z = transform_frame(
        background_test,
        all_model_features,
        scaler,
    )

    feature_positions = {
        feature: all_model_features.index(feature)
        for feature in features
    }

    scaler_mean = getattr(scaler, "mean_", None)
    scaler_scale = getattr(scaler, "scale_", None)
    if scaler_mean is None or scaler_scale is None:
        raise TypeError("Expected a fitted StandardScaler.")

    result: dict[str, Any] = {}

    for feature in features:
        position = feature_positions[feature]
        train_values = train[feature].astype(float)
        calibration_values = calibration[feature].astype(float)
        background_values = background_test[feature].astype(float)

        train_min = float(train_values.min())
        train_max = float(train_values.max())
        cal_z = np.abs(calibration_z[:, position])
        test_z = np.abs(background_z[:, position])

        item: dict[str, Any] = {
            "train_scaler_mean": float(scaler_mean[position]),
            "train_scaler_scale": float(scaler_scale[position]),
            "train_raw": raw_quantile_summary(train_values),
            "calibration_raw": raw_quantile_summary(
                calibration_values
            ),
            "background_test_raw": raw_quantile_summary(
                background_values
            ),
            "calibration_below_train_min_fraction": float(
                (calibration_values < train_min).mean()
            ),
            "calibration_above_train_max_fraction": float(
                (calibration_values > train_max).mean()
            ),
            "background_test_below_train_min_fraction": float(
                (background_values < train_min).mean()
            ),
            "background_test_above_train_max_fraction": float(
                (background_values > train_max).mean()
            ),
        }

        for threshold in abs_z_thresholds:
            key = str(threshold).replace(".", "_")
            item[
                f"calibration_fraction_abs_z_ge_{key}"
            ] = float(np.mean(cal_z >= threshold))
            item[
                f"background_test_fraction_abs_z_ge_{key}"
            ] = float(np.mean(test_z >= threshold))

        result[feature] = item

    return result


def analog_min_coherence(
    frame: pd.DataFrame,
    *,
    sensor: str,
    all_model_features: list[str],
    scaler: Scaler,
    min_z_threshold: float,
    context_z_threshold: float,
) -> dict[str, Any]:
    required = [
        f"{sensor}__mean",
        f"{sensor}__std",
        f"{sensor}__min",
        f"{sensor}__max",
        f"{sensor}__last",
    ]
    missing = [
        feature for feature in required
        if feature not in frame.columns
    ]
    if missing:
        raise ValueError(
            f"Missing analog features for {sensor}: {missing}"
        )

    z = transform_frame(
        frame,
        all_model_features,
        scaler,
    )
    positions = {
        feature: all_model_features.index(feature)
        for feature in required
    }

    z_min = np.abs(z[:, positions[f"{sensor}__min"]])
    z_mean = np.abs(z[:, positions[f"{sensor}__mean"]])
    z_last = np.abs(z[:, positions[f"{sensor}__last"]])

    extreme_min = z_min >= min_z_threshold
    count = int(extreme_min.sum())

    if count == 0:
        return {
            "extreme_min_count": 0,
            "extreme_min_fraction": 0.0,
        }

    mean_extreme = z_mean >= context_z_threshold
    last_extreme = z_last >= context_z_threshold
    min_only = extreme_min & ~mean_extreme & ~last_extreme

    values = frame.loc[extreme_min, required].astype(float)

    return {
        "extreme_min_count": count,
        "extreme_min_fraction": float(extreme_min.mean()),
        "among_extreme_min_mean_extreme_fraction": float(
            mean_extreme[extreme_min].mean()
        ),
        "among_extreme_min_last_extreme_fraction": float(
            last_extreme[extreme_min].mean()
        ),
        "among_extreme_min_mean_and_last_nonextreme_fraction": float(
            min_only.sum() / count
        ),
        "raw_mean_minus_min": raw_quantile_summary(
            values[f"{sensor}__mean"]
            - values[f"{sensor}__min"]
        ),
        "raw_last_minus_min": raw_quantile_summary(
            values[f"{sensor}__last"]
            - values[f"{sensor}__min"]
        ),
    }


def digital_support_summary(
    frame: pd.DataFrame,
    *,
    sensor: str,
) -> dict[str, Any]:
    last = frame[f"{sensor}__last"].astype(float)
    active_ratio = frame[
        f"{sensor}__active_ratio"
    ].astype(float)
    transitions = frame[
        f"{sensor}__transitions"
    ].astype(float)

    unique_last = sorted(
        float(value) for value in last.unique()
    )
    unique_transitions = sorted(
        float(value)
        for value in transitions.unique()
    )

    return {
        "rows": len(frame),
        "last": {
            "unique_values": unique_last[:20],
            "unique_value_count": int(last.nunique()),
            "nonzero_fraction": float((last != 0.0).mean()),
            "active_fraction_gt_0_5": float(
                (last > 0.5).mean()
            ),
        },
        "active_ratio": {
            "raw": raw_quantile_summary(active_ratio),
            "any_active_fraction": float(
                (active_ratio > 0.0).mean()
            ),
            "mostly_active_fraction_ge_0_5": float(
                (active_ratio >= 0.5).mean()
            ),
        },
        "transitions": {
            "raw": raw_quantile_summary(transitions),
            "unique_values": unique_transitions[:20],
            "unique_value_count": int(
                transitions.nunique()
            ),
            "any_transition_fraction": float(
                (transitions > 0.0).mean()
            ),
        },
    }


def contribution_share(
    contributions: pd.DataFrame,
) -> dict[str, Any]:
    if contributions.empty:
        return {
            "row_count": 0,
            "feature_share": {},
        }

    totals = contributions.sum(axis=0).astype(float)
    total = float(totals.sum())
    if total <= 0.0:
        shares = totals * 0.0
    else:
        shares = totals / total

    shares = shares.sort_values(ascending=False)

    return {
        "row_count": len(contributions),
        "feature_share": {
            str(feature): float(value)
            for feature, value in shares.items()
        },
    }


def grouped_feature_share(
    feature_share: dict[str, float],
    *,
    explicit_features: list[str],
) -> dict[str, float]:
    return {
        "tp2_family": float(
            sum(
                value for feature, value in feature_share.items()
                if feature.startswith("tp2__")
            )
        ),
        "dv_pressure_family": float(
            sum(
                value for feature, value in feature_share.items()
                if feature.startswith("dv_pressure__")
            )
        ),
        "pressure_switch_family": float(
            sum(
                value for feature, value in feature_share.items()
                if feature.startswith("pressure_switch__")
            )
        ),
        "explicit_focus_set": float(
            sum(
                feature_share.get(feature, 0.0)
                for feature in explicit_features
            )
        ),
    }


def extreme_run_summary(
    series: pd.Series,
    *,
    threshold: float,
    gap_minutes: int,
) -> dict[str, Any]:
    series = series.sort_index().astype(float)
    mask = series.abs() >= threshold
    positions = np.flatnonzero(mask.to_numpy(dtype=bool))

    if len(positions) == 0:
        return {
            "extreme_row_count": 0,
            "extreme_fraction": 0.0,
            "run_count": 0,
        }

    gap = pd.Timedelta(minutes=gap_minutes)
    runs: list[tuple[int, int]] = []
    start = int(positions[0])
    previous = int(positions[0])

    for position_raw in positions[1:]:
        position = int(position_raw)
        contiguous = (
            position == previous + 1
            and series.index[position] - series.index[previous]
            <= gap
        )
        if contiguous:
            previous = position
            continue

        runs.append((start, previous))
        start = position
        previous = position

    runs.append((start, previous))

    row_lengths = pd.Series(
        [end - start + 1 for start, end in runs],
        dtype=float,
    )
    durations = pd.Series(
        [
            float(
                (
                    series.index[end]
                    - series.index[start]
                )
                / pd.Timedelta(minutes=1)
            )
            for start, end in runs
        ],
        dtype=float,
    )

    def quantiles(values: pd.Series) -> dict[str, float]:
        return {
            "median": float(values.median()),
            "q90": float(
                values.quantile(0.90, interpolation="higher")
            ),
            "q95": float(
                values.quantile(0.95, interpolation="higher")
            ),
            "q99": float(
                values.quantile(0.99, interpolation="higher")
            ),
            "max": float(values.max()),
        }

    return {
        "extreme_row_count": int(mask.sum()),
        "extreme_fraction": float(mask.mean()),
        "first_extreme_timestamp": str(
            series.index[positions[0]]
        ),
        "last_extreme_timestamp": str(
            series.index[positions[-1]]
        ),
        "run_count": len(runs),
        "run_rows": quantiles(row_lengths),
        "run_duration_minutes": quantiles(durations),
    }
