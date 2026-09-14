from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.detection.regime_pca import (
    REGIME_NAMES,
    RegimePCADetector,
)


def distribution_summary(
    series: pd.Series,
) -> dict[str, float | int]:
    values = series.astype(float).dropna()
    if values.empty:
        return {"count": 0}

    return {
        "count": len(values),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median": float(values.median()),
        "iqr": float(values.quantile(0.75) - values.quantile(0.25)),
        "q90": float(values.quantile(0.90, interpolation="higher")),
        "q95": float(values.quantile(0.95, interpolation="higher")),
        "q99": float(values.quantile(0.99, interpolation="higher")),
        "q995": float(values.quantile(0.995, interpolation="higher")),
        "max": float(values.max()),
    }


def score_distribution_by_regime(
    scores: pd.Series,
    regimes: pd.Series,
) -> dict[str, Any]:
    scores, regimes = scores.align(regimes, join="inner")
    result: dict[str, Any] = {
        "pooled": distribution_summary(scores),
        "by_regime": {},
    }

    for regime in REGIME_NAMES:
        mask = regimes == regime
        result["by_regime"][regime] = distribution_summary(
            scores.loc[mask]
        )

    return result


def occupancy_comparison(
    train_regimes: pd.Series,
    calibration_regimes: pd.Series,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    train_n = len(train_regimes)
    calibration_n = len(calibration_regimes)

    for regime in REGIME_NAMES:
        train_count = int((train_regimes == regime).sum())
        calibration_count = int((calibration_regimes == regime).sum())
        train_fraction = train_count / train_n
        calibration_fraction = calibration_count / calibration_n

        result[regime] = {
            "train_count": train_count,
            "train_fraction": float(train_fraction),
            "calibration_count": calibration_count,
            "calibration_fraction": float(calibration_fraction),
            "fraction_change": float(
                calibration_fraction - train_fraction
            ),
        }

    return result


def pooled_tail_composition(
    scores: pd.Series,
    regimes: pd.Series,
    *,
    quantiles: list[float],
) -> dict[str, Any]:
    scores, regimes = scores.align(regimes, join="inner")
    ordinary = {
        regime: float((regimes == regime).mean())
        for regime in REGIME_NAMES
    }

    result: dict[str, Any] = {}

    for quantile in quantiles:
        cutoff = float(
            scores.quantile(
                quantile,
                interpolation="higher",
            )
        )
        mask = scores >= cutoff
        tail_regimes = regimes.loc[mask]
        tail_count = int(mask.sum())

        composition: dict[str, Any] = {}
        for regime in REGIME_NAMES:
            tail_fraction = (
                float((tail_regimes == regime).mean())
                if tail_count
                else 0.0
            )
            ordinary_fraction = ordinary[regime]
            enrichment = (
                tail_fraction / ordinary_fraction
                if ordinary_fraction > 0.0
                else None
            )
            composition[regime] = {
                "tail_fraction": tail_fraction,
                "calibration_fraction": ordinary_fraction,
                "enrichment": (
                    float(enrichment)
                    if enrichment is not None
                    else None
                ),
            }

        result[str(quantile)] = {
            "cutoff": cutoff,
            "tail_count": tail_count,
            "composition": composition,
        }

    return result


def segment_ids(
    index: pd.DatetimeIndex,
    *,
    reset_gap_minutes: int,
) -> np.ndarray:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex.")

    n = len(index)
    if n == 0:
        return np.empty(0, dtype=int)

    reset_gap = pd.Timedelta(minutes=reset_gap_minutes)
    ids = np.zeros(n, dtype=int)
    segment = 0

    for position in range(1, n):
        if index[position] - index[position - 1] > reset_gap:
            segment += 1
        ids[position] = segment

    return ids


def transition_flags(
    regimes: pd.Series,
    *,
    reset_gap_minutes: int,
) -> pd.Series:
    if not isinstance(regimes.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex.")

    regimes = regimes.sort_index()
    segments = segment_ids(
        regimes.index,
        reset_gap_minutes=reset_gap_minutes,
    )
    values = regimes.astype(str).to_numpy()

    flags = np.zeros(len(regimes), dtype=bool)
    if len(regimes) > 1:
        same_segment = segments[1:] == segments[:-1]
        changed = values[1:] != values[:-1]
        flags[1:] = same_segment & changed

    return pd.Series(
        flags,
        index=regimes.index,
        name="is_transition",
        dtype=bool,
    )


def near_transition_flags(
    transitions: pd.Series,
    *,
    radius: int,
    reset_gap_minutes: int,
) -> pd.Series:
    if radius < 0:
        raise ValueError("radius must be non-negative.")

    transitions = transitions.sort_index().astype(bool)
    segments = segment_ids(
        transitions.index,
        reset_gap_minutes=reset_gap_minutes,
    )
    values = transitions.to_numpy()
    near = np.zeros(len(transitions), dtype=bool)
    positions = np.flatnonzero(values)

    for position in positions:
        left = max(0, position - radius)
        right = min(len(transitions), position + radius + 1)
        for candidate in range(left, right):
            if segments[candidate] == segments[position]:
                near[candidate] = True

    return pd.Series(
        near,
        index=transitions.index,
        name=f"near_transition_{radius}",
        dtype=bool,
    )


def router_margin_frame(
    calibration: pd.DataFrame,
    train: pd.DataFrame,
    detector: RegimePCADetector,
) -> pd.DataFrame:
    current = calibration[detector.current_feature].astype(float)
    pressure = calibration[detector.pressure_feature].astype(float)

    current_std = float(
        train[detector.current_feature].astype(float).std(ddof=1)
    )
    pressure_std = float(
        train[detector.pressure_feature].astype(float).std(ddof=1)
    )

    if current_std <= 0.0 or pressure_std <= 0.0:
        raise ValueError("Router training standard deviation must be positive.")

    current_margin = current - detector.current_threshold
    pressure_margin = pressure - detector.pressure_threshold

    normalized_current = current_margin / current_std
    normalized_pressure = pressure_margin / pressure_std

    return pd.DataFrame(
        {
            "current_value": current,
            "pressure_value": pressure,
            "current_margin": current_margin,
            "pressure_margin": pressure_margin,
            "normalized_current_margin": normalized_current,
            "normalized_pressure_margin": normalized_pressure,
            "router_boundary_distance": np.minimum(
                normalized_current.abs(),
                normalized_pressure.abs(),
            ),
        },
        index=calibration.index,
    )


def _jump_summary(values: pd.Series) -> dict[str, float | int]:
    clean = values.astype(float).dropna().abs()
    if clean.empty:
        return {"count": 0}

    return {
        "count": len(clean),
        "median": float(clean.median()),
        "q90": float(clean.quantile(0.90, interpolation="higher")),
        "q95": float(clean.quantile(0.95, interpolation="higher")),
        "q99": float(clean.quantile(0.99, interpolation="higher")),
    }


def transition_score_diagnostics(
    raw_scores: pd.Series,
    smoothed_scores: pd.Series,
    regimes: pd.Series,
    *,
    reset_gap_minutes: int,
    window_rows: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    raw_scores, smoothed_scores = raw_scores.align(
        smoothed_scores,
        join="inner",
    )
    raw_scores, regimes = raw_scores.align(regimes, join="inner")
    smoothed_scores = smoothed_scores.reindex(raw_scores.index)
    regimes = regimes.reindex(raw_scores.index)

    transitions = transition_flags(
        regimes,
        reset_gap_minutes=reset_gap_minutes,
    )
    near1 = near_transition_flags(
        transitions,
        radius=1,
        reset_gap_minutes=reset_gap_minutes,
    )
    near2 = near_transition_flags(
        transitions,
        radius=2,
        reset_gap_minutes=reset_gap_minutes,
    )

    segments = segment_ids(
        raw_scores.index,
        reset_gap_minutes=reset_gap_minutes,
    )
    raw_jump = raw_scores.diff()
    smooth_jump = smoothed_scores.diff()

    same_segment = np.ones(len(raw_scores), dtype=bool)
    if len(raw_scores):
        same_segment[0] = False
    if len(raw_scores) > 1:
        same_segment[1:] = segments[1:] == segments[:-1]

    raw_jump.loc[~same_segment] = np.nan
    smooth_jump.loc[~same_segment] = np.nan

    diagnostics = {
        "transition_count": int(transitions.sum()),
        "raw_score_absolute_jump": {
            "transition": _jump_summary(raw_jump.loc[transitions]),
            "non_transition": _jump_summary(
                raw_jump.loc[(~transitions) & same_segment]
            ),
        },
        "ewma_score_absolute_jump": {
            "transition": _jump_summary(smooth_jump.loc[transitions]),
            "non_transition": _jump_summary(
                smooth_jump.loc[(~transitions) & same_segment]
            ),
        },
    }

    rows: list[dict[str, Any]] = []
    transition_positions = np.flatnonzero(transitions.to_numpy())

    for transition_position in transition_positions:
        for offset in range(-window_rows, window_rows + 1):
            position = transition_position + offset
            if position < 0 or position >= len(raw_scores):
                continue
            if segments[position] != segments[transition_position]:
                continue

            timestamp = raw_scores.index[position]
            rows.append(
                {
                    "transition_timestamp": raw_scores.index[
                        transition_position
                    ],
                    "timestamp": timestamp,
                    "offset": int(offset),
                    "regime": str(regimes.iloc[position]),
                    "raw_score": float(raw_scores.iloc[position]),
                    "ewma_score": float(smoothed_scores.iloc[position]),
                }
            )

    windows = pd.DataFrame(rows)

    return diagnostics, windows, transitions, near1, near2


def tail_transition_proximity(
    smoothed_scores: pd.Series,
    transitions: pd.Series,
    near1: pd.Series,
    near2: pd.Series,
    *,
    quantiles: list[float],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    baseline = {
        "at_transition": float(transitions.mean()),
        "within_1": float(near1.mean()),
        "within_2": float(near2.mean()),
    }

    for quantile in quantiles:
        cutoff = float(
            smoothed_scores.quantile(
                quantile,
                interpolation="higher",
            )
        )
        tail = smoothed_scores >= cutoff
        tail_count = int(tail.sum())

        measures = {
            "at_transition": float(transitions.loc[tail].mean()),
            "within_1": float(near1.loc[tail].mean()),
            "within_2": float(near2.loc[tail].mean()),
        }

        enrichment = {}
        for key, value in measures.items():
            base = baseline[key]
            enrichment[key] = (
                float(value / base)
                if base > 0.0
                else None
            )

        result[str(quantile)] = {
            "cutoff": cutoff,
            "tail_count": tail_count,
            "baseline_fraction": baseline,
            "tail_fraction": measures,
            "enrichment": enrichment,
            "not_within_2_fraction": float(
                (~near2.loc[tail]).mean()
            ),
        }

    return result


def regime_run_summary(
    regimes: pd.Series,
    *,
    reset_gap_minutes: int,
) -> dict[str, Any]:
    regimes = regimes.sort_index()
    segments = segment_ids(
        regimes.index,
        reset_gap_minutes=reset_gap_minutes,
    )
    labels = regimes.astype(str).to_numpy()

    runs: dict[str, list[int]] = {
        regime: [] for regime in REGIME_NAMES
    }

    if len(regimes):
        start = 0
        for position in range(1, len(regimes) + 1):
            boundary = (
                position == len(regimes)
                or segments[position] != segments[position - 1]
                or labels[position] != labels[position - 1]
            )
            if boundary:
                label = labels[start]
                runs[label].append(position - start)
                start = position

    result: dict[str, Any] = {}
    for regime in REGIME_NAMES:
        lengths = np.asarray(runs[regime], dtype=float)
        if len(lengths) == 0:
            result[regime] = {"number_of_runs": 0}
            continue

        result[regime] = {
            "number_of_runs": len(lengths),
            "mean_run_length": float(lengths.mean()),
            "median_run_length": float(np.median(lengths)),
            "q90_run_length": float(
                np.quantile(lengths, 0.90, method="higher")
            ),
            "max_run_length": int(lengths.max()),
        }

    return result


def gap_aware_autocorrelation(
    series: pd.Series,
    *,
    lags: list[int],
    reset_gap_minutes: int,
    regimes: pd.Series | None = None,
) -> dict[str, Any]:
    series = series.sort_index().astype(float)
    if regimes is not None:
        regimes = regimes.reindex(series.index)

    segments = segment_ids(
        series.index,
        reset_gap_minutes=reset_gap_minutes,
    )
    values = series.to_numpy()
    labels = (
        regimes.astype(str).to_numpy()
        if regimes is not None
        else None
    )

    def one_group(target: str | None) -> dict[str, Any]:
        output: dict[str, Any] = {}
        n = len(values)

        for lag in lags:
            if lag <= 0:
                raise ValueError("lags must be positive.")
            if lag >= n:
                output[str(lag)] = {
                    "pair_count": 0,
                    "correlation": None,
                }
                continue

            left = np.arange(0, n - lag)
            right = left + lag
            valid = segments[left] == segments[right]

            if target is not None and labels is not None:
                valid &= labels[left] == target
                valid &= labels[right] == target

            x = values[left[valid]]
            y = values[right[valid]]

            if len(x) < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
                correlation = None
            else:
                correlation = float(np.corrcoef(x, y)[0, 1])

            output[str(lag)] = {
                "pair_count": len(x),
                "correlation": correlation,
            }

        return output

    result: dict[str, Any] = {
        "pooled": one_group(None),
        "by_regime": {},
    }

    if regimes is not None:
        for regime in REGIME_NAMES:
            result["by_regime"][regime] = one_group(regime)

    return result


def bootstrap_threshold_composition(
    smoothed_scores: pd.Series,
    regimes: pd.Series,
    *,
    threshold_quantile: float,
    bootstrap_replicates: int,
    block_lengths: list[int],
    seed: int,
    confidence: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    smoothed_scores, regimes = smoothed_scores.align(regimes, join="inner")
    values = smoothed_scores.to_numpy(dtype=float)
    labels = regimes.astype(str).to_numpy()

    if len(values) == 0:
        raise ValueError("No calibration values.")
    if not np.isfinite(values).all():
        raise ValueError("Calibration values must be finite.")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive.")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1).")

    rng = np.random.default_rng(seed)
    n = len(values)
    alpha = 1.0 - confidence

    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    for block_length in block_lengths:
        if block_length <= 0:
            raise ValueError("block lengths must be positive.")

        blocks_needed = int(np.ceil(n / block_length))
        offsets = np.arange(block_length, dtype=int)

        thresholds = np.empty(bootstrap_replicates, dtype=float)

        for replicate in range(bootstrap_replicates):
            starts = rng.integers(
                0,
                n,
                size=blocks_needed,
            )
            indices = (
                starts[:, None] + offsets[None, :]
            ) % n
            indices = indices.ravel()[:n]

            sampled_values = values[indices]
            sampled_labels = labels[indices]

            threshold = float(
                np.quantile(
                    sampled_values,
                    threshold_quantile,
                    method="higher",
                )
            )
            thresholds[replicate] = threshold

            row: dict[str, Any] = {
                "block_length": int(block_length),
                "replicate": int(replicate),
                "threshold": threshold,
            }
            for regime in REGIME_NAMES:
                row[f"fraction__{regime}"] = float(
                    np.mean(sampled_labels == regime)
                )
            rows.append(row)

        lower = float(np.quantile(thresholds, alpha / 2.0))
        median = float(np.median(thresholds))
        upper = float(
            np.quantile(thresholds, 1.0 - alpha / 2.0)
        )
        relative_width = (
            float((upper - lower) / abs(median))
            if median != 0.0
            else None
        )

        block_rows = pd.DataFrame(
            row for row in rows
            if row["block_length"] == block_length
        )

        correlations = {}
        for regime in REGIME_NAMES:
            column = f"fraction__{regime}"
            correlation = block_rows["threshold"].corr(
                block_rows[column],
                method="spearman",
            )
            correlations[regime] = (
                float(correlation)
                if pd.notna(correlation)
                else None
            )

        summary[f"circular_block_{block_length}"] = {
            "block_length": int(block_length),
            "replicates": int(bootstrap_replicates),
            "lower": lower,
            "median": median,
            "upper": upper,
            "relative_95_width": relative_width,
            "minimum": float(thresholds.min()),
            "maximum": float(thresholds.max()),
            "regime_fraction_threshold_spearman": correlations,
        }

    return summary, pd.DataFrame(rows)


def conditional_thresholds(
    smoothed_scores: pd.Series,
    regimes: pd.Series,
    *,
    threshold_quantile: float,
) -> dict[str, Any]:
    smoothed_scores, regimes = smoothed_scores.align(regimes, join="inner")
    result: dict[str, Any] = {}

    for regime in REGIME_NAMES:
        values = smoothed_scores.loc[regimes == regime]
        if values.empty:
            result[regime] = {
                "count": 0,
                "diagnostic_only": True,
            }
            continue

        threshold = float(
            values.quantile(
                threshold_quantile,
                interpolation="higher",
            )
        )
        result[regime] = {
            "count": len(values),
            "q995": threshold,
            "rows_at_or_above_q995": int((values >= threshold).sum()),
            "diagnostic_only": True,
        }

    return result
