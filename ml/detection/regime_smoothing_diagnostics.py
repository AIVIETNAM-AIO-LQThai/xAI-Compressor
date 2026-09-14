from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.detection.regime_pca import REGIME_NAMES


def transition_reset_ewma(
    scores: pd.Series,
    regimes: pd.Series,
    *,
    alpha: float,
    reset_gap_minutes: int,
) -> pd.Series:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1].")

    scores, regimes = scores.align(regimes, join="inner")
    scores = scores.sort_index()
    regimes = regimes.reindex(scores.index)

    if not isinstance(scores.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex.")

    reset_gap = pd.Timedelta(minutes=reset_gap_minutes)
    values = scores.to_numpy(dtype=float)
    labels = regimes.astype(str).to_numpy()

    output = np.empty(len(scores), dtype=float)
    previous_time: pd.Timestamp | None = None
    previous_value: float | None = None
    previous_regime: str | None = None

    for i, (timestamp, value, regime) in enumerate(
        zip(scores.index, values, labels, strict=True)
    ):
        reset = (
            previous_time is None
            or timestamp - previous_time > reset_gap
            or previous_regime is None
            or regime != previous_regime
        )

        if reset or previous_value is None:
            current = float(value)
        else:
            current = float(
                alpha * value
                + (1.0 - alpha) * previous_value
            )

        output[i] = current
        previous_time = timestamp
        previous_value = current
        previous_regime = regime

    return pd.Series(
        output,
        index=scores.index,
        name="transition_reset_ewma",
    )


def transition_flags(
    regimes: pd.Series,
    *,
    reset_gap_minutes: int,
) -> pd.Series:
    regimes = regimes.sort_index()
    if not isinstance(regimes.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex.")

    flags = np.zeros(len(regimes), dtype=bool)
    gap = pd.Timedelta(minutes=reset_gap_minutes)

    for i in range(1, len(regimes)):
        contiguous = regimes.index[i] - regimes.index[i - 1] <= gap
        flags[i] = (
            contiguous
            and str(regimes.iloc[i]) != str(regimes.iloc[i - 1])
        )

    return pd.Series(
        flags,
        index=regimes.index,
        name="is_transition",
        dtype=bool,
    )


def transition_carryover_table(
    raw_scores: pd.Series,
    baseline_ewma: pd.Series,
    reset_ewma: pd.Series,
    regimes: pd.Series,
    *,
    reset_gap_minutes: int,
) -> pd.DataFrame:
    frame = pd.concat(
        [
            raw_scores.rename("raw_score"),
            baseline_ewma.rename("baseline_ewma"),
            reset_ewma.rename("reset_ewma"),
            regimes.rename("regime"),
        ],
        axis=1,
    ).sort_index()

    flags = transition_flags(
        frame["regime"],
        reset_gap_minutes=reset_gap_minutes,
    )

    previous_regime = frame["regime"].shift(1)
    table = frame.loc[flags].copy()
    table["previous_regime"] = previous_regime.loc[flags].astype(str)
    table["current_regime"] = table["regime"].astype(str)
    table["carryover_delta"] = (
        table["baseline_ewma"] - table["reset_ewma"]
    )
    table["absolute_carryover_delta"] = table[
        "carryover_delta"
    ].abs()
    table["transition_pair"] = (
        table["previous_regime"]
        + " -> "
        + table["current_regime"]
    )

    return table.drop(columns=["regime"])


def transition_pair_summary(
    table: pd.DataFrame,
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for pair, group in table.groupby("transition_pair"):
        values = group["absolute_carryover_delta"].astype(float)
        signed = group["carryover_delta"].astype(float)

        result[str(pair)] = {
            "count": len(group),
            "mean_absolute_delta": float(values.mean()),
            "median_absolute_delta": float(values.median()),
            "q90_absolute_delta": float(
                values.quantile(0.90, interpolation="higher")
            ),
            "q95_absolute_delta": float(
                values.quantile(0.95, interpolation="higher")
            ),
            "mean_signed_delta": float(signed.mean()),
        }

    return result


def conditional_tail_summary(
    raw_scores: pd.Series,
    baseline_ewma: pd.Series,
    reset_ewma: pd.Series,
    regimes: pd.Series,
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    sequences = {
        "raw": raw_scores,
        "baseline_ewma": baseline_ewma,
        "transition_reset_ewma": reset_ewma,
    }

    for regime in REGIME_NAMES:
        mask = regimes == regime
        result[regime] = {}

        for name, series in sequences.items():
            values = series.loc[mask].astype(float)
            result[regime][name] = {
                "count": len(values),
                "q99": float(
                    values.quantile(
                        0.99,
                        interpolation="higher",
                    )
                ),
                "q995": float(
                    values.quantile(
                        0.995,
                        interpolation="higher",
                    )
                ),
                "max": float(values.max()),
            }

    return result


def tail_composition(
    scores: pd.Series,
    regimes: pd.Series,
    *,
    quantile: float,
) -> dict[str, Any]:
    cutoff = float(
        scores.quantile(
            quantile,
            interpolation="higher",
        )
    )
    mask = scores >= cutoff
    tail_regimes = regimes.loc[mask]

    return {
        "cutoff": cutoff,
        "tail_count": int(mask.sum()),
        "regime_fraction": {
            regime: float((tail_regimes == regime).mean())
            for regime in REGIME_NAMES
        },
    }


def gap_aware_autocorrelation(
    series: pd.Series,
    *,
    lags: list[int],
    reset_gap_minutes: int,
) -> dict[str, Any]:
    series = series.sort_index().astype(float)
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex.")

    gap = pd.Timedelta(minutes=reset_gap_minutes)
    n = len(series)
    values = series.to_numpy()

    output: dict[str, Any] = {}

    for lag in lags:
        left_values: list[float] = []
        right_values: list[float] = []

        for i in range(n - lag):
            if series.index[i + lag] - series.index[i] > gap * lag:
                continue
            valid = True
            for j in range(i + 1, i + lag + 1):
                if series.index[j] - series.index[j - 1] > gap:
                    valid = False
                    break
            if valid:
                left_values.append(float(values[i]))
                right_values.append(float(values[i + lag]))

        if len(left_values) < 2:
            corr = None
        else:
            x = np.asarray(left_values, dtype=float)
            y = np.asarray(right_values, dtype=float)
            if np.std(x) == 0.0 or np.std(y) == 0.0:
                corr = None
            else:
                corr = float(np.corrcoef(x, y)[0, 1])

        output[str(lag)] = {
            "pair_count": len(left_values),
            "correlation": corr,
        }

    return output


def bootstrap_smoothed_threshold(
    scores: pd.Series,
    regimes: pd.Series,
    *,
    threshold_quantile: float,
    replicates: int,
    block_lengths: list[int],
    seed: int,
    confidence: float,
    variant: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    scores, regimes = scores.align(regimes, join="inner")
    values = scores.to_numpy(dtype=float)
    labels = regimes.astype(str).to_numpy()
    n = len(values)

    rng = np.random.default_rng(seed)
    alpha = 1.0 - confidence
    rows: list[dict[str, Any]] = []
    result: dict[str, Any] = {}

    for block_length in block_lengths:
        blocks_needed = int(np.ceil(n / block_length))
        offsets = np.arange(block_length, dtype=int)
        thresholds = np.empty(replicates, dtype=float)

        for replicate in range(replicates):
            starts = rng.integers(
                0,
                n,
                size=blocks_needed,
            )
            indices = (
                starts[:, None] + offsets[None, :]
            ) % n
            indices = indices.ravel()[:n]

            sample = values[indices]
            sample_labels = labels[indices]
            threshold = float(
                np.quantile(
                    sample,
                    threshold_quantile,
                    method="higher",
                )
            )
            thresholds[replicate] = threshold

            row: dict[str, Any] = {
                "variant": variant,
                "block_length": int(block_length),
                "replicate": int(replicate),
                "threshold": threshold,
            }
            for regime in REGIME_NAMES:
                row[f"fraction__{regime}"] = float(
                    np.mean(sample_labels == regime)
                )
            rows.append(row)

        lower = float(np.quantile(thresholds, alpha / 2.0))
        median = float(np.median(thresholds))
        upper = float(
            np.quantile(
                thresholds,
                1.0 - alpha / 2.0,
            )
        )
        relative = (
            float((upper - lower) / abs(median))
            if median != 0.0
            else None
        )

        result[f"circular_block_{block_length}"] = {
            "lower": lower,
            "median": median,
            "upper": upper,
            "relative_95_width": relative,
            "minimum": float(thresholds.min()),
            "maximum": float(thresholds.max()),
            "replicates": int(replicates),
        }

    return result, pd.DataFrame(rows)
