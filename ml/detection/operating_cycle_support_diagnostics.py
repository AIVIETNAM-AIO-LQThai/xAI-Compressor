from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml.detection.regime_pca import REGIME_NAMES


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


def make_support_flags(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    *,
    flag_config: dict[str, dict[str, str]],
    composite_config: dict[str, dict[str, list[str]]],
) -> pd.DataFrame:
    flags = pd.DataFrame(index=frame.index)

    for name, spec in flag_config.items():
        feature = str(spec["feature"])
        direction = str(spec["direction"])

        if direction == "above_train_max":
            threshold = float(train[feature].max())
            flags[name] = frame[feature].astype(float) > threshold
        elif direction == "below_train_min":
            threshold = float(train[feature].min())
            flags[name] = frame[feature].astype(float) < threshold
        else:
            raise ValueError(f"Unknown support direction: {direction}")

    for name, spec in composite_config.items():
        members = [str(item) for item in spec["any_of"]]
        missing = [item for item in members if item not in flags]
        if missing:
            raise ValueError(
                f"Composite {name} references missing flags: {missing}"
            )
        flags[name] = flags[members].any(axis=1)

    return flags.astype(bool)


def prevalence_summary(
    flags: pd.DataFrame,
) -> dict[str, Any]:
    return {
        column: {
            "count": int(flags[column].sum()),
            "fraction": float(flags[column].mean()),
        }
        for column in flags.columns
    }


def router_coverage_summary(
    flags: pd.DataFrame,
    regimes: pd.Series,
) -> dict[str, Any]:
    regimes = regimes.reindex(flags.index)
    result: dict[str, Any] = {}

    for column in flags.columns:
        mask = flags[column]
        flagged_regimes = regimes.loc[mask]

        result[column] = {
            "flagged_rows": int(mask.sum()),
            "regime_distribution_among_flagged": {
                regime: (
                    float((flagged_regimes == regime).mean())
                    if len(flagged_regimes)
                    else 0.0
                )
                for regime in REGIME_NAMES
            },
            "prevalence_within_regime": {
                regime: (
                    float(
                        flags.loc[
                            regimes == regime,
                            column,
                        ].mean()
                    )
                    if int((regimes == regime).sum())
                    else None
                )
                for regime in REGIME_NAMES
            },
        }

    return result


def _score_summary(
    values: pd.Series,
) -> dict[str, Any]:
    values = values.astype(float).dropna()
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
        "mean": float(values.mean()),
    }


def score_association_summary(
    flags: pd.DataFrame,
    scores: pd.Series,
    *,
    threshold: float,
) -> dict[str, Any]:
    scores = scores.reindex(flags.index).astype(float)
    high = scores >= threshold
    total_score = float(scores.sum())

    result: dict[str, Any] = {}

    for column in flags.columns:
        mask = flags[column]
        flagged_scores = scores.loc[mask]

        result[column] = {
            "flagged_score": _score_summary(flagged_scores),
            "flagged_threshold_exceedance_fraction": (
                float((flagged_scores >= threshold).mean())
                if len(flagged_scores)
                else None
            ),
            "fraction_of_all_high_score_rows_with_flag": (
                float(flags.loc[high, column].mean())
                if int(high.sum())
                else None
            ),
            "share_of_total_background_score_mass": (
                float(flagged_scores.sum() / total_score)
                if total_score > 0.0
                else None
            ),
        }

    return result


def joint_pattern_summary(
    atomic_flags: pd.DataFrame,
    scores: pd.Series,
    regimes: pd.Series,
    *,
    threshold: float,
    top_n: int,
) -> dict[str, Any]:
    scores = scores.reindex(atomic_flags.index).astype(float)
    regimes = regimes.reindex(atomic_flags.index)

    names = list(atomic_flags.columns)
    pattern = atomic_flags.apply(
        lambda row: "|".join(
            name for name in names if bool(row[name])
        )
        or "none",
        axis=1,
    )

    frame = pd.DataFrame(
        {
            "pattern": pattern,
            "score": scores,
            "regime": regimes,
        },
        index=atomic_flags.index,
    )

    def summarize(subset: pd.DataFrame) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        total = len(subset)

        grouped = (
            subset.groupby("pattern", sort=False)
            .size()
            .sort_values(ascending=False)
        )

        for pattern_name in grouped.head(top_n).index:
            group = subset.loc[
                subset["pattern"] == pattern_name
            ]
            rows.append(
                {
                    "pattern": str(pattern_name),
                    "count": len(group),
                    "fraction": (
                        float(len(group) / total)
                        if total
                        else 0.0
                    ),
                    "mean_score": float(group["score"].mean()),
                    "high_score_fraction": float(
                        (group["score"] >= threshold).mean()
                    ),
                    "regime_distribution": {
                        regime: float(
                            (group["regime"] == regime).mean()
                        )
                        for regime in REGIME_NAMES
                    },
                }
            )

        return rows

    return {
        "all_background": summarize(frame),
        "high_score_background": summarize(
            frame.loc[frame["score"] >= threshold]
        ),
    }


def run_summary(
    flag: pd.Series,
    *,
    gap_minutes: int,
) -> dict[str, Any]:
    flag = flag.sort_index().astype(bool)
    positions = np.flatnonzero(flag.to_numpy())

    if len(positions) == 0:
        return {
            "flagged_rows": 0,
            "run_count": 0,
        }

    gap = pd.Timedelta(minutes=gap_minutes)
    runs: list[tuple[int, int]] = []

    start = int(positions[0])
    previous = int(positions[0])

    for raw_position in positions[1:]:
        position = int(raw_position)
        contiguous = (
            position == previous + 1
            and flag.index[position] - flag.index[previous]
            <= gap
        )

        if contiguous:
            previous = position
            continue

        runs.append((start, previous))
        start = position
        previous = position

    runs.append((start, previous))

    row_counts = pd.Series(
        [end - start + 1 for start, end in runs],
        dtype=float,
    )
    durations = pd.Series(
        [
            float(
                (
                    flag.index[end] - flag.index[start]
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
            "q90": float(values.quantile(0.90, interpolation="higher")),
            "q95": float(values.quantile(0.95, interpolation="higher")),
            "q99": float(values.quantile(0.99, interpolation="higher")),
            "max": float(values.max()),
        }

    return {
        "flagged_rows": int(flag.sum()),
        "flagged_fraction": float(flag.mean()),
        "first_timestamp": str(flag.index[positions[0]]),
        "last_timestamp": str(flag.index[positions[-1]]),
        "run_count": len(runs),
        "run_rows": quantiles(row_counts),
        "run_duration_minutes": quantiles(durations),
    }


def quality_summary(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for column in ["sample_count", "coverage_ratio"]:
        values = frame[column].astype(float)
        result[column] = {
            "count": len(values),
            "min": float(values.min()),
            "median": float(values.median()),
            "q01": float(values.quantile(0.01)),
            "q99": float(values.quantile(0.99)),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }

    return result
