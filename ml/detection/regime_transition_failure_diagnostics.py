from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from ml.detection.alerts import AlertEpisode
from ml.detection.regime_pca import REGIME_NAMES


def distribution_summary(
    series: pd.Series,
    *,
    threshold: float,
) -> dict[str, float | int]:
    values = series.astype(float).dropna()
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


def calibration_test_shift(
    calibration_scores: pd.Series,
    calibration_regimes: pd.Series,
    test_scores: pd.Series,
    test_regimes: pd.Series,
    *,
    threshold: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pooled": {},
        "by_regime": {},
    }

    cal_summary = distribution_summary(
        calibration_scores,
        threshold=threshold,
    )
    test_summary = distribution_summary(
        test_scores,
        threshold=threshold,
    )
    result["pooled"] = {
        "calibration": cal_summary,
        "test": test_summary,
        "test_to_calibration_q99_ratio": (
            float(test_summary["q99"] / cal_summary["q99"])
            if cal_summary.get("q99", 0.0) != 0.0
            else None
        ),
        "test_to_calibration_q995_ratio": (
            float(test_summary["q995"] / cal_summary["q995"])
            if cal_summary.get("q995", 0.0) != 0.0
            else None
        ),
    }

    for regime in REGIME_NAMES:
        cal = distribution_summary(
            calibration_scores.loc[
                calibration_regimes == regime
            ],
            threshold=threshold,
        )
        test = distribution_summary(
            test_scores.loc[test_regimes == regime],
            threshold=threshold,
        )
        result["by_regime"][regime] = {
            "calibration": cal,
            "test": test,
            "test_to_calibration_q99_ratio": (
                float(test["q99"] / cal["q99"])
                if cal.get("q99", 0.0) != 0.0
                else None
            ),
            "test_to_calibration_q995_ratio": (
                float(test["q995"] / cal["q995"])
                if cal.get("q995", 0.0) != 0.0
                else None
            ),
        }

    return result


def transition_flags(
    regimes: pd.Series,
    *,
    reset_gap_minutes: int,
) -> pd.Series:
    regimes = regimes.sort_index()
    if not isinstance(regimes.index, pd.DatetimeIndex):
        raise TypeError("Expected a DatetimeIndex.")

    values = regimes.astype(str).to_numpy()
    flags = np.zeros(len(regimes), dtype=bool)
    gap = pd.Timedelta(minutes=reset_gap_minutes)

    for i in range(1, len(regimes)):
        contiguous = (
            regimes.index[i] - regimes.index[i - 1]
            <= gap
        )
        flags[i] = (
            contiguous
            and values[i] != values[i - 1]
        )

    return pd.Series(
        flags,
        index=regimes.index,
        name="is_transition",
        dtype=bool,
    )


def occupancy_transition_summary(
    regimes: pd.Series,
    *,
    reset_gap_minutes: int,
) -> dict[str, Any]:
    regimes = regimes.sort_index()
    flags = transition_flags(
        regimes,
        reset_gap_minutes=reset_gap_minutes,
    )

    if len(regimes) >= 2:
        observed_days = float(
            (regimes.index.max() - regimes.index.min())
            / pd.Timedelta(days=1)
        )
    else:
        observed_days = 0.0

    return {
        "rows": len(regimes),
        "regime_fraction": {
            regime: float((regimes == regime).mean())
            for regime in REGIME_NAMES
        },
        "transition_count": int(flags.sum()),
        "transition_fraction": float(flags.mean()),
        "observed_span_days": observed_days,
        "transitions_per_observed_day": (
            float(flags.sum() / observed_days)
            if observed_days > 0.0
            else None
        ),
    }


def persistence_window_frame(
    scores: pd.Series,
    regimes: pd.Series,
    *,
    threshold: float,
    required_hits: int,
    window_bins: int,
    reset_gap_minutes: int,
) -> pd.DataFrame:
    scores, regimes = scores.align(regimes, join="inner")
    scores = scores.sort_index()
    regimes = regimes.reindex(scores.index)

    gap = pd.Timedelta(minutes=reset_gap_minutes)
    history: deque[tuple[bool, str]] = deque(
        maxlen=window_bins
    )

    previous_time: pd.Timestamp | None = None
    rows: list[dict[str, Any]] = []

    for timestamp, score, regime in zip(
        scores.index,
        scores.to_numpy(dtype=float),
        regimes.astype(str).to_numpy(),
        strict=True,
    ):
        if (
            previous_time is None
            or timestamp - previous_time > gap
        ):
            history.clear()

        hit = bool(score >= threshold)
        history.append((hit, regime))

        hits = int(sum(item[0] for item in history))
        labels = [item[1] for item in history]
        unique_regimes = len(set(labels))
        regime_changes = int(
            sum(
                labels[i] != labels[i - 1]
                for i in range(1, len(labels))
            )
        )
        alert = hits >= required_hits

        rows.append(
            {
                "timestamp": timestamp,
                "score": float(score),
                "threshold_hit": hit,
                "alert": bool(alert),
                "history_hits": hits,
                "history_length": len(history),
                "history_unique_regimes": int(unique_regimes),
                "history_regime_changes": regime_changes,
                "history_spans_regimes": bool(
                    unique_regimes > 1
                ),
                "current_regime": regime,
            }
        )

        previous_time = timestamp

    return pd.DataFrame(rows).set_index("timestamp")


def near_transition_flags(
    transitions: pd.Series,
    *,
    radius: int,
    reset_gap_minutes: int,
) -> pd.Series:
    transitions = transitions.sort_index().astype(bool)
    gap = pd.Timedelta(minutes=reset_gap_minutes)
    near = np.zeros(len(transitions), dtype=bool)
    positions = np.flatnonzero(
        transitions.to_numpy(dtype=bool)
    )

    for position in positions:
        for candidate in range(
            max(0, position - radius),
            min(len(transitions), position + radius + 1),
        ):
            left = min(position, candidate)
            right = max(position, candidate)
            valid = True
            for i in range(left + 1, right + 1):
                if (
                    transitions.index[i]
                    - transitions.index[i - 1]
                    > gap
                ):
                    valid = False
                    break
            if valid:
                near[candidate] = True

    return pd.Series(
        near,
        index=transitions.index,
        name=f"near_transition_{radius}",
        dtype=bool,
    )


def relevant_episode_flags(
    episodes: list[AlertEpisode],
    incidents: list[dict[str, Any]],
    *,
    early_warning_hours: int,
) -> list[bool]:
    early = pd.Timedelta(hours=early_warning_hours)
    result: list[bool] = []

    for episode in episodes:
        relevant = False
        for incident in incidents:
            start = pd.Timestamp(incident["start"])
            end = pd.Timestamp(incident["end"])
            relevant_start = start - early
            if (
                episode.start <= end
                and episode.end >= relevant_start
            ):
                relevant = True
                break
        result.append(relevant)

    return result


def episode_diagnostics(
    episodes: list[AlertEpisode],
    persistence: pd.DataFrame,
    transitions: pd.Series,
    incidents: list[dict[str, Any]],
    *,
    early_warning_hours: int,
    reset_gap_minutes: int,
) -> pd.DataFrame:
    relevant = relevant_episode_flags(
        episodes,
        incidents,
        early_warning_hours=early_warning_hours,
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

    rows: list[dict[str, Any]] = []

    for index, (episode, is_relevant) in enumerate(
        zip(episodes, relevant, strict=True)
    ):
        start = episode.start
        row = persistence.loc[start]
        duration_minutes = float(
            (episode.end - episode.start)
            / pd.Timedelta(minutes=1)
        )

        rows.append(
            {
                "episode_index": int(index),
                "start": episode.start,
                "end": episode.end,
                "alert_bins": int(episode.alert_bins),
                "duration_minutes": duration_minutes,
                "is_relevant": bool(is_relevant),
                "is_false": bool(not is_relevant),
                "start_regime": str(row["current_regime"]),
                "start_score": float(row["score"]),
                "start_score_threshold_ratio": float(
                    row["score"] / row["score"] * 0.0
                ),
                "history_hits": int(row["history_hits"]),
                "history_length": int(row["history_length"]),
                "history_unique_regimes": int(
                    row["history_unique_regimes"]
                ),
                "history_regime_changes": int(
                    row["history_regime_changes"]
                ),
                "history_spans_regimes": bool(
                    row["history_spans_regimes"]
                ),
                "start_at_transition": bool(
                    transitions.loc[start]
                ),
                "start_near_transition_1": bool(
                    near1.loc[start]
                ),
                "start_near_transition_2": bool(
                    near2.loc[start]
                ),
            }
        )

    return pd.DataFrame(rows)


def add_threshold_ratio(
    episodes: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    episodes = episodes.copy()
    if not episodes.empty:
        episodes["start_score_threshold_ratio"] = (
            episodes["start_score"].astype(float)
            / float(threshold)
        )
    return episodes


def _quantile_summary(
    values: pd.Series,
) -> dict[str, Any]:
    clean = values.astype(float).dropna()
    if clean.empty:
        return {"count": 0}
    return {
        "count": len(clean),
        "median": float(clean.median()),
        "q90": float(
            clean.quantile(0.90, interpolation="higher")
        ),
        "q95": float(
            clean.quantile(0.95, interpolation="higher")
        ),
        "q99": float(
            clean.quantile(0.99, interpolation="higher")
        ),
        "max": float(clean.max()),
    }


def episode_group_summary(
    frame: pd.DataFrame,
) -> dict[str, Any]:
    if frame.empty:
        return {"count": 0}

    return {
        "count": len(frame),
        "history_spans_regimes_fraction": float(
            frame["history_spans_regimes"].mean()
        ),
        "start_at_transition_fraction": float(
            frame["start_at_transition"].mean()
        ),
        "start_near_transition_1_fraction": float(
            frame["start_near_transition_1"].mean()
        ),
        "start_near_transition_2_fraction": float(
            frame["start_near_transition_2"].mean()
        ),
        "alert_bins": _quantile_summary(
            frame["alert_bins"]
        ),
        "duration_minutes": _quantile_summary(
            frame["duration_minutes"]
        ),
        "start_score_threshold_ratio": _quantile_summary(
            frame["start_score_threshold_ratio"]
        ),
    }


def fragmentation_summary(
    episodes: list[AlertEpisode],
) -> dict[str, Any]:
    if not episodes:
        return {"count": 0}

    bins = pd.Series(
        [episode.alert_bins for episode in episodes],
        dtype=float,
    )
    durations = pd.Series(
        [
            float(
                (episode.end - episode.start)
                / pd.Timedelta(minutes=1)
            )
            for episode in episodes
        ],
        dtype=float,
    )

    starts = [
        episode.start
        for episode in episodes
    ]
    ends = [
        episode.end
        for episode in episodes
    ]
    gaps = pd.Series(
        [
            float(
                (starts[i] - ends[i - 1])
                / pd.Timedelta(minutes=1)
            )
            for i in range(1, len(episodes))
        ],
        dtype=float,
    )

    return {
        "count": len(episodes),
        "alert_bins": _quantile_summary(bins),
        "duration_minutes": _quantile_summary(durations),
        "inter_episode_gap_minutes": _quantile_summary(gaps),
        "one_bin_fraction": float((bins <= 1).mean()),
        "up_to_2_bins_fraction": float((bins <= 2).mean()),
        "up_to_3_bins_fraction": float((bins <= 3).mean()),
    }
