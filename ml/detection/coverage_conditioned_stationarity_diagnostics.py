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
            flags[name] = (
                frame[feature].astype(float)
                > float(train[feature].max())
            )
        elif direction == "below_train_min":
            flags[name] = (
                frame[feature].astype(float)
                < float(train[feature].min())
            )
        else:
            raise ValueError(
                f"Unknown support direction: {direction}"
            )

    for name, spec in composite_config.items():
        members = [str(item) for item in spec["any_of"]]
        flags[name] = flags[members].any(axis=1)

    return flags.astype(bool)


def coverage_strata(frame: pd.DataFrame) -> pd.Series:
    coverage = frame["coverage_ratio"].astype(float)
    values = np.select(
        [
            coverage >= 1.0,
            (coverage >= 0.90) & (coverage < 1.0),
        ],
        [
            "full",
            "partial_high",
        ],
        default="partial_low",
    )
    return pd.Series(
        values,
        index=frame.index,
        name="coverage_stratum",
        dtype="object",
    )


def sample_count_strata(frame: pd.DataFrame) -> pd.Series:
    count = frame["sample_count"].astype(float)
    values = np.select(
        [
            count >= 30,
            (count >= 27) & (count <= 29),
        ],
        [
            "complete",
            "mild_underfill",
        ],
        default="severe_underfill",
    )
    return pd.Series(
        values,
        index=frame.index,
        name="sample_count_stratum",
        dtype="object",
    )


def prevalence_by_stratum(
    flags: pd.DataFrame,
    strata: pd.Series,
) -> dict[str, Any]:
    strata = strata.reindex(flags.index)
    result: dict[str, Any] = {}

    for stratum in pd.unique(strata):
        mask = strata == stratum
        result[str(stratum)] = {
            "rows": int(mask.sum()),
            "fraction_of_rows": float(mask.mean()),
            "flag_prevalence": {
                column: float(flags.loc[mask, column].mean())
                for column in flags.columns
            },
        }

    return result


def _score_summary(
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
        "q90": float(
            values.quantile(0.90, interpolation="higher")
        ),
        "q95": float(
            values.quantile(0.95, interpolation="higher")
        ),
        "q99": float(
            values.quantile(0.99, interpolation="higher")
        ),
        "q995": float(
            values.quantile(0.995, interpolation="higher")
        ),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "threshold_exceedance_fraction": float(
            (values >= threshold).mean()
        ),
    }


def quality_novelty_cells(
    frame: pd.DataFrame,
    flags: pd.DataFrame,
    scores: pd.Series,
    regimes: pd.Series,
    *,
    threshold: float,
) -> dict[str, Any]:
    flags = flags.reindex(frame.index)
    scores = scores.reindex(frame.index)
    regimes = regimes.reindex(frame.index)

    full = frame["coverage_ratio"].astype(float) >= 1.0
    novel = flags["any_cycle_novel"].astype(bool)

    cells = {
        "full_supported": full & ~novel,
        "full_novel": full & novel,
        "partial_supported": ~full & ~novel,
        "partial_novel": ~full & novel,
    }

    total_score = float(scores.sum())
    result: dict[str, Any] = {}

    for name, mask in cells.items():
        item = {
            "rows": int(mask.sum()),
            "fraction_of_background": float(mask.mean()),
            "score": _score_summary(
                scores.loc[mask],
                threshold=threshold,
            ),
            "share_of_background_score_mass": (
                float(scores.loc[mask].sum() / total_score)
                if total_score > 0.0
                else None
            ),
            "regime_distribution": {
                regime: (
                    float((regimes.loc[mask] == regime).mean())
                    if int(mask.sum())
                    else 0.0
                )
                for regime in REGIME_NAMES
            },
            "within_regime": {},
        }

        for regime in REGIME_NAMES:
            regime_mask = mask & (regimes == regime)
            item["within_regime"][regime] = {
                "rows": int(regime_mask.sum()),
                "score": _score_summary(
                    scores.loc[regime_mask],
                    threshold=threshold,
                ),
            }

        result[name] = item

    return result


def matched_transport_summary(
    calibration_scores: pd.Series,
    calibration_frame: pd.DataFrame,
    test_scores: pd.Series,
    test_frame: pd.DataFrame,
    *,
    threshold: float,
) -> dict[str, Any]:
    cal_full = calibration_frame["coverage_ratio"].astype(float) >= 1.0
    test_full = test_frame["coverage_ratio"].astype(float) >= 1.0
    cal_complete = calibration_frame["sample_count"].astype(float) >= 30
    test_complete = test_frame["sample_count"].astype(float) >= 30

    result: dict[str, Any] = {}

    for name, cal_mask, test_mask in (
        ("full_coverage", cal_full, test_full),
        ("complete_sample_count", cal_complete, test_complete),
    ):
        cal_summary = _score_summary(
            calibration_scores.loc[cal_mask],
            threshold=threshold,
        )
        test_summary = _score_summary(
            test_scores.loc[test_mask],
            threshold=threshold,
        )
        result[name] = {
            "calibration": cal_summary,
            "background_test": test_summary,
            "test_to_calibration_q99_ratio": (
                float(
                    test_summary["q99"]
                    / cal_summary["q99"]
                )
                if cal_summary.get("q99", 0.0) != 0.0
                else None
            ),
            "test_to_calibration_q995_ratio": (
                float(
                    test_summary["q995"]
                    / cal_summary["q995"]
                )
                if cal_summary.get("q995", 0.0) != 0.0
                else None
            ),
        }

    return result


def raw_summary(series: pd.Series) -> dict[str, Any]:
    values = series.astype(float).dropna()
    return {
        "count": len(values),
        "min": float(values.min()),
        "q01": float(values.quantile(0.01)),
        "median": float(values.median()),
        "q99": float(values.quantile(0.99)),
        "max": float(values.max()),
    }


def feature_support_by_quality(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    background: pd.DataFrame,
    *,
    features: list[str],
) -> dict[str, Any]:
    full = background["coverage_ratio"].astype(float) >= 1.0
    partial = ~full
    result: dict[str, Any] = {}

    for feature in features:
        train_min = float(train[feature].min())
        train_max = float(train[feature].max())

        def one(
            frame: pd.DataFrame, *,
            feature_name: str = feature,
            minimum: float = train_min,
            maximum: float = train_max,
        ) -> dict[str, Any]:
            values = frame[feature_name].astype(float)
            return {
                "raw": raw_summary(values),
                "below_train_min_fraction": float(
                    (values < minimum).mean()
                ),
                "above_train_max_fraction": float(
                    (values > maximum).mean()
                ),
            }

        result[feature] = {
            "train": {
                "raw": raw_summary(train[feature]),
                "train_min": train_min,
                "train_max": train_max,
            },
            "calibration": one(calibration),
            "background_full_coverage": one(
                background.loc[full]
            ),
            "background_partial_coverage": one(
                background.loc[partial]
            ),
        }

    return result


def binary_association(
    flag: pd.Series,
    exposure: pd.Series,
) -> dict[str, Any]:
    flag, exposure = flag.align(exposure, join="inner")
    flag = flag.astype(bool)
    exposure = exposure.astype(bool)

    a = int((flag & exposure).sum())
    b = int((~flag & exposure).sum())
    c = int((flag & ~exposure).sum())
    d = int((~flag & ~exposure).sum())

    exposed_risk = a / (a + b) if a + b else None
    unexposed_risk = c / (c + d) if c + d else None

    risk_ratio = (
        exposed_risk / unexposed_risk
        if (
            exposed_risk is not None
            and unexposed_risk not in (None, 0.0)
        )
        else None
    )

    aa, bb, cc, dd = (
        a + 0.5,
        b + 0.5,
        c + 0.5,
        d + 0.5,
    )
    odds_ratio = (aa * dd) / (bb * cc)

    return {
        "exposed_flag_count": a,
        "exposed_nonflag_count": b,
        "unexposed_flag_count": c,
        "unexposed_nonflag_count": d,
        "flag_prevalence_exposed": exposed_risk,
        "flag_prevalence_unexposed": unexposed_risk,
        "risk_ratio": risk_ratio,
        "odds_ratio_haldane_anscombe": float(odds_ratio),
    }


def association_summary(
    flags: pd.DataFrame,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    partial = frame["coverage_ratio"].astype(float) < 1.0
    underfilled = frame["sample_count"].astype(float) < 30

    return {
        column: {
            "partial_vs_full_coverage": binary_association(
                flags[column],
                partial,
            ),
            "underfilled_vs_complete_sample_count": binary_association(
                flags[column],
                underfilled,
            ),
        }
        for column in flags.columns
    }


def joint_patterns_full_coverage(
    atomic_flags: pd.DataFrame,
    frame: pd.DataFrame,
    scores: pd.Series,
    regimes: pd.Series,
    *,
    threshold: float,
    top_n: int,
) -> dict[str, Any]:
    full = frame["coverage_ratio"].astype(float) >= 1.0
    flags = atomic_flags.loc[full]
    scores = scores.loc[full]
    regimes = regimes.loc[full]
    names = list(flags.columns)

    patterns = flags.apply(
        lambda row: "|".join(
            name for name in names if bool(row[name])
        )
        or "none",
        axis=1,
    )

    table = pd.DataFrame(
        {
            "pattern": patterns,
            "score": scores,
            "regime": regimes,
        },
        index=flags.index,
    )

    def summarize(subset: pd.DataFrame) -> list[dict[str, Any]]:
        if subset.empty:
            return []
        counts = (
            subset.groupby("pattern")
            .size()
            .sort_values(ascending=False)
        )
        output = []
        for pattern in counts.head(top_n).index:
            group = subset.loc[
                subset["pattern"] == pattern
            ]
            output.append(
                {
                    "pattern": str(pattern),
                    "count": len(group),
                    "fraction": float(
                        len(group) / len(subset)
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
        return output

    return {
        "full_coverage_all": summarize(table),
        "full_coverage_high_score": summarize(
            table.loc[table["score"] >= threshold]
        ),
    }
