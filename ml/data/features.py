from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import numpy as np
import pandas as pd

from ml.data.schema import ANALOG_COLUMNS, DIGITAL_COLUMNS

QUALITY_COLUMNS = [
    "sample_count",
    "coverage_ratio",
]


def count_transitions(series: pd.Series) -> float:
    values = series.dropna().to_numpy()

    if len(values) <= 1:
        return 0.0

    return float(
        np.count_nonzero(
            values[1:] != values[:-1]
        )
    )


def aggregate_causal_bins(
    frame: pd.DataFrame,
    *,
    bin_minutes: int,
    expected_samples: int,
) -> pd.DataFrame:
    indexed = (
        frame
        .set_index("timestamp")
        .sort_index()
        .copy()
    )

    # Physically meaningful pressure relationships.
    indexed["tp3_minus_reservoirs"] = (
        indexed["tp3"]
        - indexed["reservoirs"]
    )

    indexed["tp2_minus_tp3"] = (
        indexed["tp2"]
        - indexed["tp3"]
    )

    rule = f"{bin_minutes}min"

    grouped = indexed.resample(
        rule,
        label="right",
        closed="right",
    )

    parts: list[pd.DataFrame] = []

    # Analog sensor summaries.
    analog = grouped[ANALOG_COLUMNS].agg(
        [
            "mean",
            "std",
            "min",
            "max",
            "last",
        ]
    )

    analog_columns = cast(
        pd.MultiIndex,
        analog.columns,
    )

    analog.columns = [
        f"{sensor}__{stat}"
        for sensor, stat in analog_columns
    ]

    parts.append(analog)

    # Digital-state behavior.
    for sensor in DIGITAL_COLUMNS:
        sensor_group = grouped[sensor]

        digital = pd.DataFrame(
            {
                f"{sensor}__active_ratio":
                    sensor_group.mean(),
                f"{sensor}__transitions":
                    sensor_group.apply(
                        count_transitions
                    ),
                f"{sensor}__last":
                    sensor_group.last(),
            }
        )

        parts.append(digital)

    # Pressure relationships can contain useful
    # compressor-cycle information.
    for column in [
        "tp3_minus_reservoirs",
        "tp2_minus_tp3",
    ]:
        delta = grouped[column].agg(
            ["mean", "std"]
        )

        delta.columns = [
            f"{column}__{stat}"
            for stat in delta.columns
        ]

        parts.append(delta)

    sample_count = grouped.size()

    coverage_ratio = (
        sample_count.astype(float)
        / expected_samples
    ).clip(upper=1.0)

    features = pd.concat(
        parts,
        axis=1,
    )

    features["sample_count"] = sample_count
    features["coverage_ratio"] = coverage_ratio

    features.index.name = "timestamp"

    return features.replace(
        [np.inf, -np.inf],
        np.nan,
    )


def model_feature_columns(
    columns: Iterable[str],
) -> list[str]:
    return [
        column
        for column in columns
        if column not in QUALITY_COLUMNS
    ]


def filter_valid_bins(
    features: pd.DataFrame,
    *,
    minimum_coverage: float,
) -> pd.DataFrame:
    model_columns = model_feature_columns(
        features.columns
    )

    adequate_coverage = (
        features["coverage_ratio"]
        >= minimum_coverage
    )

    complete_features = (
        features[model_columns]
        .notna()
        .all(axis=1)
    )

    return (
        features.loc[
            adequate_coverage
            & complete_features
        ]
        .copy()
        .sort_index()
    )


def select_time_range(
    frame: pd.DataFrame,
    *,
    start: str,
    end: str,
) -> pd.DataFrame:
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)

    mask = (
        (frame.index >= start_time)
        & (frame.index <= end_time)
    )

    return frame.loc[mask].copy()