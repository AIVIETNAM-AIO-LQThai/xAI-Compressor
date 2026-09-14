from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

RAW_TO_CANONICAL = {
    "timestamp": "timestamp",
    "TP2": "tp2",
    "TP3": "tp3",
    "H1": "h1",
    "DV_pressure": "dv_pressure",
    "Reservoirs": "reservoirs",
    "Oil_temperature": "oil_temperature",
    "Flowmeter": "flowmeter",
    "Motor_current": "motor_current",
    "COMP": "comp",
    "DV_eletric": "dv_electric",
    "Towers": "towers",
    "MPG": "mpg",
    "LPS": "lps",
    "Pressure_switch": "pressure_switch",
    "Oil_level": "oil_level",
    "Caudal_impulses": "caudal_impulses",
}

RAW_COLUMNS = tuple(RAW_TO_CANONICAL)

ANALOG_COLUMNS = (
    "tp2",
    "tp3",
    "h1",
    "dv_pressure",
    "reservoirs",
    "oil_temperature",
    "flowmeter",
    "motor_current",
)

DIGITAL_COLUMNS = (
    "comp",
    "dv_electric",
    "towers",
    "mpg",
    "lps",
    "pressure_switch",
    "oil_level",
    "caudal_impulses",
)

PRESSURE_RELATIONSHIPS = (
    "tp3_minus_reservoirs",
    "tp2_minus_tp3",
)

QUALITY_COLUMNS = (
    "sample_count",
    "coverage_ratio",
)


@dataclass(frozen=True)
class SplitFeatureResult:
    features: pd.DataFrame
    raw_rows: int
    raw_start: pd.Timestamp | None
    raw_end: pd.Timestamp | None
    candidate_bins: int
    valid_bins: int
    coverage_rejected_bins: int
    incomplete_rejected_bins: int
    gap_tainted_bins: int
    empty_bins: int


def load_metropt2_compressor_frame(
    path: str,
) -> pd.DataFrame:
    dtype = {
        column: "float32"
        for column in RAW_COLUMNS
        if column != "timestamp"
    }

    frame = pd.read_csv(
        path,
        usecols=list(RAW_COLUMNS),
        dtype=dtype,
        parse_dates=["timestamp"],
        low_memory=False,
    )

    frame = frame.rename(
        columns=RAW_TO_CANONICAL
    )

    if frame["timestamp"].isna().any():
        raise ValueError(
            "MetroPT2 contains invalid timestamps."
        )

    if frame["timestamp"].duplicated().any():
        raise ValueError(
            "MetroPT2 contains duplicate timestamps."
        )

    frame = (
        frame
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return frame


def _count_transitions(
    series: pd.Series,
) -> float:
    values = series.dropna().to_numpy()

    if len(values) <= 1:
        return 0.0

    return float(
        np.count_nonzero(
            values[1:] != values[:-1]
        )
    )


def feature_group(
    feature_name: str,
) -> str:
    base_name = feature_name.split(
        "__",
        maxsplit=1,
    )[0]

    if base_name in PRESSURE_RELATIONSHIPS:
        return "pressure_relationships"

    return base_name


def model_feature_columns(
    columns: Iterable[str],
    *,
    excluded_groups: Iterable[str] = (),
) -> list[str]:
    excluded = set(
        excluded_groups
    )

    return [
        column
        for column in columns
        if (
            column not in QUALITY_COLUMNS
            and feature_group(column)
            not in excluded
        )
    ]


def _gap_tainted_bin_endpoints(
    timestamps: pd.Series,
    *,
    rule: str,
    max_gap_seconds: float,
) -> pd.DatetimeIndex:
    deltas = (
        timestamps
        .sort_values()
        .diff()
        .dt.total_seconds()
    )

    first_after_gap = timestamps.loc[
        deltas > max_gap_seconds
    ]

    if first_after_gap.empty:
        return pd.DatetimeIndex([])

    return pd.DatetimeIndex(
        first_after_gap.dt.ceil(rule)
    ).unique()


def aggregate_metropt2_split(
    frame: pd.DataFrame,
    *,
    start: str,
    end: str,
    bin_minutes: int,
    expected_samples_per_bin: int,
    minimum_coverage: float,
    max_gap_seconds: float,
) -> SplitFeatureResult:
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError(
            "minimum_coverage must be in (0, 1]."
        )

    if max_gap_seconds <= 0.0:
        raise ValueError(
            "max_gap_seconds must be positive."
        )

    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)

    if start_time >= end_time:
        raise ValueError(
            "split start must precede split end."
        )

    split = frame.loc[
        (
            frame["timestamp"]
            >= start_time
        )
        & (
            frame["timestamp"]
            <= end_time
        )
    ].copy()

    raw_rows = len(split)

    if split.empty:
        return SplitFeatureResult(
            features=pd.DataFrame(),
            raw_rows=0,
            raw_start=None,
            raw_end=None,
            candidate_bins=0,
            valid_bins=0,
            coverage_rejected_bins=0,
            incomplete_rejected_bins=0,
            gap_tainted_bins=0,
            empty_bins=0,
        )

    raw_start = split["timestamp"].iloc[0]
    raw_end = split["timestamp"].iloc[-1]

    rule = f"{bin_minutes}min"

    tainted_endpoints = (
        _gap_tainted_bin_endpoints(
            split["timestamp"],
            rule=rule,
            max_gap_seconds=max_gap_seconds,
        )
    )

    indexed = (
        split
        .set_index("timestamp")
        .sort_index()
    )

    indexed[
        "tp3_minus_reservoirs"
    ] = (
        indexed["tp3"]
        - indexed["reservoirs"]
    )

    indexed[
        "tp2_minus_tp3"
    ] = (
        indexed["tp2"]
        - indexed["tp3"]
    )

    grouped = indexed.resample(
        rule,
        label="right",
        closed="right",
    )

    parts: list[pd.DataFrame] = []

    analog = grouped[
        list(ANALOG_COLUMNS)
    ].agg(
        [
            "mean",
            "std",
            "min",
            "max",
            "last",
        ]
    )

    analog.columns = [
        f"{sensor}__{stat}"
        for sensor, stat
        in analog.columns
    ]

    parts.append(
        analog
    )

    for sensor in DIGITAL_COLUMNS:
        sensor_group = grouped[
            sensor
        ]

        digital = pd.DataFrame(
            {
                f"{sensor}__active_ratio":
                    sensor_group.mean(),
                f"{sensor}__transitions":
                    sensor_group.apply(
                        _count_transitions
                    ),
                f"{sensor}__last":
                    sensor_group.last(),
            }
        )

        parts.append(
            digital
        )

    for column in PRESSURE_RELATIONSHIPS:
        delta = grouped[
            column
        ].agg(
            [
                "mean",
                "std",
            ]
        )

        delta.columns = [
            f"{column}__{stat}"
            for stat in delta.columns
        ]

        parts.append(
            delta
        )

    sample_count = grouped.size()

    coverage_ratio = (
        sample_count.astype(float)
        / expected_samples_per_bin
    ).clip(
        upper=1.0
    )

    features = pd.concat(
        parts,
        axis=1,
    )

    features[
        "sample_count"
    ] = sample_count

    features[
        "coverage_ratio"
    ] = coverage_ratio

    features.index.name = "timestamp"

    # Keep only bin endpoints that belong to this raw split.
    # Because each raw split is aggregated independently, the
    # first/last partial bins cannot leak samples across splits.
    features = features.loc[
        (
            features.index
            >= start_time
        )
        & (
            features.index
            <= end_time
        )
    ].copy()

    model_columns = model_feature_columns(
        features.columns
    )

    complete = (
        features[
            model_columns
        ]
        .notna()
        .all(axis=1)
    )

    adequate_coverage = (
        features[
            "coverage_ratio"
        ]
        >= minimum_coverage
    )

    gap_tainted = features.index.isin(
        tainted_endpoints
    )

    valid_mask = (
        complete
        & adequate_coverage
        & ~gap_tainted
    )

    valid = (
        features.loc[
            valid_mask
        ]
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .dropna(
            subset=model_columns
        )
        .sort_index()
    )

    return SplitFeatureResult(
        features=valid,
        raw_rows=raw_rows,
        raw_start=raw_start,
        raw_end=raw_end,
        candidate_bins=len(
            features
        ),
        valid_bins=len(
            valid
        ),
        coverage_rejected_bins=int(
            (
                ~adequate_coverage
            ).sum()
        ),
        incomplete_rejected_bins=int(
            (
                ~complete
            ).sum()
        ),
        gap_tainted_bins=int(
            gap_tainted.sum()
        ),
        empty_bins=int(
            (
                features[
                    "sample_count"
                ]
                == 0
            ).sum()
        ),
    )
