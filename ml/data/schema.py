from __future__ import annotations

import pandas as pd

ANALOG_COLUMNS = [
    "tp2",
    "tp3",
    "h1",
    "dv_pressure",
    "reservoirs",
    "oil_temperature",
    "motor_current",
]

DIGITAL_COLUMNS = [
    "comp",
    "dv_electric",
    "towers",
    "mpg",
    "lps",
    "pressure_switch",
    "oil_level",
    "caudal_impulses",
]

SENSOR_COLUMNS = ANALOG_COLUMNS + DIGITAL_COLUMNS
REQUIRED_COLUMNS = ["timestamp", *SENSOR_COLUMNS]


ALIASES = {
    # The released dataset contains this original spelling.
    "dv_eletric": "dv_electric",
}


def canonicalize_column_name(name: object) -> str:
    value = (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    return ALIASES.get(value, value)


def normalize_raw_frame(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(
        columns={
            column: canonicalize_column_name(column)
            for column in frame.columns
        }
    )

    drop_columns = [
        column
        for column in renamed.columns
        if column.startswith("unnamed")
        or column in {"index", "level_0"}
    ]

    normalized = renamed.drop(
        columns=drop_columns,
        errors="ignore",
    ).copy()

    missing = sorted(
        set(REQUIRED_COLUMNS) - set(normalized.columns)
    )

    if missing:
        raise ValueError(
            f"Missing MetroPT-3 columns: {missing}"
        )

    normalized = normalized[REQUIRED_COLUMNS]

    normalized["timestamp"] = pd.to_datetime(
        normalized["timestamp"],
        errors="raise",
    )

    for column in SENSOR_COLUMNS:
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="raise",
        )

    normalized = (
        normalized
        .sort_values("timestamp")
        .drop_duplicates(
            subset="timestamp",
            keep="last",
        )
        .reset_index(drop=True)
    )

    return normalized