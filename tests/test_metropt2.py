from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml.data.metropt2 import (
    REQUIRED_CORE_COLUMNS,
    audit_metropt2_csv,
    canonicalize_metropt2_columns,
    read_metropt2_header,
)


def _sample_frame() -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2022-06-04 10:19:24.120",
            "2022-06-04 10:19:25.090",
            "2022-06-04 10:19:26.170",
            "2022-06-04 10:19:27.080",
            "2022-06-04 10:19:29.200",
        ]
    )

    rows = len(
        timestamps
    )

    payload: dict[
        str,
        object,
    ] = {
        "timestamp": timestamps,
        "TP2": [1.0] * rows,
        "TP3": [2.0] * rows,
        "H1": [3.0] * rows,
        "DV_pressure": [4.0] * rows,
        "Reservoirs": [5.0] * rows,
        "Oil_temperature": [60.0] * rows,
        "Flowmeter": [0.1] * rows,
        "Motor_current": [7.0] * rows,
        "COMP": [1] * rows,
        "LPS": [0] * rows,
        "latitude": [41.0] * rows,
    }

    return pd.DataFrame(
        payload
    )


def test_header_canonicalizes_core_columns(
    tmp_path: Path,
):
    path = tmp_path / "MetroPT2.csv"

    _sample_frame().to_csv(
        path,
        index=False,
    )

    header = read_metropt2_header(
        path
    )

    assert (
        header[
            "missing_core_columns"
        ]
        == []
    )

    assert set(
        REQUIRED_CORE_COLUMNS
    ).issubset(
        header[
            "canonical_columns"
        ]
    )


def test_audit_uses_tolerant_1hz_window(
    tmp_path: Path,
):
    path = tmp_path / "MetroPT2.csv"

    _sample_frame().to_csv(
        path,
        index=False,
    )

    audit = audit_metropt2_csv(
        path,
        chunksize=2,
        reported_incidents=[
            {
                "id": 1,
                "condition": "air_leak",
                "start": (
                    "2022-06-04 "
                    "10:19:24"
                ),
                "end": (
                    "2022-06-04 "
                    "10:19:26.500"
                ),
            }
        ],
    )

    cadence = audit[
        "cadence"
    ]

    assert audit[
        "row_count"
    ] == 5

    assert (
        cadence[
            "positive_step_count"
        ]
        == 4
    )

    # The first three transitions are close to
    # one second despite millisecond jitter.
    assert (
        cadence[
            "nominal_1hz_steps"
        ]
        == 3
    )

    # The final 2.12 s transition is a real gap,
    # not normal timestamp jitter.
    assert (
        cadence[
            "gap_steps_gt_1_5s"
        ]
        == 1
    )

    assert (
        cadence[
            "backward_steps"
        ]
        == 0
    )

    assert (
        audit[
            "reported_incidents"
        ][0][
            "rows_in_interval"
        ]
        == 3
    )


def test_canonicalizer_rejects_collisions():
    with pytest.raises(
        ValueError,
        match="duplicate canonical names",
    ):
        canonicalize_metropt2_columns(
            [
                "Motor Current",
                "Motor-Current",
            ]
        )
