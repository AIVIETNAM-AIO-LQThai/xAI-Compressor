from __future__ import annotations

import pandas as pd
import pytest

from ml.detection.representation_ablation import (
    concentration_at_timestamp,
    matched_timestamp_table,
)


def test_concentration_at_timestamp_uses_group_mass():
    frame = pd.DataFrame(
        {
            "a": [8.0],
            "b": [1.0],
            "c": [1.0],
        },
        index=pd.to_datetime(
            [
                "2022-06-04 11:05:00",
            ]
        ),
    )

    result = concentration_at_timestamp(
        frame,
        "2022-06-04 11:05:00",
    )

    assert (
        result[
            "dominant_group"
        ]
        == "a"
    )

    assert (
        result[
            "top1_concentration"
        ]
        == pytest.approx(0.8)
    )

    assert (
        result[
            "top3_concentration"
        ]
        == pytest.approx(1.0)
    )


def test_matched_timestamp_uses_reference_alert():
    timestamp = pd.Timestamp(
        "2022-06-04 11:05:00"
    )

    robust_frame = pd.DataFrame(
        {
            "a": [9.0],
            "b": [1.0],
        },
        index=[timestamp],
    )

    other_frame = pd.DataFrame(
        {
            "a": [5.0],
            "b": [5.0],
        },
        index=[timestamp],
    )

    results = {
        "robust_pca": {
            "metrics": {
                "incident_results": [
                    {
                        "id": 1,
                        "first_alert": str(
                            timestamp
                        ),
                    }
                ]
            },
            "_smoothed_contributions": (
                robust_frame
            ),
        },
        "feature_energy": {
            "metrics": {
                "incident_results": [
                    {
                        "id": 1,
                        "first_alert": (
                            "2022-06-04 "
                            "11:10:00"
                        ),
                    }
                ]
            },
            "_smoothed_contributions": (
                other_frame
            ),
        },
    }

    table = matched_timestamp_table(
        representation_results=results,
        reference_name="robust_pca",
        incidents=[
            {
                "id": 1,
                "condition": "air_leak",
            }
        ],
    )

    assert (
        table[0][
            "reference_timestamp"
        ]
        == str(timestamp)
    )

    assert (
        table[0][
            "representations"
        ][
            "robust_pca"
        ][
            "top1_concentration"
        ]
        == pytest.approx(0.9)
    )

    assert (
        table[0][
            "representations"
        ][
            "feature_energy"
        ][
            "top1_concentration"
        ]
        == pytest.approx(0.5)
    )
