from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.data.iitk_air_compressor import (
    FEATURE_GROUPS,
    extract_recording_features,
    feature_group_columns,
    healthy_split_for_reading,
    parse_recording_text,
)


def test_parse_recording_text_accepts_whitespace():
    values = parse_recording_text(
        "1.0  2.0\t3.5\n4.0"
    )

    assert values.tolist() == [
        1.0,
        2.0,
        3.5,
        4.0,
    ]


def test_feature_schema_is_finite_and_complete():
    t = np.linspace(
        0.0,
        20.0 * np.pi,
        4096,
        endpoint=False,
    )

    signal = (
        np.sin(
            t
        )
        + 0.25
        * np.sin(
            4.0 * t
        )
    )

    features = (
        extract_recording_features(
            signal,
            spectral_band_edges=[
                0.000,
                0.125,
                0.250,
                0.375,
                0.500,
                0.625,
                0.750,
                0.875,
                1.000,
            ],
        )
    )

    assert set(
        features
    ) == set(
        FEATURE_GROUPS
    )

    assert np.isfinite(
        np.asarray(
            list(
                features.values()
            )
        )
    ).all()

    band_sum = sum(
        features[
            f"spectral_band_{index}_energy_fraction"
        ]
        for index in range(
            8
        )
    )

    assert band_sum == pytest.approx(
        1.0,
        abs=1.0e-12,
    )


def test_feature_groups_have_five_semantic_families():
    groups = feature_group_columns(
        FEATURE_GROUPS
    )

    assert set(
        groups
    ) == {
        "amplitude",
        "distribution_shape",
        "temporal_structure",
        "spectral_shape",
        "spectral_band_energy",
    }


def test_healthy_split_is_contiguous_and_locked():
    config = {
        "train": {
            "first_reading": 1,
            "last_reading": 120,
        },
        "calibration": {
            "first_reading": 121,
            "last_reading": 180,
        },
        "test": {
            "first_reading": 181,
            "last_reading": 225,
        },
    }

    assert healthy_split_for_reading(
        1,
        config,
    ) == "train"

    assert healthy_split_for_reading(
        120,
        config,
    ) == "train"

    assert healthy_split_for_reading(
        121,
        config,
    ) == "calibration"

    assert healthy_split_for_reading(
        180,
        config,
    ) == "calibration"

    assert healthy_split_for_reading(
        181,
        config,
    ) == "test"

    assert healthy_split_for_reading(
        225,
        config,
    ) == "test"

    with pytest.raises(
        ValueError
    ):
        healthy_split_for_reading(
            226,
            config,
        )


def test_feature_table_metadata_does_not_overlap_features():
    metadata = pd.DataFrame(
        {
            "condition": [
                "healthy"
            ],
            "reading": [
                1
            ],
            "split": [
                "train"
            ],
            "is_fault": [
                False
            ],
        }
    )

    assert not (
        set(
            metadata.columns
        )
        & set(
            FEATURE_GROUPS
        )
    )
