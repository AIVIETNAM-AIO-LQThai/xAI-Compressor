from __future__ import annotations

import pandas as pd
import pytest

from ml.detection.pca_research_benchmark import (
    AlertProtocol,
)
from ml.explainability.temporal_dynamics import (
    concentration_trajectory,
    incident_window,
    phase_mask,
    summarize_phase,
)


def _protocol() -> AlertProtocol:
    return AlertProtocol(
        ewma_alpha=1.0,
        threshold_quantile=0.5,
        persistence_hits=1,
        persistence_window=1,
        reset_gap_minutes=10,
        merge_minutes=30,
        early_warning_hours=24,
        late_tolerance_hours=2,
        bin_minutes=5,
    )


def test_concentration_trajectory_tracks_distribution():
    index = pd.date_range(
        "2022-06-01",
        periods=2,
        freq="5min",
    )

    contributions = pd.DataFrame(
        {
            "a": [
                9.0,
                5.0,
            ],
            "b": [
                1.0,
                5.0,
            ],
        },
        index=index,
    )

    result = concentration_trajectory(
        contributions,
        threshold=1.0,
        protocol=_protocol(),
    )

    assert (
        result.iloc[0][
            "top1_concentration"
        ]
        == pytest.approx(0.9)
    )

    assert (
        result.iloc[1][
            "top1_concentration"
        ]
        == pytest.approx(0.5)
    )

    assert (
        result.iloc[1][
            "effective_group_count"
        ]
        > result.iloc[0][
            "effective_group_count"
        ]
    )


def test_incident_window_is_onset_relative():
    index = pd.date_range(
        "2022-06-01 00:00:00",
        periods=13,
        freq="1h",
    )

    frame = pd.DataFrame(
        {
            "value": range(
                len(index)
            )
        },
        index=index,
    )

    window = incident_window(
        frame,
        onset=(
            "2022-06-01 "
            "06:00:00"
        ),
        pre_onset_hours=2,
        post_onset_hours=2,
    )

    assert (
        window.index.min()
        == pd.Timestamp(
            "2022-06-01 "
            "04:00:00"
        )
    )

    assert (
        window.index.max()
        == pd.Timestamp(
            "2022-06-01 "
            "08:00:00"
        )
    )

    assert (
        window.loc[
            pd.Timestamp(
                "2022-06-01 "
                "06:00:00"
            ),
            "hours_from_onset",
        ]
        == pytest.approx(0.0)
    )


def test_phase_mask_avoids_overlap():
    hours = pd.Series(
        [
            -6.0,
            -0.1,
            0.0,
            1.9,
            2.0,
            6.0,
        ]
    )

    near = phase_mask(
        hours,
        start_hours=-6.0,
        end_hours=0.0,
        is_last_phase=False,
    )

    early = phase_mask(
        hours,
        start_hours=0.0,
        end_hours=2.0,
        is_last_phase=False,
    )

    established = phase_mask(
        hours,
        start_hours=2.0,
        end_hours=6.0,
        is_last_phase=True,
    )

    assert not bool(
        (
            near & early
        ).any()
    )

    assert not bool(
        (
            early & established
        ).any()
    )

    assert int(
        near.sum()
    ) == 2

    assert int(
        early.sum()
    ) == 2

    assert int(
        established.sum()
    ) == 2


def test_summarize_phase_counts_switches():
    frame = pd.DataFrame(
        {
            "alert": [
                True,
                True,
                False,
            ],
            "top1_concentration": [
                0.8,
                0.6,
                0.7,
            ],
            "top3_concentration": [
                1.0,
                1.0,
                1.0,
            ],
            "effective_group_count": [
                1.5,
                2.0,
                1.8,
            ],
            "dominant_group": [
                "a",
                "b",
                "b",
            ],
        },
        index=pd.date_range(
            "2022-06-01",
            periods=3,
            freq="5min",
        ),
    )

    summary = summarize_phase(
        frame
    )

    assert (
        summary[
            "valid_bins"
        ]
        == 3
    )

    assert (
        summary[
            "alert_bins"
        ]
        == 2
    )

    assert (
        summary[
            "dominant_group_switches"
        ]
        == 1
    )

    assert (
        summary[
            "max_effective_groups"
        ]
        == pytest.approx(2.0)
    )
