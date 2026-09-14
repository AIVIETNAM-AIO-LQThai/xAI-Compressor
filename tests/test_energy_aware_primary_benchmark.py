from __future__ import annotations

import pandas as pd
import pytest

from ml.detection.alerts import AlertEpisode
from ml.energy.primary_benchmark import (
    bin_coverage,
    episode_coverage,
    incident_related_mask,
    verify_frozen_metrics,
)


def test_verify_frozen_metrics_passes_exact_match() -> None:
    expected = {
        "timely_incident_recall": 1.0,
        "pre_onset_incident_recall": 0.5,
        "incident_overlap_recall": 1.0,
        "false_alerts_per_24h": 0.8,
        "time_in_alert_fraction": 0.1,
        "relevant_episode_precision": 0.2,
        "pr_auc": 0.3,
    }

    deltas = verify_frozen_metrics(
        expected,
        expected,
        tolerance=1.0e-12,
    )

    assert all(value == 0.0 for value in deltas.values())


def test_verify_frozen_metrics_rejects_drift() -> None:
    expected = {
        "timely_incident_recall": 1.0,
        "pre_onset_incident_recall": 0.5,
        "incident_overlap_recall": 1.0,
        "false_alerts_per_24h": 0.8,
        "time_in_alert_fraction": 0.1,
        "relevant_episode_precision": 0.2,
        "pr_auc": 0.3,
    }
    observed = dict(expected)
    observed["pr_auc"] = 0.31

    with pytest.raises(RuntimeError, match="pr_auc"):
        verify_frozen_metrics(
            observed,
            expected,
            tolerance=1.0e-12,
        )


def test_incident_related_mask_uses_early_window_through_end() -> None:
    index = pd.date_range(
        "2026-01-01 00:00",
        periods=8,
        freq="1h",
    )
    incidents = [
        {
            "id": 1,
            "start": "2026-01-01 04:00",
            "end": "2026-01-01 05:00",
        }
    ]

    mask = incident_related_mask(
        index,
        incidents,
        early_warning_hours=2,
    )

    assert mask.tolist() == [
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
    ]


def test_bin_and_episode_coverage() -> None:
    index = pd.date_range(
        "2026-01-01 00:00",
        periods=6,
        freq="5min",
    )
    alerts = pd.Series(
        [False, True, True, False, True, False],
        index=index,
    )
    routed = pd.Series(
        [False, False, True, False, False, False],
        index=index,
    )
    related = pd.Series(
        [False, True, True, False, False, False],
        index=index,
    )

    bins = bin_coverage(
        alerts=alerts,
        routed=routed,
        incident_related=related,
    )

    assert bins["tcn_alert_bins"] == 3
    assert bins["covered_tcn_alert_bins"] == 1
    assert bins["tcn_alert_bin_coverage"] == pytest.approx(1 / 3)
    assert bins["tcn_incident_related_alert_bins"] == 2
    assert bins["tcn_incident_alert_bin_coverage"] == pytest.approx(0.5)

    episodes = [
        AlertEpisode(
            start=index[1],
            end=index[2],
            alert_bins=2,
        ),
        AlertEpisode(
            start=index[4],
            end=index[4],
            alert_bins=1,
        ),
    ]
    incidents = [
        {
            "id": 1,
            "start": str(index[2]),
            "end": str(index[2]),
        }
    ]

    episode_metrics = episode_coverage(
        episodes,
        alerts=alerts,
        routed=routed,
        incidents=incidents,
        early_warning_hours=1,
    )

    assert episode_metrics["alert_episodes"] == 2
    assert episode_metrics["covered_alert_episodes"] == 1
    assert episode_metrics["alert_episode_coverage"] == pytest.approx(0.5)
    assert episode_metrics["incident_related_alert_episodes"] == 1
    assert (
        episode_metrics["incident_related_alert_episode_coverage"]
        == pytest.approx(1.0)
    )
