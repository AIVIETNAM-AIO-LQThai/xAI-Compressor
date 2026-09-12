from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
)

from ml.detection.alerts import (
    AlertEpisode,
)


def evaluate_detection(
    *,
    scores: pd.Series,
    alerts: pd.Series,
    episodes: list[AlertEpisode],
    incidents: list[dict[str, Any]],
    early_warning_hours: int,
    late_tolerance_hours: int,
    bin_minutes: int,
) -> dict[str, Any]:
    scores = scores.sort_index()
    alerts = alerts.reindex(
        scores.index,
        fill_value=False,
    )

    labels = pd.Series(
        False,
        index=scores.index,
    )

    incident_results: list[
        dict[str, Any]
    ] = []

    timely_count = 0
    anytime_count = 0

    early_window = pd.Timedelta(
        hours=early_warning_hours
    )

    late_window = pd.Timedelta(
        hours=late_tolerance_hours
    )

    for incident in incidents:
        start = pd.Timestamp(
            incident["start"]
        )

        end = pd.Timestamp(
            incident["end"]
        )

        labels.loc[
            (labels.index >= start)
            & (labels.index <= end)
        ] = True

        timely_start = (
            start - early_window
        )

        timely_end = (
            start + late_window
        )

        timely_candidates = [
            episode
            for episode in episodes
            if (
                timely_start
                <= episode.start
                <= timely_end
            )
        ]

        anytime_candidates = [
            episode
            for episode in episodes
            if (
                episode.start <= end
                and episode.end >= start
            )
        ]

        timely_detected = bool(
            timely_candidates
        )

        anytime_detected = bool(
            anytime_candidates
        )

        timely_count += int(
            timely_detected
        )

        anytime_count += int(
            anytime_detected
        )

        first_alert = (
            min(
                timely_candidates,
                key=lambda episode:
                    episode.start,
            )
            if timely_candidates
            else None
        )

        lead_hours = None
        delay_hours = None

        if first_alert is not None:
            difference = (
                start - first_alert.start
            ).total_seconds() / 3600.0

            if difference >= 0:
                lead_hours = difference
            else:
                delay_hours = -difference

        incident_results.append(
            {
                "id": incident["id"],
                "timely_detected":
                    timely_detected,
                "anytime_detected":
                    anytime_detected,
                "first_alert":
                    (
                        str(first_alert.start)
                        if first_alert
                        else None
                    ),
                "lead_hours":
                    lead_hours,
                "delay_hours":
                    delay_hours,
            }
        )

    relevant_episodes: set[int] = set()

    for index, episode in enumerate(
        episodes
    ):
        for incident in incidents:
            start = pd.Timestamp(
                incident["start"]
            )

            end = pd.Timestamp(
                incident["end"]
            )

            relevant_start = (
                start - early_window
            )

            if (
                episode.start <= end
                and episode.end
                >= relevant_start
            ):
                relevant_episodes.add(
                    index
                )
                break

    false_episode_count = (
        len(episodes)
        - len(relevant_episodes)
    )

    valid_exposure_days = (
        len(scores)
        * bin_minutes
        / (60.0 * 24.0)
    )

    false_alerts_per_24h = (
        false_episode_count
        / valid_exposure_days
        if valid_exposure_days > 0
        else None
    )

    precision_recall_auc = float(
        average_precision_score(
            labels.astype(int),
            scores.to_numpy(),
        )
    )

    return {
        "timely_incident_recall": (
            timely_count
            / len(incidents)
            if incidents
            else None
        ),
        "anytime_incident_recall": (
            anytime_count
            / len(incidents)
            if incidents
            else None
        ),
        "incident_results":
            incident_results,
        "episodes_total":
            len(episodes),
        "false_episodes":
            false_episode_count,
        "false_alerts_per_24h":
            false_alerts_per_24h,
        "time_in_alert_fraction":
            float(alerts.mean()),
        "pr_auc":
            precision_recall_auc,
        "valid_exposure_days":
            valid_exposure_days,
    }