from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from ml.detection.alerts import AlertEpisode

FROZEN_PCA_METRIC_KEYS = (
    "timely_incident_recall",
    "pre_onset_incident_recall",
    "incident_overlap_recall",
    "false_alerts_per_24h",
    "time_in_alert_fraction",
    "relevant_episode_precision",
    "pr_auc",
)


def verify_frozen_metrics(
    observed: dict[str, Any],
    expected: dict[str, Any],
    *,
    tolerance: float,
) -> dict[str, float]:
    if tolerance < 0.0:
        raise ValueError("tolerance cannot be negative.")

    deltas: dict[str, float] = {}

    for key in FROZEN_PCA_METRIC_KEYS:
        observed_value = float(observed[key])
        expected_value = float(expected[key])
        delta = abs(observed_value - expected_value)
        deltas[key] = delta

        if delta > tolerance:
            raise RuntimeError(
                "Frozen PCA reproduction failed for "
                f"{key}: observed={observed_value}, "
                f"expected={expected_value}, delta={delta}, "
                f"tolerance={tolerance}."
            )

    return deltas


def incident_related_mask(
    index: pd.DatetimeIndex,
    incidents: list[dict[str, Any]],
    *,
    early_warning_hours: int,
) -> pd.Series:
    mask = pd.Series(False, index=index, dtype=bool)
    early = pd.Timedelta(hours=early_warning_hours)

    for incident in incidents:
        start = pd.Timestamp(incident["start"])
        end = pd.Timestamp(incident["end"])
        mask.loc[
            (mask.index >= start - early)
            & (mask.index <= end)
        ] = True

    return mask.rename("incident_related")


def episode_is_incident_related(
    episode: AlertEpisode,
    incidents: list[dict[str, Any]],
    *,
    early_warning_hours: int,
) -> bool:
    early = pd.Timedelta(hours=early_warning_hours)

    return any(
        (
            episode.start <= pd.Timestamp(incident["end"])
            and episode.end
            >= pd.Timestamp(incident["start"]) - early
        )
        for incident in incidents
    )


def episode_coverage(
    episodes: list[AlertEpisode],
    *,
    alerts: pd.Series,
    routed: pd.Series,
    incidents: list[dict[str, Any]],
    early_warning_hours: int,
) -> dict[str, int | float | None]:
    alerts = alerts.astype(bool)
    routed = routed.reindex(alerts.index, fill_value=False).astype(bool)

    covered_total = 0
    incident_total = 0
    incident_covered = 0

    for episode in episodes:
        within = (
            (alerts.index >= episode.start)
            & (alerts.index <= episode.end)
        )
        evidence_bins = alerts.loc[within]

        covered = bool(
            (
                evidence_bins
                & routed.reindex(evidence_bins.index)
            ).any()
        )

        covered_total += int(covered)

        related = episode_is_incident_related(
            episode,
            incidents,
            early_warning_hours=early_warning_hours,
        )

        if related:
            incident_total += 1
            incident_covered += int(covered)

    total = len(episodes)

    return {
        "alert_episodes": total,
        "covered_alert_episodes": covered_total,
        "alert_episode_coverage": (
            covered_total / total
            if total
            else None
        ),
        "incident_related_alert_episodes": incident_total,
        "covered_incident_related_alert_episodes": incident_covered,
        "incident_related_alert_episode_coverage": (
            incident_covered / incident_total
            if incident_total
            else None
        ),
    }


def bin_coverage(
    *,
    alerts: pd.Series,
    routed: pd.Series,
    incident_related: pd.Series,
) -> dict[str, int | float | None]:
    alerts = alerts.astype(bool)
    routed = routed.reindex(alerts.index, fill_value=False).astype(bool)
    incident_related = incident_related.reindex(
        alerts.index,
        fill_value=False,
    ).astype(bool)

    alert_bins = alerts
    incident_alert_bins = alerts & incident_related

    alert_total = int(alert_bins.sum())
    incident_total = int(incident_alert_bins.sum())

    covered_alert = int((alert_bins & routed).sum())
    covered_incident = int((incident_alert_bins & routed).sum())

    return {
        "tcn_alert_bins": alert_total,
        "covered_tcn_alert_bins": covered_alert,
        "tcn_alert_bin_coverage": (
            covered_alert / alert_total
            if alert_total
            else None
        ),
        "tcn_incident_related_alert_bins": incident_total,
        "covered_tcn_incident_related_alert_bins": covered_incident,
        "tcn_incident_alert_bin_coverage": (
            covered_incident / incident_total
            if incident_total
            else None
        ),
    }


def serializable_episodes(
    episodes: list[AlertEpisode],
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []

    for episode in episodes:
        record = asdict(episode)
        record["start"] = str(record["start"])
        record["end"] = str(record["end"])
        payload.append(record)

    return payload


def percentile(
    values: list[float],
    q: float,
) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q))
