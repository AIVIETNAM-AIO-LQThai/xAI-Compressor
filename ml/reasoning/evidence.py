from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GroupEvidence:
    group: str
    share: float
    calibration_percentile: float
    counterfactual_alert_cleared: bool | None


@dataclass(frozen=True)
class IncidentEvidence:
    incident_id: int
    incident_start: str
    explanation_timestamp: str
    timing: str
    groups: tuple[GroupEvidence, ...]
    temporal_window_complete: bool
    temporal_bins_requested: int
    temporal_bins_observed: int
    causal_claim: bool

    @property
    def dominant_group(self) -> GroupEvidence:
        if not self.groups:
            raise ValueError(
                "Incident evidence contains no groups."
            )

        return self.groups[0]


def _repair_map(
    incident: dict[str, Any],
) -> dict[str, bool]:
    result: dict[str, bool] = {}

    for repair in incident[
        "single_group_repairs"
    ]:
        groups = repair["repaired_groups"]

        if len(groups) != 1:
            continue

        result[str(groups[0])] = bool(
            repair["alert_cleared"]
        )

    return result


def incident_evidence_from_report(
    report: dict[str, Any],
    *,
    incident_id: int,
) -> IncidentEvidence:
    incidents = report.get(
        "incidents",
        [],
    )

    selected = next(
        (
            incident
            for incident in incidents
            if int(incident["id"])
            == incident_id
        ),
        None,
    )

    if selected is None:
        raise ValueError(
            f"Unknown incident id: {incident_id}"
        )

    repair_map = _repair_map(selected)

    groups = tuple(
        GroupEvidence(
            group=str(item["group"]),
            share=float(item["share"]),
            calibration_percentile=float(
                item[
                    "calibration_percentile"
                ]
            ),
            counterfactual_alert_cleared=(
                repair_map.get(
                    str(item["group"])
                )
            ),
        )
        for item in selected[
            "ranked_groups"
        ]
    )

    temporal = selected[
        "temporal_evidence"
    ]

    evidence = IncidentEvidence(
        incident_id=int(
            selected["id"]
        ),
        incident_start=str(
            selected["incident_start"]
        ),
        explanation_timestamp=str(
            selected[
                "explanation_timestamp"
            ]
        ),
        timing=str(
            selected["timing"]
        ),
        groups=groups,
        temporal_window_complete=bool(
            temporal["window_complete"]
        ),
        temporal_bins_requested=int(
            temporal[
                "window_bins_requested"
            ]
        ),
        temporal_bins_observed=int(
            temporal[
                "window_rows_observed"
            ]
        ),
        causal_claim=bool(
            selected["causal_claim"]
        ),
    )

    if evidence.causal_claim:
        raise ValueError(
            "Verified detector evidence must "
            "not contain a causal claim."
        )

    return evidence