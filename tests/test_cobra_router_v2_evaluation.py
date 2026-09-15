from __future__ import annotations

import numpy as np

from scripts.evaluate_cobra_router_v2 import (
    _per_day_metrics,
    criterion_status,
)


def test_high_criterion_passes_when_adequate() -> None:
    assert criterion_status(
        evidence_count=20,
        minimum_count=20,
        coverage=0.80,
        minimum_coverage=0.80,
    ) == "PASS"


def test_high_criterion_fails_on_coverage() -> None:
    assert criterion_status(
        evidence_count=20,
        minimum_count=20,
        coverage=0.79,
        minimum_coverage=0.80,
    ) == "FAIL"


def test_alert_criterion_can_be_insufficient() -> None:
    assert criterion_status(
        evidence_count=2,
        minimum_count=3,
        coverage=1.0,
        minimum_coverage=0.80,
    ) == "INSUFFICIENT_EVIDENCE"


def test_per_day_metrics() -> None:
    days = np.array(
        [
            "2025-02-17",
            "2025-02-17",
            "2025-02-18",
        ]
    )

    route = np.array(
        [True, False, True]
    )

    high = np.array(
        [True, False, True]
    )

    alert = np.array(
        [False, False, True]
    )

    reports = _per_day_metrics(
        days=days,
        route=route,
        high=high,
        alert=alert,
    )

    assert len(reports) == 2

    assert reports[0][
        "eligible_targets"
    ] == 2

    assert reports[0][
        "tcn_invocation_fraction"
    ] == 0.5

    assert reports[0][
        "high_evidence_coverage"
    ] == 1.0

    assert reports[1][
        "alert_evidence_coverage"
    ] == 1.0
