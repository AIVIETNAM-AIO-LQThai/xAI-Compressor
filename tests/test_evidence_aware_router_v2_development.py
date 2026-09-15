from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ml.energy.evidence_aware_router_v2 import (
    V2_FEATURE_NAMES,
    RouterV2Candidate,
    candidate_grid,
    candidate_scores,
    evaluate_candidate,
    route_candidate,
    select_candidate,
)

CONFIG_PATH = Path(
    "configs/evidence_aware_router_v2.yaml"
)


def _frame(
    rows: list[list[float]],
) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=V2_FEATURE_NAMES,
    )


def test_candidate_grid_is_frozen_120() -> None:
    config = yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8"
        )
    )

    candidates = candidate_grid(
        config
    )

    assert len(candidates) == 120

    assert sum(
        item.family == "max3"
        for item in candidates
    ) == 15

    assert sum(
        item.family == "top2_mean"
        for item in candidates
    ) == 15

    assert sum(
        item.family == "convex"
        for item in candidates
    ) == 90


def test_max3_scores() -> None:
    frame = _frame(
        [
            [0.20, 0.40, 0.60],
            [0.80, 0.10, 0.30],
        ]
    )

    candidate = RouterV2Candidate(
        family="max3",
        threshold=0.50,
    )

    scores = candidate_scores(
        frame,
        candidate,
    )

    assert np.allclose(
        scores,
        [0.60, 0.80],
    )


def test_top2_mean_scores() -> None:
    frame = _frame(
        [
            [0.20, 0.40, 0.60],
            [0.80, 0.10, 0.30],
        ]
    )

    candidate = RouterV2Candidate(
        family="top2_mean",
        threshold=0.50,
    )

    scores = candidate_scores(
        frame,
        candidate,
    )

    assert np.allclose(
        scores,
        [0.50, 0.55],
    )


def test_convex_scores() -> None:
    frame = _frame(
        [
            [0.20, 0.40, 0.60],
        ]
    )

    candidate = RouterV2Candidate(
        family="convex",
        threshold=0.50,
        weights=(0.50, 0.25, 0.25),
    )

    scores = candidate_scores(
        frame,
        candidate,
    )

    assert np.allclose(
        scores,
        [0.35],
    )


def test_missing_input_forces_tcn() -> None:
    frame = _frame(
        [
            [np.nan, 0.10, 0.10],
            [0.20, 0.10, 0.10],
        ]
    )

    candidate = RouterV2Candidate(
        family="max3",
        threshold=0.95,
    )

    route, scores, forced = (
        route_candidate(
            frame,
            candidate,
        )
    )

    assert forced == 1
    assert route.tolist() == [
        True,
        False,
    ]
    assert np.isnan(scores[0])


def test_finite_out_of_range_input_fails() -> None:
    frame = _frame(
        [
            [1.10, 0.20, 0.30],
        ]
    )

    candidate = RouterV2Candidate(
        family="max3",
        threshold=0.50,
    )

    with pytest.raises(ValueError):
        route_candidate(
            frame,
            candidate,
        )


def test_alert_adequacy_can_be_insufficient() -> None:
    frame = _frame(
        [[0.90, 0.90, 0.90]]
        * 25
    )

    high = np.zeros(
        25,
        dtype=bool,
    )
    high[:20] = True

    alert = np.zeros(
        25,
        dtype=bool,
    )
    alert[:2] = True

    report = evaluate_candidate(
        candidate=RouterV2Candidate(
            family="max3",
            threshold=0.50,
        ),
        domains={
            "synthetic": {
                "features": frame,
                "high": high,
                "alert": alert,
            }
        },
        minimum_high_units=20,
        minimum_alert_units=3,
        minimum_high_coverage=0.80,
        minimum_alert_coverage=0.80,
    )

    domain = report["domains"][
        "synthetic"
    ]

    assert (
        domain["high_constraint_applied"]
        is True
    )
    assert (
        domain["alert_constraint_applied"]
        is False
    )
    assert report["eligible"] is True


def _report(
    *,
    family: str,
    threshold: float,
    worst_invocation: float,
    mean_invocation: float,
    worst_high: float,
    worst_alert: float,
    weights=None,
) -> dict:
    return {
        "family": family,
        "threshold": threshold,
        "weights": weights,
        "eligible": True,
        "worst_domain_tcn_invocation_fraction": (
            worst_invocation
        ),
        "mean_domain_tcn_invocation_fraction": (
            mean_invocation
        ),
        "worst_domain_high_evidence_coverage": (
            worst_high
        ),
        "worst_applicable_alert_coverage": (
            worst_alert
        ),
    }


def test_selection_prefers_worst_domain_invocation() -> None:
    reports = [
        _report(
            family="max3",
            threshold=0.50,
            worst_invocation=0.40,
            mean_invocation=0.20,
            worst_high=0.90,
            worst_alert=0.90,
        ),
        _report(
            family="top2_mean",
            threshold=0.50,
            worst_invocation=0.30,
            mean_invocation=0.29,
            worst_high=0.85,
            worst_alert=0.85,
        ),
    ]

    selected = select_candidate(
        reports,
        convex_weight_order=[],
    )

    assert selected is not None
    assert (
        selected["family"]
        == "top2_mean"
    )


def test_selection_family_then_higher_threshold() -> None:
    reports = [
        _report(
            family="top2_mean",
            threshold=0.90,
            worst_invocation=0.30,
            mean_invocation=0.20,
            worst_high=0.85,
            worst_alert=0.85,
        ),
        _report(
            family="max3",
            threshold=0.80,
            worst_invocation=0.30,
            mean_invocation=0.20,
            worst_high=0.85,
            worst_alert=0.85,
        ),
        _report(
            family="max3",
            threshold=0.90,
            worst_invocation=0.30,
            mean_invocation=0.20,
            worst_high=0.85,
            worst_alert=0.85,
        ),
    ]

    selected = select_candidate(
        reports,
        convex_weight_order=[],
    )

    assert selected is not None
    assert selected["family"] == "max3"
    assert selected["threshold"] == 0.90


def test_no_eligible_candidate_returns_none() -> None:
    report = _report(
        family="max3",
        threshold=0.50,
        worst_invocation=0.50,
        mean_invocation=0.50,
        worst_high=0.70,
        worst_alert=0.70,
    )
    report["eligible"] = False

    selected = select_candidate(
        [report],
        convex_weight_order=[],
    )

    assert selected is None
