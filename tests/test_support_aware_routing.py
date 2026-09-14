from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.preprocessing import StandardScaler

from ml.energy.support_aware_routing import (
    SupportPolicy,
    candidate_grid,
    empirical_percentile,
    evidence_coverage,
    feature_support_statistic,
    select_policy,
    support_aware_route,
)

ROOT_CONFIG = (
    "configs/support_aware_energy_routing.yaml"
)


def test_empirical_percentile_is_right_inclusive() -> None:
    reference = np.asarray(
        [1.0, 2.0, 3.0, 4.0]
    )
    values = np.asarray(
        [0.0, 1.0, 2.5, 4.0, 5.0]
    )

    observed = empirical_percentile(
        reference,
        values,
    )

    assert observed.tolist() == [
        0.0,
        0.25,
        0.50,
        1.0,
        1.0,
    ]


def test_feature_support_uses_row_q95_abs_standard_z() -> None:
    train = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0],
            "b": [10.0, 11.0, 12.0, 13.0],
        }
    )
    scaler = StandardScaler().fit(train)

    observed = feature_support_statistic(
        train,
        features=["a", "b"],
        scaler=scaler,
        quantile=0.95,
    )

    scaled = np.abs(
        scaler.transform(train)
    )
    expected = np.quantile(
        scaled,
        0.95,
        axis=1,
        method="linear",
    )

    assert np.allclose(
        observed.to_numpy(),
        expected,
    )


def test_support_policy_switches_q99_q95_q90() -> None:
    index = pd.RangeIndex(3)
    pca = pd.Series(
        [8.0, 8.0, 8.0],
        index=index,
    )
    support = pd.Series(
        [0.50, 0.85, 0.99],
        index=index,
    )

    route = support_aware_route(
        pca,
        support,
        policy=SupportPolicy(
            mid_support_percentile=0.80,
            high_support_percentile=0.95,
        ),
        router_thresholds={
            "route_q90": 5.0,
            "route_q95": 7.0,
            "route_q99": 9.0,
        },
    )

    assert route.tolist() == [
        False,
        True,
        True,
    ]


def test_evidence_coverage() -> None:
    route = pd.Series(
        [True, False, True, False]
    )
    evidence = pd.Series(
        [True, True, False, False]
    )

    assert evidence_coverage(
        route,
        evidence,
    ) == pytest.approx(0.5)


def test_candidate_grid_matches_preregistered_values() -> None:
    with open(
        ROOT_CONFIG,
        "r",
        encoding="utf-8",
    ) as handle:
        config = yaml.safe_load(handle)
    grid = config[
        "support_aware_policy"
    ]["calibration_candidate_grid"]

    policies = candidate_grid(
        mid_values=grid[
            "mid_support_percentile"
        ],
        high_values=grid[
            "high_support_percentile"
        ],
    )

    pairs = {
        (
            policy.mid_support_percentile,
            policy.high_support_percentile,
        )
        for policy in policies
    }

    assert pairs == {
        (0.80, 0.95),
        (0.80, 0.975),
        (0.80, 0.99),
        (0.90, 0.95),
        (0.90, 0.975),
        (0.90, 0.99),
        (0.95, 0.975),
        (0.95, 0.99),
    }


def test_select_policy_uses_frozen_tie_breaks() -> None:
    reports = [
        {
            "mid_support_percentile": 0.80,
            "high_support_percentile": 0.95,
            "mean_tcn_invocation_fraction": 0.10,
            "worst_dataset_alert_coverage": 0.85,
            "eligible": True,
        },
        {
            "mid_support_percentile": 0.90,
            "high_support_percentile": 0.99,
            "mean_tcn_invocation_fraction": 0.10,
            "worst_dataset_alert_coverage": 0.90,
            "eligible": True,
        },
        {
            "mid_support_percentile": 0.95,
            "high_support_percentile": 0.99,
            "mean_tcn_invocation_fraction": 0.10,
            "worst_dataset_alert_coverage": 0.90,
            "eligible": True,
        },
    ]

    selected = select_policy(
        reports
    )

    assert (
        selected[
            "mid_support_percentile"
        ]
        == 0.95
    )
    assert (
        selected[
            "high_support_percentile"
        ]
        == 0.99
    )


def test_select_policy_rejects_no_eligible_candidate() -> None:
    with pytest.raises(
        RuntimeError,
        match="No support-aware candidate",
    ):
        select_policy(
            [
                {
                    "eligible": False,
                }
            ]
        )
