from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.preprocessing import StandardScaler

from ml.energy.evidence_aware_routing import (
    ROUTER_FEATURE_NAMES,
    build_router_features,
    chronological_split,
    evidence_coverage,
    logistic_candidate_grid,
    route_from_probability,
    select_candidate,
)


def test_chronological_split_is_60_40_floor() -> None:
    index = pd.date_range(
        "2026-01-01",
        periods=11,
        freq="5min",
    )

    split = chronological_split(
        index,
        fit_fraction=0.60,
    )

    assert len(split.fit_index) == 6
    assert (
        len(split.validation_index)
        == 5
    )
    assert (
        split.fit_index[-1]
        < split.validation_index[0]
    )


def test_build_router_features_are_finite_and_ordered() -> None:
    index = pd.date_range(
        "2026-01-01",
        periods=40,
        freq="5min",
    )

    pca = pd.Series(
        np.linspace(1.0, 20.0, 40),
        index=index,
    )
    feature_support = pd.Series(
        np.linspace(0.5, 3.0, 40),
        index=index,
    )
    latent_support = pd.Series(
        np.linspace(1.0, 4.0, 40),
        index=index,
    )

    target_index = index[12:]
    fit_index = target_index[:16]

    frame, references = (
        build_router_features(
            pca_ewma=pca,
            feature_support_raw=(
                feature_support
            ),
            latent_support_raw=(
                latent_support
            ),
            target_index=target_index,
            fit_index=fit_index,
            rolling_history_bins=12,
        )
    )

    assert (
        frame.columns.tolist()
        == ROUTER_FEATURE_NAMES
    )
    assert np.isfinite(
        frame.to_numpy()
    ).all()
    assert (
        references[
            "pca_q90_threshold"
        ]
        > 0.0
    )


def test_candidate_grid_has_48_candidates() -> None:
    candidates = logistic_candidate_grid(
        C_values=[
            0.01,
            0.1,
            1.0,
            10.0,
        ],
        class_weights=[
            None,
            "balanced",
        ],
        probability_thresholds=[
            0.05,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
        ],
    )

    assert len(candidates) == 48


def test_probability_router_is_not_bounded_by_pca_q90() -> None:
    probabilities = np.asarray(
        [0.9, 0.1],
        dtype=float,
    )

    route = route_from_probability(
        probabilities,
        threshold=0.5,
    )

    # The route depends only on the cheap-model
    # probability; there is no PCA-q90 gate.
    assert route.tolist() == [
        True,
        False,
    ]


def test_evidence_coverage() -> None:
    route = np.asarray(
        [True, False, True, False]
    )
    evidence = np.asarray(
        [True, True, False, False]
    )

    assert evidence_coverage(
        route,
        evidence,
    ) == pytest.approx(0.5)


def test_select_candidate_uses_frozen_tie_breaks() -> None:
    reports = [
        {
            "C": 1.0,
            "class_weight": "balanced",
            "probability_threshold": 0.30,
            "mean_tcn_invocation_fraction": 0.20,
            "worst_dataset_alert_coverage": 0.90,
            "worst_dataset_high_evidence_coverage": 0.85,
            "eligible": True,
        },
        {
            "C": 0.1,
            "class_weight": None,
            "probability_threshold": 0.40,
            "mean_tcn_invocation_fraction": 0.20,
            "worst_dataset_alert_coverage": 0.90,
            "worst_dataset_high_evidence_coverage": 0.85,
            "eligible": True,
        },
    ]

    selected = select_candidate(
        reports
    )

    assert selected["C"] == 0.1
    assert (
        selected["class_weight"]
        is None
    )


def test_preregistration_grid_matches_implementation() -> None:
    path = Path(
        "configs/evidence_aware_compute_routing.yaml"
    )

    config = yaml.safe_load(
        path.read_text(encoding="utf-8")
    )

    grid = config["candidate_grid"]

    candidates = logistic_candidate_grid(
        C_values=grid["C"],
        class_weights=grid[
            "class_weight"
        ],
        probability_thresholds=grid[
            "probability_threshold"
        ],
    )

    assert len(candidates) == int(
        grid["total_candidates"]
    )
    assert (
        config[
            "development_split"
        ]["router_fit_fraction"]
        == 0.60
    )
    assert (
        config[
            "development_split"
        ][
            "router_validation_fraction"
        ]
        == 0.40
    )


def test_standard_scaler_can_pool_router_features() -> None:
    frame = pd.DataFrame(
        np.arange(
            80,
            dtype=float,
        ).reshape(10, 8),
        columns=ROUTER_FEATURE_NAMES,
    )

    scaler = StandardScaler().fit(
        frame.to_numpy()
    )

    transformed = scaler.transform(
        frame.to_numpy()
    )

    assert transformed.shape == (
        10,
        8,
    )
    assert np.isfinite(
        transformed
    ).all()
