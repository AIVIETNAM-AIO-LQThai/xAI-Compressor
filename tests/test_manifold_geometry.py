from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from ml.representation.manifold_geometry import (
    chronological_split,
    fit_global_pca,
    local_pca_audit,
    relationship_shape_audit,
    two_nearest_neighbor_dimension,
)


def test_chronological_split_preserves_order():
    index = pd.date_range(
        "2020-01-01",
        periods=10,
        freq="h",
    )

    frame = pd.DataFrame(
        {
            "x": np.arange(
                10,
                dtype=float,
            )
        },
        index=index,
    )

    fit, audit = chronological_split(
        frame,
        fit_fraction=0.8,
    )

    assert len(fit) == 8
    assert len(audit) == 2
    assert fit.index.max() < audit.index.min()


def test_participation_ratio_matches_equal_spectrum():
    rng = np.random.default_rng(123)
    x = rng.normal(
        size=(
            5000,
            4,
        )
    )

    result = fit_global_pca(
        x,
        variance_targets=[
            0.95
        ],
    )

    assert result.participation_ratio == pytest.approx(
        4.0,
        rel=0.08,
    )


def test_two_nn_dimension_is_reasonable_for_2d_cloud():
    rng = np.random.default_rng(456)
    x = rng.uniform(
        size=(
            2500,
            2,
        )
    )

    result = (
        two_nearest_neighbor_dimension(
            x
        )
    )

    assert 1.4 < float(
        result[
            "estimate"
        ]
    ) < 2.8


def test_local_pca_can_fit_curved_manifold_better():
    x_reference = np.linspace(
        -2.0,
        2.0,
        800,
    )

    reference = np.column_stack(
        [
            x_reference,
            x_reference**2,
        ]
    )

    x_audit = np.linspace(
        -1.9,
        1.9,
        120,
    ) + 0.001

    audit = np.column_stack(
        [
            x_audit,
            x_audit**2,
        ]
    )

    global_model = PCA(
        svd_solver="full"
    ).fit(reference)

    points, summary = local_pca_audit(
        reference,
        audit,
        global_model=global_model,
        latent_dimension=1,
        neighbor_counts=[
            25
        ],
    )

    assert (
        summary[
            "25"
        ][
            "median_local_global_ratio"
        ]
        < 0.2
    )

    assert (
        points[
            "local_pca_mse_k25"
        ].median()
        < points[
            "global_pca_mse"
        ].median()
    )


def test_quadratic_model_beats_linear():
    fit_x = np.linspace(
        -2.0,
        1.0,
        200,
    )

    audit_x = np.linspace(
        1.01,
        2.0,
        80,
    )

    fit = pd.DataFrame(
        {
            "x": fit_x,
            "y": fit_x**2,
        }
    )

    audit = pd.DataFrame(
        {
            "x": audit_x,
            "y": audit_x**2,
        }
    )

    result = relationship_shape_audit(
        fit,
        audit,
        pairs=[
            {
                "x": "x",
                "y": "y",
            }
        ],
        spline_knots=6,
    )

    linear_mse = float(
        result.loc[
            result[
                "model"
            ]
            == "linear",
            "mse",
        ].iloc[0]
    )

    quadratic_mse = float(
        result.loc[
            result[
                "model"
            ]
            == "quadratic",
            "mse",
        ].iloc[0]
    )

    assert quadratic_mse < (
        linear_mse
        * 1.0e-6
    )
