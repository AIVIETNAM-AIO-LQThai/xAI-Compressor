from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ml.representation.regime_geometry import (
    define_current_pressure_regimes,
    fit_pca_95,
    neighbor_regime_purity,
    reconstruct_fixed_dimension,
    regime_conditioned_pca,
)


def test_regime_thresholds_use_fit_only():
    fit = pd.DataFrame(
        {
            "motor": [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "pressure": [
                10.0,
                11.0,
                12.0,
                13.0,
                14.0,
            ],
        }
    )

    audit = pd.DataFrame(
        {
            "motor": [
                100.0,
            ],
            "pressure": [
                100.0,
            ],
        }
    )

    thresholds, _, audit_labels = (
        define_current_pressure_regimes(
            fit_frame=fit,
            audit_frame=audit,
            current_feature="motor",
            pressure_feature="pressure",
            high_quantile=0.8,
            label_map={
                "low_low": "ll",
                "high_current": "hc",
                "high_pressure": "hp",
                "high_both": "hh",
            },
        )
    )

    assert thresholds[
        "current_threshold"
    ] < 100.0

    assert thresholds[
        "pressure_threshold"
    ] < 100.0

    assert audit_labels.iloc[
        0
    ] == "hh"


def test_fixed_dimension_reconstruction_shape():
    rng = np.random.default_rng(
        10
    )

    x = rng.normal(
        size=(
            200,
            5,
        )
    )

    model = PCA(
        svd_solver="full"
    ).fit(x)

    reconstructed = (
        reconstruct_fixed_dimension(
            model,
            x[:10],
            component_count=2,
        )
    )

    assert reconstructed.shape == (
        10,
        5,
    )


def test_regime_conditioning_recovers_two_distinct_planes():
    rng = np.random.default_rng(
        11
    )

    n_fit = 300
    n_audit = 80

    def plane_a(n: int) -> np.ndarray:
        u = rng.normal(
            size=n
        )
        v = rng.normal(
            size=n
        )

        return np.column_stack(
            [
                u,
                v,
                0.02
                * rng.normal(
                    size=n
                ),
            ]
        )

    def plane_b(n: int) -> np.ndarray:
        u = rng.normal(
            size=n
        )
        v = rng.normal(
            size=n
        )

        return np.column_stack(
            [
                0.02
                * rng.normal(
                    size=n
                ),
                u,
                v,
            ]
        )

    fit = np.vstack(
        [
            plane_a(n_fit),
            plane_b(n_fit),
        ]
    )

    audit = np.vstack(
        [
            plane_a(n_audit),
            plane_b(n_audit),
        ]
    )

    fit_labels = pd.Series(
        ["a"] * n_fit
        + ["b"] * n_fit
    )

    audit_labels = pd.Series(
        ["a"] * n_audit
        + ["b"] * n_audit
    )

    global_fit = fit_pca_95(
        fit,
        variance_target=0.60,
    )

    result = regime_conditioned_pca(
        fit_scaled=fit,
        audit_scaled=audit,
        fit_regime=fit_labels,
        audit_regime=audit_labels,
        global_fit=global_fit,
        variance_target=0.95,
        minimum_fit_rows=50,
        minimum_audit_rows=20,
    )

    eligible = result.loc[
        result[
            "eligible"
        ]
    ]

    assert (
        eligible[
            "median_regime_global_ratio"
        ]
        < 0.2
    ).all()


def test_neighbor_purity_is_high_for_separated_clusters():
    rng = np.random.default_rng(
        12
    )

    fit_a = rng.normal(
        loc=-5.0,
        scale=0.2,
        size=(
            100,
            2,
        ),
    )

    fit_b = rng.normal(
        loc=5.0,
        scale=0.2,
        size=(
            100,
            2,
        ),
    )

    audit_a = rng.normal(
        loc=-5.0,
        scale=0.2,
        size=(
            20,
            2,
        ),
    )

    audit_b = rng.normal(
        loc=5.0,
        scale=0.2,
        size=(
            20,
            2,
        ),
    )

    fit = np.vstack(
        [
            fit_a,
            fit_b,
        ]
    )

    audit = np.vstack(
        [
            audit_a,
            audit_b,
        ]
    )

    scaler = StandardScaler().fit(
        fit
    )

    result = neighbor_regime_purity(
        fit_scaled=scaler.transform(
            fit
        ),
        audit_scaled=scaler.transform(
            audit
        ),
        fit_regime=pd.Series(
            ["a"] * 100
            + ["b"] * 100
        ),
        audit_regime=pd.Series(
            ["a"] * 20
            + ["b"] * 20
        ),
        k=20,
    )

    assert result[
        "mean_same_regime_fraction"
    ] > 0.95
