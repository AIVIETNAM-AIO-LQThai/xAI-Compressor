from __future__ import annotations

import numpy as np
import pandas as pd

from ml.detection.regime_pca import (
    REGIME_NAMES,
    assign_regimes,
    explain_regime_pca_detector,
    fit_regime_pca_detector,
    score_regime_pca_detector,
)

FEATURES = [
    "motor_current__mean",
    "reservoirs__mean",
    "signal_a__mean",
    "signal_b__mean",
    "signal_c__mean",
]


def _training_frame() -> pd.DataFrame:
    rng = np.random.default_rng(
        1234
    )

    rows = []

    for current_high, pressure_high in (
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ):
        for _ in range(120):
            current = (
                10.0
                if current_high
                else 0.0
            ) + rng.normal(
                scale=0.2
            )

            pressure = (
                10.0
                if pressure_high
                else 0.0
            ) + rng.normal(
                scale=0.2
            )

            latent = rng.normal()

            if current_high:
                signal_a = (
                    2.0 * latent
                    + rng.normal(
                        scale=0.05
                    )
                )
            else:
                signal_a = (
                    latent
                    + rng.normal(
                        scale=0.05
                    )
                )

            if pressure_high:
                signal_b = (
                    -1.5 * latent
                    + rng.normal(
                        scale=0.05
                    )
                )
            else:
                signal_b = (
                    0.5 * latent
                    + rng.normal(
                        scale=0.05
                    )
                )

            signal_c = (
                signal_a
                + signal_b
                + rng.normal(
                    scale=0.05
                )
            )

            rows.append(
                {
                    "motor_current__mean": (
                        current
                    ),
                    "reservoirs__mean": (
                        pressure
                    ),
                    "signal_a__mean": (
                        signal_a
                    ),
                    "signal_b__mean": (
                        signal_b
                    ),
                    "signal_c__mean": (
                        signal_c
                    ),
                }
            )

    return pd.DataFrame(
        rows,
        index=pd.date_range(
            "2020-01-01",
            periods=len(rows),
            freq="5min",
        ),
    )


def test_assign_regimes_covers_all_four_labels():
    frame = pd.DataFrame(
        {
            "motor_current__mean": [
                0.0,
                10.0,
                0.0,
                10.0,
            ],
            "reservoirs__mean": [
                0.0,
                0.0,
                10.0,
                10.0,
            ],
        }
    )

    regimes = assign_regimes(
        frame,
        current_feature=(
            "motor_current__mean"
        ),
        pressure_feature=(
            "reservoirs__mean"
        ),
        current_threshold=5.0,
        pressure_threshold=5.0,
    )

    assert set(regimes) == set(
        REGIME_NAMES
    )


def test_fit_uses_one_fixed_component_count():
    frame = _training_frame()

    detector = fit_regime_pca_detector(
        frame,
        FEATURES,
        current_feature=(
            "motor_current__mean"
        ),
        pressure_feature=(
            "reservoirs__mean"
        ),
        high_quantile=0.80,
        variance_retained=0.95,
    )

    counts = {
        int(model.n_components_)
        for model in detector.models.values()
    }

    assert counts == {
        detector.component_count
    }

    assert detector.scaler_name == (
        "standard"
    )


def test_score_and_explanation_are_exactly_additive():
    frame = _training_frame()

    detector = fit_regime_pca_detector(
        frame,
        FEATURES,
        current_feature=(
            "motor_current__mean"
        ),
        pressure_feature=(
            "reservoirs__mean"
        ),
        high_quantile=0.80,
        variance_retained=0.95,
    )

    sample = frame.iloc[
        -40:
    ]

    scores = score_regime_pca_detector(
        sample,
        detector,
    )

    explanation = (
        explain_regime_pca_detector(
            sample,
            detector,
        )
    )

    np.testing.assert_allclose(
        scores.to_numpy(),
        explanation.score.to_numpy(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    np.testing.assert_allclose(
        explanation.contributions.sum(
            axis=1
        ).to_numpy(),
        scores.to_numpy(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_router_thresholds_are_training_derived():
    frame = _training_frame()

    detector = fit_regime_pca_detector(
        frame,
        FEATURES,
        current_feature=(
            "motor_current__mean"
        ),
        pressure_feature=(
            "reservoirs__mean"
        ),
        high_quantile=0.80,
        variance_retained=0.95,
    )

    expected_current = float(
        frame[
            "motor_current__mean"
        ].quantile(0.80)
    )

    expected_pressure = float(
        frame[
            "reservoirs__mean"
        ].quantile(0.80)
    )

    assert detector.current_threshold == (
        expected_current
    )

    assert detector.pressure_threshold == (
        expected_pressure
    )
