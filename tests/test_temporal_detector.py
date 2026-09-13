from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ml.detection.common import (
    fit_scaler,
)
from ml.temporal.detector import (
    TemporalDetector,
    prepare_temporal_batch,
    score_temporal_detector,
)
from ml.temporal.tcn import (
    TCNForecaster,
    TemporalForecastConfig,
)


def _frame() -> pd.DataFrame:
    index = pd.date_range(
        "2020-01-01",
        periods=30,
        freq="5min",
    )

    return pd.DataFrame(
        {
            "a": np.linspace(
                0.0,
                1.0,
                len(index),
            ),
            "b": np.sin(
                np.linspace(
                    0.0,
                    2.0,
                    len(index),
                )
            ),
            "c": np.cos(
                np.linspace(
                    0.0,
                    2.0,
                    len(index),
                )
            ),
        },
        index=index,
    )


def test_temporal_detector_scores_target_times():
    frame = _frame()
    features = [
        "a",
        "b",
        "c",
    ]

    scaler = fit_scaler(
        frame,
        features,
        method="robust",
    )

    model = TCNForecaster(
        TemporalForecastConfig(
            input_dim=3,
            hidden_dim=8,
            dilations=(1, 2),
            dropout=0.0,
        )
    )

    detector = TemporalDetector(
        features=features,
        scaler=scaler,
        model=model,
        sequence_length=4,
        bin_minutes=5,
    )

    scores = score_temporal_detector(
        frame,
        detector,
        device=torch.device(
            "cpu"
        ),
        batch_size=8,
    )

    assert len(scores) == 26

    assert (
        scores.index[0]
        == frame.index[4]
    )

    assert np.isfinite(
        scores.to_numpy()
    ).all()


def test_prepare_batch_uses_scaled_features():
    frame = _frame()
    features = [
        "a",
        "b",
        "c",
    ]

    scaler = fit_scaler(
        frame,
        features,
        method="robust",
    )

    batch = prepare_temporal_batch(
        frame,
        features=features,
        scaler=scaler,
        sequence_length=4,
        bin_minutes=5,
    )

    assert batch.inputs.shape == (
        26,
        4,
        3,
    )

    assert batch.targets.shape == (
        26,
        3,
    )
