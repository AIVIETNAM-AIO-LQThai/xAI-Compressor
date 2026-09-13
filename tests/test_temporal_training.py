from __future__ import annotations

import numpy as np
import pandas as pd

from ml.temporal.sequences import (
    build_causal_sequences,
)
from ml.temporal.tcn import (
    TemporalForecastConfig,
)
from ml.temporal.training import (
    TemporalTrainingConfig,
    train_tcn_forecaster,
)


def test_tcn_training_runs_on_small_sequence():
    rng = np.random.default_rng(
        42
    )

    values = rng.normal(
        size=(80, 4)
    ).astype(
        np.float32
    )

    timestamps = pd.date_range(
        "2020-01-01",
        periods=80,
        freq="5min",
    )

    batch = build_causal_sequences(
        values,
        timestamps,
        sequence_length=6,
        expected_step=pd.Timedelta(
            minutes=5
        ),
    )

    result = train_tcn_forecaster(
        batch,
        model_config=(
            TemporalForecastConfig(
                input_dim=4,
                hidden_dim=8,
                dilations=(1, 2),
                dropout=0.0,
            )
        ),
        training_config=(
            TemporalTrainingConfig(
                seed=42,
                epochs=2,
                batch_size=16,
                learning_rate=0.001,
                validation_fraction=0.2,
                patience=2,
                device="cpu",
            )
        ),
    )

    assert result.epochs_ran == 2
    assert result.best_epoch >= 1
    assert result.fit_samples > 0
    assert result.validation_samples > 0
    assert len(
        result.train_losses
    ) == 2
    assert len(
        result.validation_losses
    ) == 2
