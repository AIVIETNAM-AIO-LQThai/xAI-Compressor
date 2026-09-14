from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ml.temporal.energy_aware_training import (
    EnergyAwareTrainingConfig,
    train_energy_aware_tcn,
)
from ml.temporal.sequences import CausalSequenceBatch
from ml.temporal.tcn import TemporalForecastConfig
from scripts.train_energy_aware_tcn import validate_study_config

ROOT = Path(__file__).resolve().parents[1]


def _synthetic_batch() -> CausalSequenceBatch:
    rng = np.random.default_rng(7)
    inputs = rng.normal(size=(20, 4, 3)).astype(np.float32)
    targets = inputs[:, -1, :].copy()

    return CausalSequenceBatch(
        inputs=inputs,
        targets=targets,
        target_index=pd.date_range(
            "2026-01-01",
            periods=20,
            freq="5min",
        ),
        sequence_length=4,
        feature_count=3,
    )


def test_energy_aware_training_uses_chronological_80_20_split() -> None:
    result = train_energy_aware_tcn(
        _synthetic_batch(),
        model_config=TemporalForecastConfig(
            input_dim=3,
            hidden_dim=4,
            kernel_size=3,
            dilations=(1,),
            dropout=0.0,
        ),
        training_config=EnergyAwareTrainingConfig(
            seed=123,
            max_epochs=2,
            batch_size=4,
            learning_rate=1.0e-3,
            weight_decay=1.0e-4,
            validation_fraction=0.20,
            patience=1,
            device="cpu",
        ),
    )

    assert result.fit_samples == 16
    assert result.validation_samples == 4
    assert 1 <= result.epochs_ran <= 2
    assert len(result.train_losses) == result.epochs_ran
    assert len(result.validation_losses) == result.epochs_ran
    assert np.isfinite(result.train_losses).all()
    assert np.isfinite(result.validation_losses).all()


def test_current_preregistered_config_passes_guard() -> None:
    path = (
        ROOT
        / "configs"
        / "energy_aware_hierarchical_intelligence.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    validate_study_config(config)


def test_guard_rejects_scaler_change() -> None:
    path = (
        ROOT
        / "configs"
        / "energy_aware_hierarchical_intelligence.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    changed = deepcopy(config)
    changed["temporal_model"]["scaler"] = "robust"

    with pytest.raises(ValueError, match="StandardScaler"):
        validate_study_config(changed)


def test_guard_rejects_objective_change() -> None:
    path = (
        ROOT
        / "configs"
        / "energy_aware_hierarchical_intelligence.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    changed = deepcopy(config)
    changed["temporal_model"]["training"]["objective"] = "huber"

    with pytest.raises(ValueError, match="next-step MSE"):
        validate_study_config(changed)
