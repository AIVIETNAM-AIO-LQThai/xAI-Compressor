from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.temporal.sequences import CausalSequenceBatch
from ml.temporal.tcn import TCNForecaster, TemporalForecastConfig
from ml.temporal.training import resolve_device, set_training_seed


@dataclass(frozen=True)
class EnergyAwareTrainingConfig:
    seed: int
    max_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    validation_fraction: float
    patience: int
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative.")
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError(
                "validation_fraction must be strictly between 0 and 0.5."
            )
        if self.patience <= 0:
            raise ValueError("patience must be positive.")


@dataclass(frozen=True)
class EnergyAwareTrainingResult:
    model: TCNForecaster
    train_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    best_epoch: int
    epochs_ran: int
    device: str
    fit_samples: int
    validation_samples: int


def _mean_mse(
    model: TCNForecaster,
    loader: DataLoader,
    *,
    device: torch.device,
) -> float:
    model.eval()
    criterion = nn.MSELoss()

    weighted_loss = 0.0
    sample_count = 0

    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)
            loss = criterion(predictions, targets)

            count = int(inputs.shape[0])
            weighted_loss += float(loss.item()) * count
            sample_count += count

    if sample_count == 0:
        raise ValueError("Validation loader is empty.")

    return weighted_loss / sample_count


def train_energy_aware_tcn(
    batch: CausalSequenceBatch,
    *,
    model_config: TemporalForecastConfig,
    training_config: EnergyAwareTrainingConfig,
) -> EnergyAwareTrainingResult:
    """Train the preregistered next-step TCN with MSE and AdamW.

    The chronological tail of the TRAIN-derived sequence batch is used
    for validation. CALIBRATION and TEST are not accepted by this API.
    """

    if batch.feature_count != model_config.input_dim:
        raise ValueError(
            "Sequence feature count does not match model input_dim."
        )

    sample_count = len(batch.inputs)
    validation_count = max(
        1,
        round(sample_count * training_config.validation_fraction),
    )
    fit_count = sample_count - validation_count

    if fit_count < 2:
        raise ValueError(
            "Not enough temporal samples for chronological fit/validation."
        )

    set_training_seed(training_config.seed)
    device = resolve_device(training_config.device)

    model = TCNForecaster(model_config).to(device)

    fit_dataset = TensorDataset(
        torch.from_numpy(batch.inputs[:fit_count]),
        torch.from_numpy(batch.targets[:fit_count]),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(batch.inputs[fit_count:]),
        torch.from_numpy(batch.targets[fit_count:]),
    )

    generator = torch.Generator()
    generator.manual_seed(training_config.seed)

    fit_loader = DataLoader(
        fit_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training_config.batch_size,
        shuffle=False,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    train_losses: list[float] = []
    validation_losses: list[float] = []

    best_validation = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, training_config.max_epochs + 1):
        model.train()

        weighted_loss = 0.0
        seen = 0

        for inputs, targets in fit_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            count = int(inputs.shape[0])
            weighted_loss += float(loss.item()) * count
            seen += count

        if seen == 0:
            raise ValueError("Training loader is empty.")

        train_loss = weighted_loss / seen
        validation_loss = _mean_mse(
            model,
            validation_loader,
            device=device,
        )

        train_losses.append(float(train_loss))
        validation_losses.append(float(validation_loss))

        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= training_config.patience:
            break

    if best_state is None:
        raise RuntimeError("Training never produced a best model state.")

    model.load_state_dict(best_state)
    model.eval()

    return EnergyAwareTrainingResult(
        model=model,
        train_losses=tuple(train_losses),
        validation_losses=tuple(validation_losses),
        best_epoch=best_epoch,
        epochs_ran=len(train_losses),
        device=str(device),
        fit_samples=fit_count,
        validation_samples=validation_count,
    )
