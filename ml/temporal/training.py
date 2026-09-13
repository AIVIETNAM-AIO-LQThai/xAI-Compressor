from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from ml.temporal.sequences import (
    CausalSequenceBatch,
)
from ml.temporal.tcn import (
    TCNForecaster,
    TemporalForecastConfig,
)


@dataclass(frozen=True)
class TemporalTrainingConfig:
    seed: int = 42
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    huber_delta: float = 1.0
    validation_fraction: float = 0.15
    patience: int = 10
    min_delta: float = 1.0e-5
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError(
                "epochs must be positive."
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        if self.learning_rate <= 0.0:
            raise ValueError(
                "learning_rate must be positive."
            )

        if self.weight_decay < 0.0:
            raise ValueError(
                "weight_decay cannot be negative."
            )

        if self.huber_delta <= 0.0:
            raise ValueError(
                "huber_delta must be positive."
            )

        if not (
            0.0
            < self.validation_fraction
            < 0.5
        ):
            raise ValueError(
                "validation_fraction must be "
                "strictly between 0 and 0.5."
            )

        if self.patience <= 0:
            raise ValueError(
                "patience must be positive."
            )

        if self.min_delta < 0.0:
            raise ValueError(
                "min_delta cannot be negative."
            )


@dataclass(frozen=True)
class TemporalTrainingResult:
    model: TCNForecaster
    train_losses: tuple[float, ...]
    validation_losses: tuple[float, ...]
    best_epoch: int
    epochs_ran: int
    device: str
    fit_samples: int
    validation_samples: int


def resolve_device(
    requested: str,
) -> torch.device:
    if requested == "auto":
        return torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(
        requested
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is "
            "not available."
        )

    return device


def set_training_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    if hasattr(
        torch.backends,
        "cudnn",
    ):
        torch.backends.cudnn.deterministic = (
            True
        )
        torch.backends.cudnn.benchmark = (
            False
        )


def _mean_loss(
    model: TCNForecaster,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()

    weighted_loss = 0.0
    sample_count = 0

    with torch.inference_mode():
        for inputs, targets in loader:
            inputs = inputs.to(
                device
            )
            targets = targets.to(
                device
            )

            predictions = model(
                inputs
            )

            loss = criterion(
                predictions,
                targets,
            )

            count = int(
                inputs.shape[0]
            )

            weighted_loss += (
                float(loss.item())
                * count
            )
            sample_count += count

    if sample_count == 0:
        raise ValueError(
            "Validation loader is empty."
        )

    return (
        weighted_loss
        / sample_count
    )


def train_tcn_forecaster(
    batch: CausalSequenceBatch,
    *,
    model_config: TemporalForecastConfig,
    training_config: TemporalTrainingConfig,
) -> TemporalTrainingResult:
    if (
        batch.feature_count
        != model_config.input_dim
    ):
        raise ValueError(
            "Sequence feature count does not "
            "match model input_dim."
        )

    sample_count = len(
        batch.inputs
    )

    validation_count = max(
        1,
        round(
            sample_count
            * training_config
            .validation_fraction
        ),
    )

    fit_count = (
        sample_count
        - validation_count
    )

    if fit_count < 2:
        raise ValueError(
            "Not enough temporal samples "
            "for chronological fit/validation."
        )

    set_training_seed(
        training_config.seed
    )

    device = resolve_device(
        training_config.device
    )

    model = TCNForecaster(
        model_config
    ).to(device)

    fit_dataset = TensorDataset(
        torch.from_numpy(
            batch.inputs[:fit_count]
        ),
        torch.from_numpy(
            batch.targets[:fit_count]
        ),
    )

    validation_dataset = TensorDataset(
        torch.from_numpy(
            batch.inputs[fit_count:]
        ),
        torch.from_numpy(
            batch.targets[fit_count:]
        ),
    )

    generator = torch.Generator()
    generator.manual_seed(
        training_config.seed
    )

    fit_loader = DataLoader(
        fit_dataset,
        batch_size=(
            training_config.batch_size
        ),
        shuffle=True,
        generator=generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=(
            training_config.batch_size
        ),
        shuffle=False,
    )

    criterion = nn.HuberLoss(
        delta=(
            training_config.huber_delta
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=(
            training_config.learning_rate
        ),
        weight_decay=(
            training_config.weight_decay
        ),
    )

    train_losses: list[float] = []
    validation_losses: list[
        float
    ] = []

    best_validation = float(
        "inf"
    )
    best_epoch = 0
    best_state: dict[
        str,
        torch.Tensor,
    ] | None = None
    epochs_without_improvement = 0

    for epoch in range(
        1,
        training_config.epochs + 1,
    ):
        model.train()

        weighted_loss = 0.0
        seen = 0

        for inputs, targets in fit_loader:
            inputs = inputs.to(
                device
            )
            targets = targets.to(
                device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            predictions = model(
                inputs
            )

            loss = criterion(
                predictions,
                targets,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            count = int(
                inputs.shape[0]
            )

            weighted_loss += (
                float(loss.item())
                * count
            )
            seen += count

        train_loss = (
            weighted_loss
            / seen
        )

        validation_loss = _mean_loss(
            model,
            validation_loader,
            criterion=criterion,
            device=device,
        )

        train_losses.append(
            float(train_loss)
        )
        validation_losses.append(
            float(validation_loss)
        )

        improved = (
            validation_loss
            < (
                best_validation
                - training_config.min_delta
            )
        )

        if improved:
            best_validation = (
                validation_loss
            )
            best_epoch = epoch
            best_state = copy.deepcopy(
                {
                    key: value
                    .detach()
                    .cpu()
                    for key, value
                    in model
                    .state_dict()
                    .items()
                }
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= training_config.patience
        ):
            break

    if best_state is None:
        raise RuntimeError(
            "TCN training did not produce "
            "a valid model state."
        )

    model.load_state_dict(
        best_state
    )
    model = model.cpu()

    return TemporalTrainingResult(
        model=model,
        train_losses=tuple(
            train_losses
        ),
        validation_losses=tuple(
            validation_losses
        ),
        best_epoch=best_epoch,
        epochs_ran=len(
            train_losses
        ),
        device=str(device),
        fit_samples=fit_count,
        validation_samples=(
            validation_count
        ),
    )
