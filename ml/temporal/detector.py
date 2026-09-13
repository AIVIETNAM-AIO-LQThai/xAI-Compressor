from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

from ml.detection.common import transform_frame
from ml.temporal.sequences import (
    CausalSequenceBatch,
    build_causal_sequences,
)
from ml.temporal.tcn import TCNForecaster


@dataclass
class TemporalDetector:
    features: list[str]
    scaler: RobustScaler
    model: TCNForecaster
    sequence_length: int
    bin_minutes: int


def prepare_temporal_batch(
    frame: pd.DataFrame,
    *,
    features: list[str],
    scaler: RobustScaler,
    sequence_length: int,
    bin_minutes: int,
) -> CausalSequenceBatch:
    values = transform_frame(
        frame,
        features,
        scaler,
    )

    return build_causal_sequences(
        values,
        pd.DatetimeIndex(frame.index),
        sequence_length=sequence_length,
        expected_step=pd.Timedelta(
            minutes=bin_minutes
        ),
    )


def score_temporal_detector(
    frame: pd.DataFrame,
    detector: TemporalDetector,
    *,
    device: str | torch.device,
    batch_size: int = 512,
) -> pd.Series:
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    batch = prepare_temporal_batch(
        frame,
        features=detector.features,
        scaler=detector.scaler,
        sequence_length=(
            detector.sequence_length
        ),
        bin_minutes=detector.bin_minutes,
    )

    torch_device = torch.device(
        device
    )

    model = detector.model.to(
        torch_device
    )
    model.eval()

    scores: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(
            0,
            len(batch.inputs),
            batch_size,
        ):
            stop = min(
                len(batch.inputs),
                start + batch_size,
            )

            inputs = torch.from_numpy(
                batch.inputs[start:stop]
            ).to(torch_device)

            targets = torch.from_numpy(
                batch.targets[start:stop]
            ).to(torch_device)

            predictions = model(
                inputs
            )

            residual = (
                targets
                - predictions
            )

            batch_scores = torch.mean(
                residual.square(),
                dim=1,
            )

            scores.append(
                batch_scores
                .detach()
                .cpu()
                .numpy()
            )

    values = np.concatenate(
        scores
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Temporal anomaly scores contain "
            "non-finite values."
        )

    return pd.Series(
        values,
        index=batch.target_index,
        name="raw_score",
        dtype=float,
    )
