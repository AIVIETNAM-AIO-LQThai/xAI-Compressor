from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ml.detection.common import (
    Scaler,
    fit_scaler,
    transform_frame,
)


@dataclass
class FeatureEnergyDetector:
    features: list[str]
    scaler: Scaler
    scaler_name: str


@dataclass(frozen=True)
class FeatureEnergyExplanation:
    score: pd.Series
    contributions: pd.DataFrame
    normalized_contributions: pd.DataFrame


def fit_feature_energy_detector(
    frame: pd.DataFrame,
    features: list[str],
    *,
    scaler_name: str,
) -> FeatureEnergyDetector:
    scaler = fit_scaler(
        frame,
        features,
        method=scaler_name,
    )

    return FeatureEnergyDetector(
        features=features,
        scaler=scaler,
        scaler_name=scaler_name,
    )


def explain_feature_energy_detector(
    frame: pd.DataFrame,
    detector: FeatureEnergyDetector,
) -> FeatureEnergyExplanation:
    x = transform_frame(
        frame,
        detector.features,
        detector.scaler,
    )

    energy = x ** 2

    feature_count = len(
        detector.features
    )

    contributions_array = (
        energy / feature_count
    )

    contributions = pd.DataFrame(
        contributions_array,
        index=frame.index,
        columns=detector.features,
    )

    score = (
        contributions
        .sum(axis=1)
        .rename("raw_score")
    )

    total_energy = energy.sum(
        axis=1
    )

    normalized_array = np.divide(
        energy,
        total_energy[:, None],
        out=np.zeros_like(
            energy,
            dtype=float,
        ),
        where=(
            total_energy[:, None]
            > 0.0
        ),
    )

    normalized = pd.DataFrame(
        normalized_array,
        index=frame.index,
        columns=detector.features,
    )

    return FeatureEnergyExplanation(
        score=score,
        contributions=contributions,
        normalized_contributions=normalized,
    )


def score_feature_energy_detector(
    frame: pd.DataFrame,
    detector: FeatureEnergyDetector,
) -> pd.Series:
    return explain_feature_energy_detector(
        frame,
        detector,
    ).score
