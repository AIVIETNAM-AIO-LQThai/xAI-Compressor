from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ml.detection.common import Scaler, fit_scaler, transform_frame


@dataclass
class PCADetector:
    features: list[str]
    scaler: Scaler
    scaler_name: str
    model: PCA


def fit_pca_detector(
    frame: pd.DataFrame,
    features: list[str],
    *,
    variance_retained: float = 0.95,
    scaler_name: str = "robust",
) -> PCADetector:
    scaler = fit_scaler(
        frame,
        features,
        method=scaler_name,
    )

    x = transform_frame(
        frame,
        features,
        scaler,
    )

    model = PCA(
        n_components=variance_retained,
        svd_solver="full",
    )

    model.fit(x)

    return PCADetector(
        features=features,
        scaler=scaler,
        scaler_name=scaler_name,
        model=model,
    )


def score_pca_detector(
    frame: pd.DataFrame,
    detector: PCADetector,
) -> pd.Series:
    x = transform_frame(
        frame,
        detector.features,
        detector.scaler,
    )

    encoded = detector.model.transform(x)

    reconstructed = (
        detector.model.inverse_transform(encoded)
    )

    residual = x - reconstructed

    score = np.mean(
        residual ** 2,
        axis=1,
    )

    return pd.Series(
        score,
        index=frame.index,
        name="raw_score",
    )