from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import (
    IsolationForest,
)
from sklearn.preprocessing import RobustScaler

from ml.detection.common import (
    fit_scaler,
    transform_frame,
)


@dataclass
class IsolationForestDetector:
    features: list[str]
    scaler: RobustScaler
    model: IsolationForest


def fit_isolation_forest(
    frame: pd.DataFrame,
    features: list[str],
    *,
    n_estimators: int = 300,
    random_state: int = 42,
) -> IsolationForestDetector:
    scaler = fit_scaler(
        frame,
        features,
    )

    x = transform_frame(
        frame,
        features,
        scaler,
    )

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(x)

    return IsolationForestDetector(
        features=features,
        scaler=scaler,
        model=model,
    )


def score_isolation_forest(
    frame: pd.DataFrame,
    detector: IsolationForestDetector,
) -> pd.Series:
    x = transform_frame(
        frame,
        detector.features,
        detector.scaler,
    )

    # sklearn assigns lower values to
    # more abnormal observations.
    # Negate so AeroXAI consistently uses:
    # higher score = more anomalous.
    score = -detector.model.score_samples(x)

    return pd.Series(
        score,
        index=frame.index,
        name="raw_score",
    )