from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    RobustScaler,
    StandardScaler,
)

Scaler = RobustScaler | StandardScaler


def fit_scaler(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    method: str = "robust",
) -> Scaler:
    if method == "robust":
        scaler: Scaler = RobustScaler(
            with_centering=True,
            with_scaling=True,
            quantile_range=(25.0, 75.0),
        )

    elif method == "standard":
        scaler = StandardScaler(
            with_mean=True,
            with_std=True,
        )

    else:
        raise ValueError(
            "Unknown scaler method: "
            f"{method}. "
            "Expected 'robust' or 'standard'."
        )

    scaler.fit(
        frame.loc[:, columns].astype(float)
    )

    return scaler


def transform_frame(
    frame: pd.DataFrame,
    columns: list[str],
    scaler: Scaler,
) -> np.ndarray:
    values = scaler.transform(
        frame.loc[:, columns].astype(float)
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite scaled feature values."
        )

    return values