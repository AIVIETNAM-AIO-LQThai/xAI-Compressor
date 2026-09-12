from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


def fit_scaler(
    frame: pd.DataFrame,
    columns: list[str],
) -> RobustScaler:
    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(25.0, 75.0),
    )

    scaler.fit(
        frame.loc[:, columns].astype(float)
    )

    return scaler


def transform_frame(
    frame: pd.DataFrame,
    columns: list[str],
    scaler: RobustScaler,
) -> np.ndarray:
    values = scaler.transform(
        frame.loc[:, columns].astype(float)
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite scaled feature values."
        )

    return values