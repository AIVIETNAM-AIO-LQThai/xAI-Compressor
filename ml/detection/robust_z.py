from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RobustZModel:
    features: list[str]
    dropped_features: list[str]
    center: dict[str, float]
    scale: dict[str, float]
    top_k: int


def fit_robust_z(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    top_k: int,
    min_scale: float,
) -> RobustZModel:
    x = frame.loc[:, feature_columns].astype(float)

    center = x.median(axis=0)

    mad = (
        x.sub(center, axis="columns")
        .abs()
        .median(axis=0)
    )

    robust_scale = 1.4826 * mad

    q25 = x.quantile(0.25)
    q75 = x.quantile(0.75)

    iqr_scale = (q75 - q25) / 1.349

    scale = robust_scale.where(
        robust_scale > min_scale,
        iqr_scale,
    )

    active_features = scale[
        scale > min_scale
    ].index.tolist()

    dropped_features = [
        column
        for column in feature_columns
        if column not in active_features
    ]

    if not active_features:
        raise ValueError(
            "No non-constant features remain."
        )

    actual_top_k = min(
        top_k,
        len(active_features),
    )

    return RobustZModel(
        features=active_features,
        dropped_features=dropped_features,
        center={
            column: float(center[column])
            for column in active_features
        },
        scale={
            column: float(scale[column])
            for column in active_features
        },
        top_k=actual_top_k,
    )


def score_robust_z(
    frame: pd.DataFrame,
    model: RobustZModel,
) -> tuple[pd.Series, pd.DataFrame]:
    x = frame.loc[
        :,
        model.features,
    ].astype(float)

    center = pd.Series(model.center)
    scale = pd.Series(model.scale)

    z_scores = (
        x.sub(center, axis="columns")
        .abs()
        .div(scale, axis="columns")
    )

    values = z_scores.to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite Robust-Z values detected."
        )

    sorted_values = np.sort(
        values,
        axis=1,
    )

    top_values = sorted_values[
        :,
        -model.top_k:
    ]

    score = pd.Series(
        top_values.mean(axis=1),
        index=frame.index,
        name="raw_score",
    )

    return score, z_scores


def save_model(
    model: RobustZModel,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            asdict(model),
            indent=2,
        ),
        encoding="utf-8",
    )