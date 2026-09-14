from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExplanationConcentration:
    group_count: int
    nonzero_group_count: int
    dominant_group: str
    dominant_share: float
    top1_concentration: float
    top3_concentration: float
    top5_concentration: float
    shannon_entropy_nats: float
    normalized_entropy: float
    effective_group_count: float
    herfindahl_index: float


def explanation_concentration(
    shares: Mapping[str, float],
) -> ExplanationConcentration:
    if not shares:
        raise ValueError("shares cannot be empty.")

    names = list(shares.keys())
    values = np.asarray(
        [float(shares[name]) for name in names],
        dtype=float,
    )

    if not np.isfinite(values).all():
        raise ValueError("shares must be finite.")

    if (values < 0.0).any():
        raise ValueError("shares cannot be negative.")

    total = float(values.sum())

    if total <= 0.0:
        raise ValueError(
            "shares must contain positive mass."
        )

    probabilities = values / total
    order = np.argsort(probabilities)[::-1]
    sorted_probabilities = probabilities[order]
    positive = probabilities[probabilities > 0.0]

    entropy = float(
        -np.sum(positive * np.log(positive))
    )

    group_count = len(probabilities)

    if group_count <= 1:
        normalized_entropy = 0.0
    else:
        normalized_entropy = (
            entropy / math.log(group_count)
        )

    dominant_position = int(order[0])

    def top_k(k: int) -> float:
        return float(
            sorted_probabilities[: min(k, group_count)].sum()
        )

    return ExplanationConcentration(
        group_count=group_count,
        nonzero_group_count=int(
            np.count_nonzero(probabilities > 0.0)
        ),
        dominant_group=names[dominant_position],
        dominant_share=float(
            probabilities[dominant_position]
        ),
        top1_concentration=top_k(1),
        top3_concentration=top_k(3),
        top5_concentration=top_k(5),
        shannon_entropy_nats=entropy,
        normalized_entropy=float(normalized_entropy),
        effective_group_count=float(math.exp(entropy)),
        herfindahl_index=float(
            np.square(probabilities).sum()
        ),
    )
