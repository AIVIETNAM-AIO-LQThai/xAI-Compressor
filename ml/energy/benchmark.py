from __future__ import annotations

import math
import statistics
from collections.abc import Sequence


def coefficient_of_variation(
    values: Sequence[float],
) -> float:
    if len(values) < 2:
        raise ValueError("at least two values are required")

    mean = statistics.fmean(values)

    if math.isclose(mean, 0.0, abs_tol=1e-15):
        raise ValueError(
            "coefficient of variation is undefined for zero mean"
        )

    return statistics.stdev(values) / mean
