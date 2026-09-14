from __future__ import annotations

import math

import pytest

from ml.explainability.concentration import (
    explanation_concentration,
)


def test_uniform_contributions_have_full_effective_count():
    metrics = explanation_concentration(
        {
            "a": 1.0,
            "b": 1.0,
            "c": 1.0,
            "d": 1.0,
        }
    )

    assert metrics.top1_concentration == pytest.approx(0.25)
    assert metrics.top3_concentration == pytest.approx(0.75)
    assert metrics.effective_group_count == pytest.approx(4.0)
    assert metrics.normalized_entropy == pytest.approx(1.0)
    assert metrics.herfindahl_index == pytest.approx(0.25)


def test_concentrated_explanation_is_detected():
    metrics = explanation_concentration(
        {
            "dominant": 0.8,
            "secondary": 0.1,
            "third": 0.1,
            "zero": 0.0,
        }
    )

    expected_entropy = -(
        0.8 * math.log(0.8)
        + 0.1 * math.log(0.1)
        + 0.1 * math.log(0.1)
    )

    assert metrics.dominant_group == "dominant"
    assert metrics.top1_concentration == pytest.approx(0.8)
    assert metrics.top3_concentration == pytest.approx(1.0)
    assert metrics.shannon_entropy_nats == pytest.approx(
        expected_entropy
    )
    assert metrics.effective_group_count < 3.0


def test_concentration_normalizes_input_mass():
    metrics = explanation_concentration(
        {
            "a": 8.0,
            "b": 1.0,
            "c": 1.0,
        }
    )

    assert metrics.dominant_share == pytest.approx(0.8)


@pytest.mark.parametrize(
    "shares",
    [
        {},
        {"a": 0.0},
        {"a": -0.1, "b": 1.1},
        {"a": float("nan")},
    ],
)
def test_invalid_contributions_are_rejected(
    shares: dict[str, float],
):
    with pytest.raises(ValueError):
        explanation_concentration(shares)
