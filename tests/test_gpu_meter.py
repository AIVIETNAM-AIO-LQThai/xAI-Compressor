from __future__ import annotations

import pytest

from ml.energy.benchmark import coefficient_of_variation
from ml.energy.gpu_meter import (
    GpuPowerSample,
    integrate_trapezoid,
    summarize_samples,
)


def test_constant_power_integration() -> None:
    samples = [
        GpuPowerSample(0.0, 100.0, 90.0),
        GpuPowerSample(1.0, 100.0, 90.0),
        GpuPowerSample(2.0, 100.0, 90.0),
    ]

    assert integrate_trapezoid(
        samples,
        "instant_w",
    ) == pytest.approx(200.0)

    assert integrate_trapezoid(
        samples,
        "average_w",
    ) == pytest.approx(180.0)


def test_linear_power_integration() -> None:
    samples = [
        GpuPowerSample(0.0, 0.0, 0.0),
        GpuPowerSample(2.0, 100.0, 50.0),
    ]

    assert integrate_trapezoid(
        samples,
        "instant_w",
    ) == pytest.approx(100.0)


def test_reverse_time_is_rejected() -> None:
    samples = [
        GpuPowerSample(1.0, 10.0, 10.0),
        GpuPowerSample(0.0, 10.0, 10.0),
    ]

    with pytest.raises(ValueError):
        integrate_trapezoid(samples)


def test_summary() -> None:
    samples = [
        GpuPowerSample(0.0, 10.0, 20.0),
        GpuPowerSample(1.0, 30.0, 40.0),
    ]

    result = summarize_samples(samples)

    assert result.duration_s == pytest.approx(1.0)
    assert result.sample_count == 2
    assert result.instant_energy_j == pytest.approx(20.0)
    assert result.average_field_energy_j == pytest.approx(30.0)
    assert result.mean_instant_w == pytest.approx(20.0)
    assert result.peak_instant_w == pytest.approx(30.0)


def test_coefficient_of_variation() -> None:
    assert coefficient_of_variation(
        [10.0, 10.0, 10.0]
    ) == pytest.approx(0.0)
