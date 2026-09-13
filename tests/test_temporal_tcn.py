from __future__ import annotations

import pytest
import torch

from ml.temporal.tcn import (
    CausalConv1d,
    TCNForecaster,
    TemporalForecastConfig,
)


def test_tcn_output_shape():
    model = TCNForecaster(
        TemporalForecastConfig(
            input_dim=63,
            hidden_dim=32,
            dilations=(1, 2, 4),
            dropout=0.0,
        )
    )

    x = torch.randn(
        8,
        12,
        63,
    )

    prediction = model(
        x
    )

    assert prediction.shape == (
        8,
        63,
    )

    encoded = model.encode(
        x
    )

    assert encoded.shape == (
        8,
        32,
    )


def test_causal_convolution_has_no_future_leakage():
    torch.manual_seed(7)

    layer = CausalConv1d(
        2,
        3,
        kernel_size=3,
        dilation=1,
    )

    original = torch.randn(
        1,
        2,
        8,
    )

    changed = original.clone()

    changed[
        :,
        :,
        5:,
    ] += 100.0

    first = layer(
        original
    )

    second = layer(
        changed
    )

    torch.testing.assert_close(
        first[:, :, :5],
        second[:, :, :5],
    )


def test_tcn_rejects_wrong_feature_count():
    model = TCNForecaster(
        TemporalForecastConfig(
            input_dim=5,
        )
    )

    with pytest.raises(
        ValueError,
        match="feature dimension",
    ):
        model(
            torch.randn(
                2,
                12,
                4,
            )
        )
