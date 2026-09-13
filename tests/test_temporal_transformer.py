from __future__ import annotations

import pytest
import torch

from ml.temporal.transformer import (
    TemporalTransformerConfig,
    TemporalTransformerForecaster,
)


def test_transformer_output_shape():
    model = TemporalTransformerForecaster(
        TemporalTransformerConfig(
            input_dim=63,
            d_model=32,
            nhead=4,
            num_layers=2,
            dim_feedforward=64,
            dropout=0.0,
            max_sequence_length=12,
        )
    )
    x = torch.randn(8, 12, 63)
    prediction = model(x)
    assert prediction.shape == (8, 63)
    assert model.encode(x).shape == (8, 32)


def test_transformer_mask_blocks_future_context():
    torch.manual_seed(7)
    model = TemporalTransformerForecaster(
        TemporalTransformerConfig(
            input_dim=3,
            d_model=12,
            nhead=3,
            num_layers=2,
            dim_feedforward=24,
            dropout=0.0,
            max_sequence_length=8,
        )
    )
    model.eval()

    original = torch.randn(1, 8, 3)
    changed = original.clone()
    changed[:, 5:, :] += 100.0

    with torch.inference_mode():
        first = model.encode_sequence(original)
        second = model.encode_sequence(changed)

    torch.testing.assert_close(
        first[:, :5, :],
        second[:, :5, :],
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_transformer_rejects_too_long_sequence():
    model = TemporalTransformerForecaster(
        TemporalTransformerConfig(
            input_dim=5,
            d_model=20,
            nhead=4,
            max_sequence_length=6,
        )
    )
    with pytest.raises(ValueError, match="sequence length"):
        model(torch.randn(2, 7, 5))
