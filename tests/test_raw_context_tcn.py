from __future__ import annotations

import torch

from ml.temporal.raw_context_tcn import (
    RawContextTCNConfig,
    RawContextTCNForecaster,
)


def test_raw_context_tcn_shapes():
    model = RawContextTCNForecaster(
        RawContextTCNConfig(
            input_dim=15,
            target_dim=63,
            hidden_dim=16,
            dilations=(
                1,
                2,
                4,
            ),
            dropout=0.0,
        )
    )

    x = torch.randn(
        4,
        30,
        15,
    )

    prediction = model(
        x
    )

    assert prediction.shape == (
        4,
        63,
    )

    assert model.encode(
        x
    ).shape == (
        4,
        16,
    )


def test_raw_context_tcn_is_causal():
    torch.manual_seed(
        7
    )

    model = RawContextTCNForecaster(
        RawContextTCNConfig(
            input_dim=3,
            target_dim=5,
            hidden_dim=8,
            dilations=(
                1,
                2,
            ),
            dropout=0.0,
        )
    )

    model.eval()

    original = torch.randn(
        1,
        12,
        3,
    )

    changed = original.clone()
    changed[
        :,
        8:,
        :,
    ] += 100.0

    with torch.inference_mode():
        first = model.encode_sequence(
            original
        )

        second = model.encode_sequence(
            changed
        )

    torch.testing.assert_close(
        first[
            :,
            :8,
            :,
        ],
        second[
            :,
            :8,
            :,
        ],
    )
