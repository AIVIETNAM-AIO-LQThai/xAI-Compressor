from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TemporalTransformerConfig:
    input_dim: int
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 256
    dropout: float = 0.10
    max_sequence_length: int = 64

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.nhead <= 0:
            raise ValueError("nhead must be positive.")
        if self.d_model % self.nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive.")


class TemporalTransformerForecaster(nn.Module):
    def __init__(self, config: TemporalTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_sequence_length, config.d_model)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.head = nn.Linear(config.d_model, config.input_dim)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                "Transformer input must have shape "
                "(batch, sequence, features)."
            )
        if x.shape[-1] != self.config.input_dim:
            raise ValueError(
                "Transformer input feature dimension "
                "does not match config.input_dim."
            )
        if x.shape[1] > self.config.max_sequence_length:
            raise ValueError(
                "Transformer sequence length exceeds max_sequence_length."
            )

    def encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        sequence_length = int(x.shape[1])
        hidden = self.input_projection(x)
        hidden = hidden + self.position_embedding[:, :sequence_length, :]

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )
        return self.encoder(hidden, mask=causal_mask)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_sequence(x)[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))
