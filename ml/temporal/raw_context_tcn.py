from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class RawContextTCNConfig:
    input_dim: int
    target_dim: int
    hidden_dim: int = 64
    kernel_size: int = 3
    dilations: tuple[int, ...] = (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
    )
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(
                "input_dim must be positive."
            )

        if self.target_dim <= 0:
            raise ValueError(
                "target_dim must be positive."
            )

        if self.hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if self.kernel_size <= 1:
            raise ValueError(
                "kernel_size must be greater "
                "than one."
            )

        if not self.dilations:
            raise ValueError(
                "dilations cannot be empty."
            )

        if any(
            dilation <= 0
            for dilation in self.dilations
        ):
            raise ValueError(
                "All dilations must be positive."
            )

        if not (
            0.0 <= self.dropout < 1.0
        ):
            raise ValueError(
                "dropout must be in [0, 1)."
            )


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()

        self.left_padding = (
            (kernel_size - 1)
            * dilation
        )

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv(
            F.pad(
                x,
                (
                    self.left_padding,
                    0,
                ),
            )
        )


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.conv_1 = CausalConv1d(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.conv_2 = CausalConv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(
            dropout
        )

        if (
            in_channels
            == hidden_channels
        ):
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv1d(
                in_channels,
                hidden_channels,
                kernel_size=1,
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        residual = self.residual(
            x
        )

        hidden = self.conv_1(
            x
        )
        hidden = self.activation(
            hidden
        )
        hidden = self.dropout(
            hidden
        )

        hidden = self.conv_2(
            hidden
        )
        hidden = self.activation(
            hidden
        )
        hidden = self.dropout(
            hidden
        )

        return hidden + residual


class RawContextTCNForecaster(
    nn.Module
):
    def __init__(
        self,
        config: RawContextTCNConfig,
    ) -> None:
        super().__init__()

        self.config = config

        blocks: list[nn.Module] = []
        channels = config.input_dim

        for dilation in config.dilations:
            blocks.append(
                ResidualBlock(
                    channels,
                    config.hidden_dim,
                    kernel_size=(
                        config.kernel_size
                    ),
                    dilation=dilation,
                    dropout=config.dropout,
                )
            )
            channels = (
                config.hidden_dim
            )

        self.encoder = nn.Sequential(
            *blocks
        )

        self.head = nn.Linear(
            config.hidden_dim,
            config.target_dim,
        )

    def encode_sequence(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_input(
            x
        )

        encoded = self.encoder(
            x.transpose(
                1,
                2,
            )
        )

        return encoded.transpose(
            1,
            2,
        )

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode_sequence(
            x
        )[
            :,
            -1,
            :,
        ]

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(
            self.encode(
                x
            )
        )

    def _validate_input(
        self,
        x: torch.Tensor,
    ) -> None:
        if x.ndim != 3:
            raise ValueError(
                "Raw TCN input must have shape "
                "(batch, sequence, features)."
            )

        if (
            x.shape[-1]
            != self.config.input_dim
        ):
            raise ValueError(
                "Raw TCN input feature dimension "
                "does not match config.input_dim."
            )
