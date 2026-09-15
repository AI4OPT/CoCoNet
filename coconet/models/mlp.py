"""Feed-forward regressor used as the point predictor F(x) and inside the baselines."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """input -> hidden layers (ReLU) -> linear output head."""

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None, output_dim: int = 1) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
