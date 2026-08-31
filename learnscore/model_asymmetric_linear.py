"""Asymmetric linear conformity score (one of the CLeS-family methods, alongside the ICNN
and the spline score).

    score(x, r) = a_plus(x) * r        if r >= 0
                = -a_minus(x) * r      if r < 0

with a_plus(x), a_minus(x) = softplus(head(x)) + epsilon > 0, learned per-coordinate. For a
multivariate residual (output_dim > 1) the joint score is the max over coordinates,

    S(x, r) = max_j [ a_plus_j(x) * relu(r_j) + a_minus_j(x) * relu(-r_j) ]

whose sublevel set {r : S(x, r) <= tau} is the axis-aligned asymmetric hyperrectangle
prod_j [-tau/a_minus_j(x), tau/a_plus_j(x)], with EXACT closed-form volume -- no root-finding
needed, unlike the ICNN. At output_dim=2 the residual is first rotated by a covariate-dependent
angle before the hyperrectangle score is applied, so the learned rectangle need not be
axis-aligned in the original coordinates; at higher output_dim rotation is skipped (a
simplification -- the general SO(d) rotation is not worth the extra machinery here).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class AsymmetricLinear(nn.Module):
    def __init__(self, input_dim: int, residual_dim: int = 1,
                 encoder_hidden_dims: list[int] | None = None, epsilon: float = 1e-3) -> None:
        super().__init__()
        encoder_hidden_dims = encoder_hidden_dims or [64, 32]
        self.residual_dim = residual_dim
        self.output_dim = residual_dim
        self.epsilon = epsilon
        self.rotate = residual_dim == 2

        layers: list[nn.Module] = []
        prev = input_dim
        for h in encoder_hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*layers)

        self.head_plus = nn.Linear(prev, residual_dim)
        self.head_minus = nn.Linear(prev, residual_dim)
        if self.rotate:
            self.angle_head = nn.Linear(prev, 1)
            nn.init.zeros_(self.angle_head.weight)
            nn.init.zeros_(self.angle_head.bias)

    def slopes(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        a_plus = F.softplus(self.head_plus(h)) + self.epsilon
        a_minus = F.softplus(self.head_minus(h)) + self.epsilon
        return a_plus, a_minus

    def _rotate_in(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """r -> Q(x)^T r, Q(x) a covariate-dependent 2x2 rotation. Zero-initialized angle head,
        so training starts from the axis-aligned score exactly."""
        if not self.rotate:
            return r
        theta = self.angle_head(self.encoder(x)).squeeze(-1)
        cos, sin = torch.cos(theta), torch.sin(theta)
        r1 = cos * r[:, 0] + sin * r[:, 1]
        r2 = -sin * r[:, 0] + cos * r[:, 1]
        return torch.stack([r1, r2], dim=1)

    def score(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        z = self._rotate_in(x, r)
        a_plus, a_minus = self.slopes(x)
        coordinate_scores = torch.where(z >= 0, a_plus * z, -a_minus * z)
        return coordinate_scores.max(dim=-1, keepdim=True).values

    def membership(self, x: torch.Tensor, r: torch.Tensor, tau: float) -> torch.Tensor:
        return (self.score(x, r) <= tau).reshape(-1)

    def log_prediction_set_size(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Exact log volume of the hyperrectangle: log V = d*log(tau) + sum_j log(1/a+_j + 1/a-_j).
        Rotation doesn't change this -- |det Q(x)| = 1, so volume is unaffected by orientation."""
        a_plus, a_minus = self.slopes(x)
        side = 1.0 / a_plus + 1.0 / a_minus
        log_v = torch.log(side.clamp_min(1e-300)).sum(dim=1, keepdim=True)
        return log_v + self.residual_dim * math.log(max(float(tau), 1e-300))

    def prediction_set_size(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        return torch.exp(self.log_prediction_set_size(x, tau))

    def width(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        if self.residual_dim != 1:
            raise ValueError("width() is only defined for residual_dim=1")
        return self.prediction_set_size(x, tau)

    def interval(self, x: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.residual_dim != 1:
            raise ValueError("interval() is only defined for residual_dim=1")
        a_plus, a_minus = self.slopes(x)
        return -tau / a_minus, tau / a_plus
