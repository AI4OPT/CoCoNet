"""Absolute-residual conformity score -- the simplest baseline, no training at all.

    score(x, r) = max_j |r_j|

the infinity-norm of the residual (at residual_dim=1 this is just |r|). Independent of x:
same interface as the trained scores (`score(x, r)`, `log_prediction_set_size(x, tau)`) so it
plugs into the same calibrate/evaluate code, but ignores x entirely.

The induced set {r : max_j |r_j| <= tau} is the axis-aligned hypercube [-tau, tau]^d, with
EXACT volume (2*tau)^d -- no root-finding needed.
"""

from __future__ import annotations

import math

import torch


class AbsoluteResidual:
    def __init__(self, residual_dim: int = 1) -> None:
        self.residual_dim = residual_dim

    def score(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return r.abs().max(dim=-1, keepdim=True).values

    def membership(self, x: torch.Tensor, r: torch.Tensor, tau: float) -> torch.Tensor:
        return (self.score(x, r) <= tau).reshape(-1)

    def log_prediction_set_size(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        log_side = math.log(max(2.0 * float(tau), 1e-300))
        return torch.full((x.shape[0], 1), self.residual_dim * log_side, dtype=x.dtype, device=x.device)

    def prediction_set_size(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        return torch.exp(self.log_prediction_set_size(x, tau))

    def width(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        if self.residual_dim != 1:
            raise ValueError("width() is only defined for residual_dim=1")
        return self.prediction_set_size(x, tau)

    def interval(self, x: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.residual_dim != 1:
            raise ValueError("interval() is only defined for residual_dim=1")
        offset = torch.full((x.shape[0], 1), float(tau), dtype=x.dtype, device=x.device)
        return -offset, offset
