"""MVCS baseline (Braun, Aolaritei, Jordan & Bach, arXiv:2503.19068) -- a different paper's
method, included here as a comparison, not one of the repo owner's own CLeS-family methods.

The set at covariate x is a p-norm ball reshaped by a covariate-dependent matrix:

    S(x, r) = || A(x) r ||_p,     ||v||_p = exp( (1/p) * logsumexp(p * log|v|) )

with a single GLOBAL learned p >= 1 (this port keeps the paper's convex ablation: p is
restricted to >= 1 via p = 1 + softplus(p_raw), rather than the paper's headline unrestricted
p < 1 regime, which gives non-convex sets and needs its own membership test -- a simplification
in exchange for reusing this repo's convex-set training/calibration/evaluation pipeline
unchanged). A(x) is an unconstrained d x d matrix from a small MLP, so the score is simply the
p-norm of a locally linear transform of the residual.

The induced set {r : ||A(x) r||_p <= tau} is A(x)^{-1} applied to a p-norm ball of radius tau,
so its volume is exact and closed-form:

    V(x; tau) = V_p(tau) / |det A(x)|,   V_p(tau) = (2 * Gamma(1/p + 1))^d / Gamma(d/p + 1) * tau^d

This repo's other three-stage joint training procedure (jointly fitting the mean f_theta
alongside A(x) and p) is also simplified away: here A(x)/p are trained the same way the CLeS
models are, against the shared frozen point predictor's residuals.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class MVCS(nn.Module):
    def __init__(self, input_dim: int, residual_dim: int = 1,
                 encoder_hidden_dims: list[int] | None = None, p_init: float = 2.0) -> None:
        super().__init__()
        encoder_hidden_dims = encoder_hidden_dims or [64, 32]
        self.residual_dim = residual_dim
        self.output_dim = residual_dim

        layers: list[nn.Module] = []
        prev = input_dim
        for h in encoder_hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.matrix_head = nn.Linear(prev, residual_dim * residual_dim)
        # identity at initialization: A(x) = I + small noise, so training starts from the
        # (unweighted) residual norm rather than an arbitrary random linear map
        nn.init.zeros_(self.matrix_head.weight)
        nn.init.zeros_(self.matrix_head.bias)

        p_raw_init = math.log(math.expm1(max(p_init - 1.0, 1e-3)))
        self.p_raw = nn.Parameter(torch.tensor(p_raw_init))

    @property
    def p(self) -> torch.Tensor:
        return 1.0 + F.softplus(self.p_raw)

    def _matrix(self, x: torch.Tensor) -> torch.Tensor:
        """A(x), shape (batch, d, d) = I + head(encoder(x))."""
        d = self.residual_dim
        eye = torch.eye(d, dtype=x.dtype, device=x.device)
        return eye + self.matrix_head(self.encoder(x)).view(-1, d, d)

    def score(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        A = self._matrix(x)
        v = torch.einsum("bij,bj->bi", A, r)
        p = self.p
        log_abs_v = torch.log(v.abs().clamp_min(1e-12))
        return torch.exp((1.0 / p) * torch.logsumexp(p * log_abs_v, dim=1, keepdim=True))

    def membership(self, x: torch.Tensor, r: torch.Tensor, tau: float) -> torch.Tensor:
        return (self.score(x, r) <= tau).reshape(-1)

    def log_prediction_set_size(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """log V(x;tau) = log V_p(tau) - log|det A(x)|, both terms exact (see module docstring)."""
        d = self.residual_dim
        p = self.p
        log_v_p_ball = (d * (math.log(2.0) + torch.lgamma(1.0 / p + 1.0))
                         - torch.lgamma(d / p + 1.0) + d * math.log(max(float(tau), 1e-300)))
        _, log_abs_det = torch.linalg.slogdet(self._matrix(x))  # (batch,)
        return (log_v_p_ball - log_abs_det).view(-1, 1)

    def prediction_set_size(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        return torch.exp(self.log_prediction_set_size(x, tau))
