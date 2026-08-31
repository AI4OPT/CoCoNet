"""Empirical-copula conformity score (Messoudi, Destercke & Rousseau, arXiv:2101.12002),
simplified.

Each residual coordinate is passed through its own empirical CDF, fit on D_score:

    U_j = Fhat_j(|r_j|),   Fhat_j(a) = (1/n) * |{ |r_score,j| <= a }|   (max-rank convention)

and the joint score is the max over coordinates of these per-coordinate ranks:

    score(x, r) = max_j Fhat_j(|r_j|)

independent of x. Two simplifications vs. the original repo, noted explicitly:
  - the original learns a small MLP normalizer sigma_j(x) = exp(mu_j(x)) + beta before ranking;
    here each coordinate is ranked directly against its own D_score empirical distribution
    (a per-coordinate empirical-quantile normalization), with no learned, x-dependent scale.
  - because Fhat_j is a monotone step function, its inverse is a plain order-statistic lookup,
    so the induced sublevel set {r : max_j Fhat_j(|r_j|) <= tau} = box with half-widths
    Fhat_j^-1(tau) has an EXACT closed form -- no need for volume.py's generic root-finder.
"""

from __future__ import annotations

import math

import numpy as np
import torch


class EmpiricalCopula:
    def __init__(self, residual_dim: int = 1) -> None:
        self.residual_dim = residual_dim
        self.sorted_abs_r: torch.Tensor | None = None  # (residual_dim, n_score), each row sorted

    def fit(self, r_score: np.ndarray) -> "EmpiricalCopula":
        """r_score: (n, residual_dim) proper-training residuals only."""
        r = torch.as_tensor(r_score, dtype=torch.float64).abs()
        self.sorted_abs_r, _ = torch.sort(r, dim=0)
        self.sorted_abs_r = self.sorted_abs_r.T.contiguous()  # (residual_dim, n)
        return self

    def _ecdf(self, abs_r: torch.Tensor) -> torch.Tensor:
        """abs_r: (batch, residual_dim) -> (batch, residual_dim) of Fhat_j(|r_j|), max-rank."""
        if self.sorted_abs_r is None:
            raise RuntimeError("EmpiricalCopula.fit must be called before scoring")
        n = self.sorted_abs_r.shape[1]
        out = torch.empty_like(abs_r, dtype=torch.float64)
        for j in range(self.residual_dim):
            counts = torch.searchsorted(self.sorted_abs_r[j], abs_r[:, j].to(torch.float64), right=True)
            out[:, j] = counts.to(torch.float64) / n
        return out

    def score(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        u = self._ecdf(r.abs())
        return u.max(dim=-1, keepdim=True).values.to(r.dtype)

    def membership(self, x: torch.Tensor, r: torch.Tensor, tau: float) -> torch.Tensor:
        return (self.score(x, r) <= tau).reshape(-1)

    def _quantiles(self, tau: float) -> torch.Tensor:
        """Fhat_j^-1(tau) per coordinate: order-statistic index ceil(n*tau) - 1, no interpolation."""
        n = self.sorted_abs_r.shape[1]
        index = max(0, min(math.ceil(n * float(tau)) - 1, n - 1))
        return self.sorted_abs_r[:, index]  # (residual_dim,)

    def log_prediction_set_size(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        q = self._quantiles(tau)
        log_v = torch.log((2.0 * q).clamp_min(1e-300)).sum().item()
        return torch.full((x.shape[0], 1), log_v, dtype=x.dtype, device=x.device)

    def prediction_set_size(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        return torch.exp(self.log_prediction_set_size(x, tau))
