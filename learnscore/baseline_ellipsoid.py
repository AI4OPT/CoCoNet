"""Global ellipsoid conformity score (Mahalanobis distance).

One covariance Sigma is fit once from D_score residuals (a ridge is added for numerical
stability, matching the original repo's 1e-6):

    score(x, r) = r^T Sigma^-1 r

independent of x -- only the LOCAL (kNN-covariance) variant from the original repo depends on
x, and that variant is not implemented here (task scope: global ellipsoid only).

The induced set {r : r^T Sigma^-1 r <= tau} is the ellipsoid r = sqrt(tau) * Sigma^(1/2) u,
|u| <= 1, so its volume is the unit ball's volume scaled by the linear map's determinant:

    V = vol_d(unit ball) * tau^(d/2) * sqrt(det Sigma)

computed in log space (via slogdet) so it doesn't overflow at high d, matching the ICNN's own
volume estimator.
"""

from __future__ import annotations

import math

import numpy as np
import torch


class GlobalEllipsoid:
    def __init__(self, residual_dim: int, ridge: float = 1e-6) -> None:
        self.residual_dim = residual_dim
        self.ridge = ridge
        self.sigma_inv: torch.Tensor | None = None
        self.log_det_sigma: float | None = None

    def fit(self, r_score: np.ndarray) -> "GlobalEllipsoid":
        """r_score: (n, residual_dim) proper-training residuals only."""
        r = torch.as_tensor(r_score, dtype=torch.float64)
        sigma = torch.cov(r.T).reshape(self.residual_dim, self.residual_dim)
        sigma = sigma + self.ridge * torch.eye(self.residual_dim, dtype=torch.float64)
        sign, log_det = torch.linalg.slogdet(sigma)
        if sign.item() <= 0:
            raise ValueError("fitted covariance is not positive definite even after ridging")
        self.log_det_sigma = float(log_det)
        self.sigma_inv = torch.linalg.inv(sigma).to(torch.float32)
        return self

    def score(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if self.sigma_inv is None:
            raise RuntimeError("GlobalEllipsoid.fit must be called before scoring")
        sigma_inv = self.sigma_inv.to(dtype=r.dtype, device=r.device)
        quad = torch.einsum("bi,ij,bj->b", r, sigma_inv, r)
        return quad.clamp_min(0.0).unsqueeze(-1)

    def membership(self, x: torch.Tensor, r: torch.Tensor, tau: float) -> torch.Tensor:
        return (self.score(x, r) <= tau).reshape(-1)

    def log_prediction_set_size(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        if self.log_det_sigma is None:
            raise RuntimeError("GlobalEllipsoid.fit must be called before computing volume")
        d = self.residual_dim
        log_unit_ball = (d / 2.0) * math.log(math.pi) - math.lgamma(d / 2.0 + 1.0)
        log_v = log_unit_ball + (d / 2.0) * math.log(max(float(tau), 1e-300)) + 0.5 * self.log_det_sigma
        return torch.full((x.shape[0], 1), log_v, dtype=x.dtype, device=x.device)

    def prediction_set_size(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        return torch.exp(self.log_prediction_set_size(x, tau))
