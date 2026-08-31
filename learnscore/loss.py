"""The tau1 prediction-set loss.

    L = L_size + lambda_miss * L_miss + lambda_cov * L_cov

Three terms, all evaluated at one threshold tau shared by every term in a given step:

  tau      fixed at `tau_fixed` (paper: 1.0), or the batch's empirical (1-alpha) score
           quantile, detached -- set by `tau_mode` in the config.
  L_size   log(1 + E[V]) or (E[V])^(1/d), from the model's own volume/width/area hook.
  L_miss   scale-invariant hinge on points outside the tau-set:
               relu((score - tau) / (|tau| + eps))
  L_cov    (smooth_coverage - (1 - alpha))^2, smooth_coverage = mean(sigmoid((tau - score)/T))
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass
class LossConfig:
    alpha: float
    tau_mode: str = "fixed"          # "fixed" or "quantile"
    tau_fixed: float = 1.0
    size_objective: str = "root_mean_volume"  # or "log1p_mean_volume"
    kappa_coverage: float = 10.0
    lambda_miss: float = 1.0
    lambda_cov: float = 1.0
    miss_eps: float = 1e-8


def resolve_tau(model, x: torch.Tensor, r: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    if cfg.tau_mode == "fixed":
        return torch.as_tensor(cfg.tau_fixed, dtype=x.dtype, device=x.device)
    if cfg.tau_mode == "quantile":
        scores = model.score(x, r).detach().reshape(-1)
        return torch.quantile(scores, 1.0 - cfg.alpha).detach()
    raise ValueError(f"tau_mode must be 'fixed' or 'quantile', got {cfg.tau_mode!r}")


def size_term(model, x: torch.Tensor, tau: torch.Tensor, size_objective: str) -> torch.Tensor:
    log_v = model.log_prediction_set_size(x, tau=float(tau)).reshape(-1)
    log_mean = torch.logsumexp(log_v, dim=0) - math.log(log_v.shape[0])
    if size_objective == "log1p_mean_volume":
        return F.softplus(log_mean)
    if size_objective == "root_mean_volume":
        return torch.exp(log_mean / model.residual_dim)
    raise ValueError(f"unknown size_objective {size_objective!r}")


def miss_term(scores: torch.Tensor, tau: torch.Tensor, eps: float) -> torch.Tensor:
    """Scale-invariant hinge: under score -> c*score the (quantile) tau scales the same way,
    so this ratio is unchanged -- unlike a fixed-kappa softplus margin, whose scale depends on
    the score's own units."""
    excess = scores - tau
    return (excess / (tau.detach().abs() + eps)).clamp_min(0.0).mean()


def coverage_term(scores: torch.Tensor, tau: torch.Tensor, alpha: float, kappa: float) -> torch.Tensor:
    smooth_coverage = torch.sigmoid((tau - scores) * kappa).mean()
    return (smooth_coverage - (1.0 - alpha)) ** 2


def prediction_set_loss(model, x: torch.Tensor, r: torch.Tensor, cfg: LossConfig) -> torch.Tensor:
    tau = resolve_tau(model, x, r, cfg)
    scores = model.score(x, r).reshape(-1)
    return (size_term(model, x, tau, cfg.size_objective)
            + cfg.lambda_miss * miss_term(scores, tau, cfg.miss_eps)
            + cfg.lambda_cov * coverage_term(scores, tau, cfg.alpha, cfg.kappa_coverage))
