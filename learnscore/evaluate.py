"""Evaluate a calibrated model on held-out test data."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EvalResult:
    coverage: float
    mean_volume_original_scale: float
    conformal_quantile: float


def evaluate_on_test(model, X_test: np.ndarray, r_test: np.ndarray, q_hat: float,
                      target_std: np.ndarray, device: str = "cpu") -> EvalResult:
    """coverage = P(score <= q_hat); volume from the model's own size hook, converted from
    standardized to original scale by the target-standardization Jacobian prod(target_std)."""
    model.eval()
    x_t = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    r_t = torch.as_tensor(r_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        scores = model.score(x_t, r_t).cpu().numpy().reshape(-1)
        coverage = float(np.mean(scores <= q_hat))
        if math.isinf(q_hat):
            mean_volume = float("inf")
        else:
            # stay in log space until the very end -- the volume itself can be far larger than
            # float32 can hold (e.g. ~1e53 on a d=16 dataset), while its log never overflows
            log_v_std = model.log_prediction_set_size(x_t, tau=q_hat).cpu().numpy().reshape(-1).astype(np.float64)
            log_jacobian = float(np.sum(np.log(target_std)))
            v_original = np.exp(log_v_std + log_jacobian)
            mean_volume = float(np.mean(v_original))
    return EvalResult(coverage=coverage, mean_volume_original_scale=mean_volume,
                       conformal_quantile=float(q_hat))
