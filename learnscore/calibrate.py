"""Split-conformal calibration: the finite-sample order-statistic threshold.

Uses the classical ceil((n+1)(1-alpha))-th order statistic, not an interpolated quantile --
that's what the split-conformal coverage guarantee is stated in terms of.
"""

from __future__ import annotations

import math

import numpy as np


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """q_hat = S_(k) where k = ceil((n+1)(1-alpha)), or +inf if k exceeds n."""
    scores = np.asarray(scores, dtype=float)
    n = scores.shape[0]
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[max(k, 1) - 1])
