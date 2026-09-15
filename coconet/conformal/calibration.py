"""Split-conformal calibration: the finite-sample order-statistic threshold."""

from __future__ import annotations

import math

import numpy as np


class CalibrationError(ValueError):
    pass


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """q_hat = S_(k) with k = ceil((n_cal + 1)(1 - alpha)), or +inf when k > n_cal."""
    if not (0.0 < alpha < 1.0):
        raise CalibrationError(f"alpha must be in (0, 1), got {alpha}")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise CalibrationError(f"scores must be 1-D, got shape {scores.shape}")
    n_cal = scores.shape[0]
    if n_cal == 0:
        raise CalibrationError("Cannot calibrate on zero calibration scores")
    if not np.all(np.isfinite(scores)):
        raise CalibrationError("Calibration scores contain non-finite values")

    k = math.ceil((n_cal + 1) * (1.0 - alpha))
    if k > n_cal:
        return float("inf")
    if k < 1:
        k = 1

    sorted_scores = np.sort(scores)
    return float(sorted_scores[k - 1])
