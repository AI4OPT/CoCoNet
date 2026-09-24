import math

import numpy as np


def conformal_threshold(scores, alpha):
    """Split-conformal threshold: the k-th smallest calibration score with
    k = ceil((n + 1)(1 - alpha)), or +inf when k > n."""
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if not np.all(np.isfinite(scores)):
        raise ValueError("calibration scores contain non-finite values")
    n = scores.shape[0]
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[max(k, 1) - 1])
