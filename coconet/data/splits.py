"""Seeded row splits."""

from dataclasses import dataclass

import numpy as np


@dataclass
class FourWaySplit:
    pred: np.ndarray
    score: np.ndarray
    cal: np.ndarray
    test: np.ndarray


def four_way_split(n, pred_frac, score_frac, cal_frac, seed):
    """One permutation of range(n) cut into D_pred / D_score / D_cal / D_test (the remainder)."""
    perm = np.random.default_rng(seed).permutation(n)
    n_pred = int(round(pred_frac * n))
    n_score = int(round(score_frac * n))
    n_cal = int(round(cal_frac * n))
    a, b, c = n_pred, n_pred + n_score, n_pred + n_score + n_cal
    if n_pred < 2 or n_score < 2 or n_cal < 1 or n - c < 1:
        raise ValueError(f"n={n} is too small for the split")
    return FourWaySplit(perm[:a], perm[a:b], perm[b:c], perm[c:])


def train_val_split(n, val_frac, seed):
    """(train_idx, val_idx) over range(n)."""
    perm = np.random.default_rng(seed).permutation(n)
    n_val = min(max(1, int(round(val_frac * n))), n - 1)
    return perm[n_val:], perm[:n_val]
