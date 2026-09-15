"""Four-way data split D_pred / D_score / D_cal / D_test from one seeded permutation, plus the
internal train/validation carve-outs used inside a subset."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class SplitError(ValueError):
    pass


@dataclass
class FourWaySplit:
    pred_idx: np.ndarray
    score_idx: np.ndarray
    cal_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class PointPredictorSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray


def make_four_way_split(
    n_total: int,
    pred_frac: float,
    score_frac: float,
    cal_frac: float,
    test_frac: float,
    seed: int,
) -> FourWaySplit:
    """One permutation of range(n_total), sliced by proportion; D_test takes the remainder."""
    fracs = [pred_frac, score_frac, cal_frac, test_frac]
    total = sum(fracs)
    if not np.isclose(total, 1.0, atol=1e-6):
        raise SplitError(f"Split fractions must sum to 1.0, got {total} ({fracs})")
    if any(f <= 0 for f in fracs):
        raise SplitError(f"All split fractions must be positive, got {fracs}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)

    n_pred = int(round(pred_frac * n_total))
    n_score = int(round(score_frac * n_total))
    n_cal = int(round(cal_frac * n_total))
    n_pred_score_cal = n_pred + n_score + n_cal
    n_test = n_total - n_pred_score_cal

    if n_pred < 2 or n_score < 2 or n_cal < 1 or n_test < 1:
        raise SplitError(
            f"n_total={n_total} is too small for the requested split proportions "
            f"(got n_pred={n_pred}, n_score={n_score}, n_cal={n_cal}, n_test={n_test})"
        )

    pred_idx = perm[:n_pred]
    score_idx = perm[n_pred : n_pred + n_score]
    cal_idx = perm[n_pred + n_score : n_pred + n_score + n_cal]
    test_idx = perm[n_pred + n_score + n_cal :]

    split = FourWaySplit(pred_idx=pred_idx, score_idx=score_idx, cal_idx=cal_idx, test_idx=test_idx)
    _assert_disjoint_and_exhaustive(split, n_total)
    return split


def make_internal_validation_split(indices: np.ndarray, val_frac: float, seed: int) -> PointPredictorSplit:
    """Carve a train/val split out of a subset's absolute row indices."""
    n = len(indices)
    if n < 4:
        raise SplitError(f"Cannot carve an internal validation split from only {n} rows")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(val_frac * n)))
    n_val = min(n_val, n - 1)
    val_local = perm[:n_val]
    train_local = perm[n_val:]
    return PointPredictorSplit(train_idx=indices[train_local], val_idx=indices[val_local])


def _assert_disjoint_and_exhaustive(split: FourWaySplit, n_total: int) -> None:
    all_idx = np.concatenate([split.pred_idx, split.score_idx, split.cal_idx, split.test_idx])
    if len(all_idx) != n_total:
        raise SplitError(f"Split indices do not cover all {n_total} rows (got {len(all_idx)})")
    if len(np.unique(all_idx)) != n_total:
        raise SplitError("Split indices are not disjoint")


def slice_xy(X: pd.DataFrame, y: pd.Series, idx: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
