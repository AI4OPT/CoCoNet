"""Four-way seeded split (D_pred / D_score / D_cal / D_test) plus standardization.

One seeded permutation, sliced into four contiguous blocks by proportion. X and Y are then
standardized (subtract mean, divide by std) with the scaler fit on D_pred only and applied to
every split -- so downstream code can invert a D_pred-scale volume back to the original
response scale via the fitted Y std (the standardization Jacobian is just prod(y_std)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SplitError(ValueError):
    pass


@dataclass
class Scaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, A: np.ndarray) -> np.ndarray:
        return (A - self.mean) / self.std

    def inverse_transform(self, A: np.ndarray) -> np.ndarray:
        return A * self.std + self.mean


def _fit_scaler(A: np.ndarray, floor_zero_variance: bool = False) -> Scaler:
    mean = A.mean(axis=0)
    std = A.std(axis=0)
    near_zero = std < 1e-8
    if np.any(near_zero):
        if not floor_zero_variance:
            raise SplitError("a column has (near-)zero variance on D_pred; cannot standardize")
        # A handful of kept datasets (e.g. osales, p=401 >> n_pred~160) can have a sparse
        # feature column that happens to be constant within one seed's D_pred subset. Floor its
        # std to 1 rather than dying: the column is just centered, not scaled, which is inert
        # for a genuinely-constant feature and never applies to the response Y.
        std = np.where(near_zero, 1.0, std)
    return Scaler(mean=mean, std=std)


@dataclass
class SplitData:
    """Raw and standardized X/Y for each of the four splits, plus internal pred/score
    train/val index arrays (absolute row indices into X_pred / X_score) and the fitted
    scalers (fit on D_pred only)."""

    X_pred: np.ndarray
    Y_pred: np.ndarray
    X_score: np.ndarray
    Y_score: np.ndarray
    X_cal: np.ndarray
    Y_cal: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray

    X_pred_std: np.ndarray
    Y_pred_std: np.ndarray
    X_score_std: np.ndarray
    Y_score_std: np.ndarray
    X_cal_std: np.ndarray
    Y_cal_std: np.ndarray
    X_test_std: np.ndarray
    Y_test_std: np.ndarray

    pred_train_idx: np.ndarray
    pred_val_idx: np.ndarray
    score_train_idx: np.ndarray
    score_val_idx: np.ndarray

    x_scaler: Scaler
    y_scaler: Scaler


def make_split(
    X: np.ndarray,
    Y: np.ndarray,
    seed: int,
    pred_frac: float = 0.25,
    score_frac: float = 0.30,
    cal_frac: float = 0.25,
    test_frac: float = 0.20,
    pred_val_frac: float = 0.20,
    score_val_frac: float = 0.20,
) -> SplitData:
    n_total = len(X)
    fracs = [pred_frac, score_frac, cal_frac, test_frac]
    if not np.isclose(sum(fracs), 1.0, atol=1e-6):
        raise SplitError(f"split fractions must sum to 1.0, got {sum(fracs)} ({fracs})")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)

    n_pred = int(round(pred_frac * n_total))
    n_score = int(round(score_frac * n_total))
    n_cal = int(round(cal_frac * n_total))
    n_test = n_total - n_pred - n_score - n_cal  # remainder, so the split always partitions exactly
    if n_pred < 2 or n_score < 2 or n_cal < 1 or n_test < 1:
        raise SplitError(
            f"n_total={n_total} too small for fractions {fracs} "
            f"(n_pred={n_pred}, n_score={n_score}, n_cal={n_cal}, n_test={n_test})"
        )

    pred_idx = perm[:n_pred]
    score_idx = perm[n_pred : n_pred + n_score]
    cal_idx = perm[n_pred + n_score : n_pred + n_score + n_cal]
    test_idx = perm[n_pred + n_score + n_cal :]

    X_pred, Y_pred = X[pred_idx], Y[pred_idx]
    X_score, Y_score = X[score_idx], Y[score_idx]
    X_cal, Y_cal = X[cal_idx], Y[cal_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    x_scaler = _fit_scaler(X_pred, floor_zero_variance=True)
    y_scaler = _fit_scaler(Y_pred)

    # Internal train/val carve-outs, local to D_pred / D_score, for early stopping / checkpoint
    # selection. Indices returned are LOCAL positions into X_pred / X_score (0..n_pred-1 /
    # 0..n_score-1), a second independent permutation scoped to just that subset.
    pred_train_idx, pred_val_idx = _carve_val(n_pred, pred_val_frac, seed + 1)
    score_train_idx, score_val_idx = _carve_val(n_score, score_val_frac, seed + 2)

    return SplitData(
        X_pred=X_pred, Y_pred=Y_pred,
        X_score=X_score, Y_score=Y_score,
        X_cal=X_cal, Y_cal=Y_cal,
        X_test=X_test, Y_test=Y_test,
        X_pred_std=x_scaler.transform(X_pred), Y_pred_std=y_scaler.transform(Y_pred),
        X_score_std=x_scaler.transform(X_score), Y_score_std=y_scaler.transform(Y_score),
        X_cal_std=x_scaler.transform(X_cal), Y_cal_std=y_scaler.transform(Y_cal),
        X_test_std=x_scaler.transform(X_test), Y_test_std=y_scaler.transform(Y_test),
        pred_train_idx=pred_train_idx, pred_val_idx=pred_val_idx,
        score_train_idx=score_train_idx, score_val_idx=score_val_idx,
        x_scaler=x_scaler, y_scaler=y_scaler,
    )


def _carve_val(n: int, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 4:
        raise SplitError(f"cannot carve an internal validation split from only {n} rows")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = min(max(1, int(round(val_frac * n))), n - 1)
    return perm[n_val:], perm[:n_val]
