"""Checkpoint selection on calibrated validation coverage and size.

Every few epochs the model is calibrated by split conformal on one half of the validation split
and its coverage and mean set size are measured on the other half. The kept checkpoint is the
smallest mean size among those whose coverage is at least (1 - alpha) - tolerance; if none
qualifies, the smallest coverage gap. Checkpoints before `min_epoch` rank after all later ones.
"""

import copy
import math

import numpy as np
import torch

from coconet.conformal.calibration import conformal_threshold


def calibrated_coverage_and_size(model, x_cal, r_cal, x_eval, r_eval, alpha):
    model.eval()
    with torch.no_grad():
        q = conformal_threshold(model.score(x_cal, r_cal).cpu().numpy().reshape(-1), alpha)
        eval_scores = model.score(x_eval, r_eval).cpu().numpy().reshape(-1)
        coverage = float(np.mean(eval_scores <= q))
        if math.isinf(q):
            sizes = np.array([float("inf")])
        else:
            sizes = model.size(x_eval, q).cpu().numpy().reshape(-1)
    finite = sizes[np.isfinite(sizes)]
    if finite.size == 0:
        finite = np.array([float("inf")])
    return coverage, float(np.mean(finite))


class CheckpointTracker:
    def __init__(self, alpha, tolerance, min_epoch):
        self.target = 1.0 - alpha
        self.floor = 1.0 - alpha - tolerance
        self.min_epoch = min_epoch
        self.best = None            # (rank, epoch, coverage, size)
        self.best_state = None

    def _rank(self, epoch, coverage, size):
        admissible = coverage >= self.floor
        gap = abs(coverage - self.target)
        return (1 if epoch < self.min_epoch else 0,
                0 if admissible else 1,
                size if admissible else gap,
                gap if admissible else size)

    def record(self, epoch, coverage, size, model):
        rank = self._rank(epoch, coverage, size)
        if self.best is None or rank < self.best[0]:
            self.best = (rank, epoch, coverage, size)
            self.best_state = copy.deepcopy(model.state_dict())

    @property
    def best_epoch(self):
        return self.best[1]

    @property
    def rule(self):
        return "coverage_admissible_min_size" if self.best[2] >= self.floor else "min_coverage_gap"

    def restore(self, model):
        model.load_state_dict(self.best_state)
