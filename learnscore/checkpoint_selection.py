"""Pick the training checkpoint by how it would actually be deployed, not by surrogate loss.

Every `checkpoint_every` epochs, run a real split-conformal calibration on half of the
validation set and measure real coverage/area on the other half. Keep the checkpoint with
smallest calibrated mean area among those whose coverage clears `(1 - alpha) - tolerance`;
if none clears it, keep the one with the smallest coverage gap. Selecting on the surrogate
loss instead tends to pick an early, undertrained checkpoint -- its "set" is just the small
epsilon-ball the score starts at, which minimizes the surrogate without having learned
anything.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch

from learnscore.calibrate import conformal_threshold


@dataclass
class Checkpoint:
    epoch: int
    coverage: float
    mean_area: float


@dataclass
class CheckpointTracker:
    alpha: float
    coverage_tolerance: float
    min_selection_epoch: int
    history: list[Checkpoint] = field(default_factory=list)
    best: Checkpoint | None = None
    best_state: dict | None = None

    @property
    def coverage_floor(self) -> float:
        return (1.0 - self.alpha) - self.coverage_tolerance

    def _rank(self, c: Checkpoint) -> tuple:
        admissible = c.coverage >= self.coverage_floor
        gap = abs(c.coverage - (1.0 - self.alpha))
        too_early = c.epoch < self.min_selection_epoch
        return (1 if too_early else 0, 0 if admissible else 1,
                c.mean_area if admissible else gap)

    def record(self, model, x_inner_cal, r_inner_cal, x_inner_eval, r_inner_eval, epoch: int) -> None:
        model.eval()
        with torch.no_grad():
            cal_scores = model.score(x_inner_cal, r_inner_cal).cpu().numpy().reshape(-1)
            q_tmp = conformal_threshold(cal_scores, self.alpha)
            eval_scores = model.score(x_inner_eval, r_inner_eval).cpu().numpy().reshape(-1)
            coverage = float(np.mean(eval_scores <= q_tmp))
            if np.isinf(q_tmp):
                mean_area = float("inf")
            else:
                areas = model.prediction_set_size(x_inner_eval, tau=q_tmp).cpu().numpy().reshape(-1)
                finite = areas[np.isfinite(areas)]
                mean_area = float(np.mean(finite)) if finite.size else float("inf")

        checkpoint = Checkpoint(epoch=epoch, coverage=coverage, mean_area=mean_area)
        self.history.append(checkpoint)
        if self.best is None or self._rank(checkpoint) < self._rank(self.best):
            self.best = checkpoint
            self.best_state = copy.deepcopy(model.state_dict())

    def restore_best(self, model) -> bool:
        if self.best_state is None:
            return False
        model.load_state_dict(self.best_state)
        return True
