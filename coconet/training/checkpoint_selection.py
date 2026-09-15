"""Checkpoint selection on calibrated validation coverage and set size.

Every `checkpoint_every` epochs the model is calibrated by split conformal on one half of the
validation split and its coverage and mean set size are measured on the other half. The kept
checkpoint is the one with the smallest mean size among those whose coverage clears
(1 - alpha) - coverage_tolerance; if none does, the one with the smallest coverage gap.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np
import torch

from coconet.conformal.calibration import conformal_threshold
from coconet.training.deterministic_size import deterministic_size_fn

CHECKPOINT_SELECTION_METRICS = ("validation_total_loss", "calibrated_validation_area")


class CheckpointSelectionError(ValueError):
    pass


@dataclass
class CalibratedCheckpoint:
    epoch: int
    validation_surrogate_loss: float
    validation_surrogate_area: float
    validation_smooth_coverage: float
    temporary_calibrated_quantile: float
    actual_validation_coverage: float
    actual_validation_mean_area: float
    actual_validation_median_area: float
    actual_validation_area_std: float
    actual_validation_area_q90: float
    actual_validation_mean_log_area: float = float("nan")
    actual_validation_mean_root_area: float = float("nan")
    actual_validation_area_q10: float = float("nan")
    n_area_nonfinite: int = 0
    n_area_nonpositive: int = 0

    def as_row(self) -> dict[str, float]:
        return {
            "epoch": self.epoch,
            "validation_surrogate_loss": self.validation_surrogate_loss,
            "validation_surrogate_area": self.validation_surrogate_area,
            "validation_smooth_coverage": self.validation_smooth_coverage,
            "temporary_calibrated_quantile": self.temporary_calibrated_quantile,
            "actual_validation_coverage": self.actual_validation_coverage,
            "actual_validation_mean_area": self.actual_validation_mean_area,
            "actual_validation_median_area": self.actual_validation_median_area,
            "actual_validation_area_std": self.actual_validation_area_std,
            "actual_validation_area_q90": self.actual_validation_area_q90,
            "actual_validation_area_q10": self.actual_validation_area_q10,
            "actual_validation_mean_log_area": self.actual_validation_mean_log_area,
            "actual_validation_mean_root_area": self.actual_validation_mean_root_area,
            "n_area_nonfinite": self.n_area_nonfinite,
            "n_area_nonpositive": self.n_area_nonpositive,
        }


def evaluate_calibrated_checkpoint(
    model: torch.nn.Module,
    x_inner_cal: torch.Tensor, r_inner_cal: torch.Tensor,
    x_inner_eval: torch.Tensor, r_inner_eval: torch.Tensor,
    alpha: float, epoch: int,
    surrogate_loss: float = float("nan"),
    surrogate_area: float = float("nan"),
    smooth_coverage: float = float("nan"),
) -> CalibratedCheckpoint:
    """Calibrate on the inner-calibration half, measure coverage and size on the other half."""
    was_training = model.training
    model.eval()
    size_fn = deterministic_size_fn(model)
    with torch.no_grad():
        cal_scores = model.score(x_inner_cal, r_inner_cal).cpu().numpy().reshape(-1)
        q_tmp = conformal_threshold(cal_scores, alpha)
        eval_scores = model.score(x_inner_eval, r_inner_eval).cpu().numpy().reshape(-1)
        coverage = float(np.mean(eval_scores <= q_tmp))
        if math.isinf(q_tmp):
            areas = np.array([float("inf")])
        else:
            areas = size_fn(x_inner_eval, tau=q_tmp).cpu().numpy().reshape(-1)
    if was_training:
        model.train()
    finite = areas[np.isfinite(areas)]
    if finite.size == 0:
        finite = np.array([float("inf")])
    n_nonfinite = int(areas.size - finite.size)
    positive = finite[finite > 0.0]
    n_nonpositive = int(finite.size - positive.size)
    mean_log_area = float(np.mean(np.log(positive))) if positive.size else float("nan")
    _d = int(getattr(model, "residual_dim", getattr(model, "output_dim", 1)) or 1)
    mean_root_area = (float(np.mean(positive ** (1.0 / _d))) if positive.size else float("nan"))
    return CalibratedCheckpoint(
        epoch=epoch,
        validation_surrogate_loss=float(surrogate_loss),
        validation_surrogate_area=float(surrogate_area),
        validation_smooth_coverage=float(smooth_coverage),
        temporary_calibrated_quantile=float(q_tmp),
        actual_validation_coverage=coverage,
        actual_validation_mean_area=float(np.mean(finite)),
        actual_validation_median_area=float(np.median(finite)),
        actual_validation_area_std=float(np.std(finite)),
        actual_validation_area_q90=float(np.quantile(finite, 0.90)),
        actual_validation_area_q10=float(np.quantile(finite, 0.10)),
        actual_validation_mean_log_area=mean_log_area,
        actual_validation_mean_root_area=mean_root_area,
        n_area_nonfinite=n_nonfinite,
        n_area_nonpositive=n_nonpositive,
    )


@dataclass
class CalibratedCheckpointTracker:
    """Keeps the best checkpoint by calibrated validation size subject to a coverage floor.
    Checkpoints before `min_selection_epoch` rank after every later one."""

    alpha: float
    coverage_tolerance: float
    history: list[CalibratedCheckpoint] = field(default_factory=list)
    best: CalibratedCheckpoint | None = None
    best_state: dict | None = None
    selection_rule: str = "no_checkpoint_recorded"
    min_selection_epoch: int = 0

    @property
    def coverage_floor(self) -> float:
        return (1.0 - self.alpha) - self.coverage_tolerance

    def _rank(self, c: CalibratedCheckpoint) -> tuple:
        admissible = c.actual_validation_coverage >= self.coverage_floor
        gap = abs(c.actual_validation_coverage - (1.0 - self.alpha))
        too_early = c.epoch < self.min_selection_epoch
        return (1 if too_early else 0,
                0 if admissible else 1,
                c.actual_validation_mean_area if admissible else gap,
                gap if admissible else c.actual_validation_mean_area)

    def record(self, checkpoint: CalibratedCheckpoint, model: torch.nn.Module) -> None:
        self.history.append(checkpoint)
        if self.best is None or self._rank(checkpoint) < self._rank(self.best):
            self.best = checkpoint
            self.best_state = copy.deepcopy(model.state_dict())
        self.selection_rule = (
            ("calibrated_area_coverage_compliant"
             + ("" if self.best is None or self.best.epoch >= self.min_selection_epoch
                else "_below_min_epoch_fallback"))
            if self.best is not None and self.best.actual_validation_coverage >= self.coverage_floor
            else "calibrated_area_fallback_min_coverage_gap"
        )

    def restore_best(self, model: torch.nn.Module) -> bool:
        if self.best_state is None:
            return False
        model.load_state_dict(self.best_state)
        return True

    def rows(self) -> list[dict[str, float]]:
        return [c.as_row() for c in self.history]


def validate_selection_metric(metric: str) -> str:
    if metric not in CHECKPOINT_SELECTION_METRICS:
        raise CheckpointSelectionError(
            f"checkpoint_selection_metric must be one of {list(CHECKPOINT_SELECTION_METRICS)}, got {metric!r}")
    return metric
