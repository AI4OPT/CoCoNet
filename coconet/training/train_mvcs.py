"""Training of the fixed-centre convex MVCS model (coconet/conformal/mvcs.py).

One stage of Adam (separate learning rates for the matrix net and p) with a cosine schedule and
gradient clipping on the MVCS objective; p is projected onto p >= 1 after every step. The kept
checkpoint is the epoch with the smallest finite validation objective (the same objective,
evaluated in one pass over the whole validation split), and early stopping watches that value.
The conformal threshold is computed afterwards by the caller on the untouched calibration split.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from coconet.conformal.mvcs import PAPER_LITERAL, MvcsScore, MvcsSingleNormModel
from coconet.utils import get_logger

logger = get_logger(__name__)

VALIDATION_MVCS_OBJECTIVE = "validation_mvcs_objective"


@dataclass
class MvcsTrainConfig:
    output_dim: int = 2
    alpha: float = 0.10
    hidden_dim: int = 256
    n_layers: int = 3
    ridge: float = 1e-6
    p_init: float = 2.0
    p_min: float = 0.1
    order_statistic_convention: str = PAPER_LITERAL
    average_volume_term: bool = False
    matrix_learning_rate: float = 1e-2
    p_learning_rate: float = 1e-2
    max_epochs: int = 400
    batch_size: int = 100
    patience: int = 60
    grad_clip_norm: float = 5.0
    checkpoint_every: int = 1
    inner_calibration_frac: float = 0.5
    nonfinite_batch_tolerance: float = 0.05
    seed: int = 0
    device: str = "cpu"


@dataclass
class MvcsTrainResult:
    score: MvcsScore
    selected_epoch: int
    selection_rule: str
    validation_coverage: float
    validation_mean_volume: float
    validation_conformal_threshold: float
    training_history: list[dict] = field(default_factory=list)
    checkpoint_history: list[dict] = field(default_factory=list)
    learned_p_history: list[dict] = field(default_factory=list)
    stage_summaries: list[dict] = field(default_factory=list)
    n_batches: int = 0
    n_nonfinite_batches: int = 0
    runtime_seconds: float = 0.0
    validation_mvcs_objective: float = float("nan")
    checkpoint_selection: str = VALIDATION_MVCS_OBJECTIVE


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _grad_norm(params) -> float:
    live = [p.grad.detach().flatten() for p in params if p is not None and p.grad is not None]
    return float(torch.cat(live).norm()) if live else 0.0


def _as_2d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(len(x), -1)


class _ObjectiveCheckpointSelector:
    """Keeps the state with the smallest finite validation MVCS objective."""

    def __init__(self, alpha: float, convention: str, average_volume_term: bool) -> None:
        self.alpha = alpha
        self.convention = convention
        self.average_volume_term = average_volume_term
        self.history: list[dict] = []
        self._best: tuple[float, int, Any] | None = None      # (objective, epoch, state)

    def consider(self, epoch: int, model: MvcsSingleNormModel, score: MvcsScore,
                 x_inner_cal, t_inner_cal, x_inner_eval, t_inner_eval,
                 center=None, x_full=None, t_full=None) -> dict:
        xv = x_inner_eval if x_full is None else x_full
        tv = t_inner_eval if t_full is None else t_full
        device = score.device
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                xt = torch.as_tensor(np.asarray(xv, dtype=float).reshape(len(xv), -1),
                                     dtype=torch.float32, device=device)
                tt = torch.as_tensor(np.asarray(tv, dtype=float), dtype=torch.float32,
                                     device=device)
                residual = tt
                parts = model.loss(xt, residual, self.alpha, self.convention,
                                   self.average_volume_term)
                objective = float(parts.total_loss)
        except Exception as exc:
            logger.warning("mvcs: validation objective failed at epoch %d (%s)", epoch, exc)
            objective = float("nan")
        finally:
            if was_training:
                model.train()

        row = {
            "epoch": epoch, "validation_mvcs_objective": objective, "learned_p": score.p,
            "validation_coverage": float("nan"), "validation_mean_volume": float("nan"),
            "validation_conformal_threshold": float("nan"),
            "selection_metric": objective, "eligible": bool(np.isfinite(objective)),
        }
        self.history.append(row)
        if not np.isfinite(objective):
            return row
        if self._best is None or objective < self._best[0]:
            self._best = (objective, epoch,
                          {k: v.detach().clone() for k, v in model.state_dict().items()})
        return row

    def best(self) -> tuple[Any, int, str]:
        if self._best is None:
            raise RuntimeError(
                "MVCS checkpoint selection found no checkpoint with a finite validation "
                "objective. Every evaluated epoch produced NaN/Inf, so there is nothing to "
                "select; refusing to return an arbitrary state. Lower the learning rates, "
                "raise `ridge`, or inspect the training history for diverging batches."
            )
        _, epoch, state = self._best
        return state, epoch, VALIDATION_MVCS_OBJECTIVE

    def summary_for(self, epoch: int) -> dict:
        for row in self.history:
            if row["epoch"] == epoch:
                return row
        return {"validation_coverage": float("nan"), "validation_mean_volume": float("nan"),
                "validation_conformal_threshold": float("nan"),
                "validation_mvcs_objective": float("nan")}


def _optimizer_groups(model: MvcsSingleNormModel, cfg: MvcsTrainConfig, include_center: bool):
    groups = [{"params": list(model.matrix_net.parameters()), "lr": cfg.matrix_learning_rate,
               "name": "matrix"}]
    groups.append({"params": [model.p_raw], "lr": cfg.p_learning_rate, "name": "p"})
    return groups


def _run_mvcs_epochs(
    model: MvcsSingleNormModel, cfg: MvcsTrainConfig, device: torch.device,
    x_fit_t: torch.Tensor, y_fit_t: torch.Tensor, center_fit_t: torch.Tensor | None,
    inner: dict, n_epochs: int, stage: str, include_center: bool,
    selector, history: list[dict], p_history: list[dict],
    epoch_offset: int = 0, checkpoint_center=None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> tuple[int, int, int]:
    """One stage of MVCS training. Returns (n_batches, n_nonfinite_batches, epochs_run)."""
    opt = optimizer if optimizer is not None else torch.optim.Adam(
        _optimizer_groups(model, cfg, include_center))
    sched = scheduler if scheduler is not None else \
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_epochs, 1))
    n = int(x_fit_t.shape[0])
    min_batch = max(4, int(math.ceil(1.0 / max(cfg.alpha, 1e-6))))
    n_batches = n_nonfinite = 0
    epochs_run = 0
    score_view = MvcsScore(model=model, device=device)
    epochs_since_improvement, best_metric_seen = 0, float("inf")

    for local_epoch in range(n_epochs):
        epoch = epoch_offset + local_epoch
        epochs_run = local_epoch + 1
        model.train()
        totals: dict[str, float] = {}
        grad_norms: dict[str, list[float]] = {"matrix": [], "p": [], "mu": [], "center": []}
        perm = torch.randperm(n, device=device)
        seen = 0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            if len(idx) < min_batch:
                continue
            xb, yb = x_fit_t[idx], y_fit_t[idx]
            residual = (yb - center_fit_t[idx]) if center_fit_t is not None else (
                yb - model.predict_center(xb))
            opt.zero_grad(set_to_none=True)
            parts = model.loss(xb, residual, cfg.alpha, cfg.order_statistic_convention,
                               cfg.average_volume_term)
            n_batches += 1
            if not torch.isfinite(parts.total_loss):
                n_nonfinite += 1
                logger.warning("mvcs[%s]: non-finite loss at epoch %d, batch skipped", stage, epoch)
                continue
            parts.total_loss.backward()
            clipped = [t for g in opt.param_groups for t in g["params"]]
            torch.nn.utils.clip_grad_norm_(clipped, cfg.grad_clip_norm)
            for g in opt.param_groups:
                grad_norms[g["name"]].append(_grad_norm(g["params"]))
            opt.step()
            model.project_p()
            seen += len(idx)
            for key in ("total_loss", "volume_term", "order_statistic_term", "unit_ball_term"):
                totals[key] = totals.get(key, 0.0) + float(getattr(parts, key).detach()) * len(idx)
            for key in ("training_threshold", "training_empirical_coverage"):
                totals[key] = totals.get(key, 0.0) + float(getattr(parts, key)) * len(idx)
            totals["order_statistic_rank"] = parts.order_statistic_rank
        if seen > 0:
            sched.step()

        model.eval()
        row = {"stage": stage, "epoch": epoch,
               "order_statistic_convention": cfg.order_statistic_convention,
               **{k: (v / max(seen, 1) if k != "order_statistic_rank" else v)
                  for k, v in totals.items()},
               **{f"gradient_norm_{k}": (float(np.mean(v)) if v else 0.0)
                  for k, v in grad_norms.items()}}
        with torch.no_grad():
            row["effective_p"] = float(model.effective_p())
            row["p_raw"] = float(model.p_raw)
        history.append(row)
        p_history.append({"stage": stage, "epoch": epoch, "p_raw": row["p_raw"],
                          "effective_p": row["effective_p"]})

        if (local_epoch % max(cfg.checkpoint_every, 1)) == 0 or local_epoch == n_epochs - 1:
            ck = selector.consider(epoch, model, score_view, inner["x_cal"], inner["t_cal"],
                                   inner["x_eval"], inner["t_eval"], center=checkpoint_center,
                                   x_full=inner.get("x_full"), t_full=inner.get("t_full"))
            metric = ck["selection_metric"]
            if np.isfinite(metric) and metric < best_metric_seen - 1e-9 and ck["eligible"]:
                best_metric_seen, epochs_since_improvement = metric, 0
            else:
                epochs_since_improvement += max(cfg.checkpoint_every, 1)
            if epochs_since_improvement >= cfg.patience:
                logger.info("mvcs[%s]: early stop at epoch %d", stage, epoch)
                break

    return n_batches, n_nonfinite, epochs_run


def _inner_split(x_val: np.ndarray, t_val: np.ndarray, frac: float, seed: int) -> dict:
    """Halve the validation split (kept for the history rows; the selector uses x_full/t_full)."""
    n = len(x_val)
    if n < 4:
        raise ValueError(f"score-validation subset has only {n} rows; need >= 4 to split")
    perm = np.random.default_rng(seed).permutation(n)
    n_cal = max(2, int(round(frac * n)))
    n_cal = min(n_cal, n - 2)
    cal_i, eval_i = perm[:n_cal], perm[n_cal:]
    return {"x_cal": x_val[cal_i], "t_cal": t_val[cal_i],
            "x_eval": x_val[eval_i], "t_eval": t_val[eval_i],
            "x_full": x_val, "t_full": t_val}


def _build_model(context_dim: int, cfg: MvcsTrainConfig) -> MvcsSingleNormModel:
    torch.manual_seed(cfg.seed)
    return MvcsSingleNormModel(
        context_dim=context_dim, output_dim=cfg.output_dim, hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers, ridge=cfg.ridge, p_init=cfg.p_init, p_min=cfg.p_min,
    )


def _finalize(model, cfg, device, selector, history, p_history, stages,
              n_batches, n_nonfinite, t0) -> MvcsTrainResult:
    state, epoch, rule = selector.best()
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    frac = n_nonfinite / max(n_batches, 1)
    if frac > cfg.nonfinite_batch_tolerance:
        raise RuntimeError(
            f"MVCS training saw {n_nonfinite}/{n_batches} non-finite batches "
            f"({frac:.1%}), above nonfinite_batch_tolerance={cfg.nonfinite_batch_tolerance:.1%}. "
            f"Lower the learning rates or raise `ridge`."
        )
    summary = selector.summary_for(epoch)
    return MvcsTrainResult(
        score=MvcsScore(model=model, device=device), selected_epoch=epoch, selection_rule=rule,
        validation_coverage=float(summary["validation_coverage"]),
        validation_mean_volume=float(summary["validation_mean_volume"]),
        validation_conformal_threshold=float(summary["validation_conformal_threshold"]),
        training_history=history, checkpoint_history=selector.history, learned_p_history=p_history,
        stage_summaries=stages, n_batches=n_batches, n_nonfinite_batches=n_nonfinite,
        runtime_seconds=time.time() - t0,
        validation_mvcs_objective=float(summary.get("validation_mvcs_objective", float("nan"))),
        checkpoint_selection=VALIDATION_MVCS_OBJECTIVE,
    )


def train_mvcs_fixed_center(
    x_fit: np.ndarray, r_fit: np.ndarray, x_val: np.ndarray, r_val: np.ndarray,
    cfg: MvcsTrainConfig,
) -> MvcsTrainResult:
    """MVCS geometry on residuals r = y - F(x) of the frozen point predictor."""
    t0 = time.time()
    device = _resolve_device(cfg.device)
    x_fit, x_val = _as_2d(x_fit), _as_2d(x_val)
    model = _build_model(x_fit.shape[1], cfg).to(device)
    model.set_context_normalization(x_fit)

    inner = _inner_split(x_val, np.asarray(r_val, dtype=float), cfg.inner_calibration_frac, cfg.seed)
    selector = _ObjectiveCheckpointSelector(cfg.alpha, cfg.order_statistic_convention,
                                            cfg.average_volume_term)
    history: list[dict] = []
    p_history: list[dict] = []

    x_t = torch.as_tensor(x_fit, dtype=torch.float32, device=device)
    r_t = torch.as_tensor(np.asarray(r_fit, dtype=float), dtype=torch.float32, device=device)
    zeros = torch.zeros_like(r_t)   # residuals are already centred

    n_batches, n_nonfinite, epochs_run = _run_mvcs_epochs(
        model, cfg, device, x_t, r_t, zeros, inner, cfg.max_epochs, "fixed_center",
        include_center=False, selector=selector, history=history, p_history=p_history,
    )
    stages = [{"stage": "fixed_center", "epochs_run": epochs_run,
               "epochs_configured": cfg.max_epochs}]
    return _finalize(model, cfg, device, selector, history, p_history, stages,
                     n_batches, n_nonfinite, t0)
