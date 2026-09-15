"""Training loop for one learned conformity score (ICNN or spline) at fixed loss weights.

Adam with gradient clipping on `prediction_set_loss`; early stopping tracks the validation
loss only to decide when to stop. Which epoch's weights are kept is decided separately, on real
calibrated coverage and set size (see checkpoint_selection.py). At the end the model's
calibrated coverage and size are measured on a held-out half of the validation split
(`final_val_real_*`); the lambda search selects candidates on those two numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from coconet.conformal.calibration import conformal_threshold
from coconet.data.splits import make_internal_validation_split
from coconet.training.deterministic_size import deterministic_size_fn
from coconet.training.checkpoint_selection import (
    CalibratedCheckpointTracker,
    evaluate_calibrated_checkpoint,
    validate_selection_metric,
)
from coconet.training.early_stopping import EarlyStopper
from coconet.training.score_thresholds import (
    compute_training_tau,
    resolve_temperature,
    threshold_diagnostics,
)
from coconet.training.prediction_set_loss import (
    MISS_SCORE_SPACE,
    LossConfig,
    prediction_set_loss,
    reference_scale,
)
from coconet.utils import get_logger

logger = get_logger(__name__)


class ScoreTrainingError(RuntimeError):
    pass


def _make_optimizer(config, params, lr: float):
    name = str(getattr(config, "optimizer", "adam")).lower()
    wd = float(getattr(config, "optimizer_weight_decay", 0.0) or 0.0)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    raise ValueError(f"optimizer must be 'adam' or 'adamw', got {name!r}")


@dataclass
class ScoreHyperparams:
    learning_rate: float
    lambda_cov: float
    lambda_l2: float
    beta: float                      # coverage sigmoid sharpness kappa_cov
    num_knots: int | None = None     # spline only
    encoder_hidden_dims: tuple[int, ...] | None = None
    icnn_hidden_dim: int | None = None
    icnn_depth: int | None = None
    icnn_hidden_dims: tuple[int, ...] | None = None

    def key(self) -> str:
        base = f"lr={self.learning_rate}_lcov={self.lambda_cov}_ll2={self.lambda_l2}_beta={self.beta}"
        if self.num_knots is not None:
            base += f"_knots={self.num_knots}"
        if self.encoder_hidden_dims is not None:
            base += f"_enc={list(self.encoder_hidden_dims)}"
        if self.icnn_hidden_dim is not None:
            base += f"_icnnh={self.icnn_hidden_dim}"
        if self.icnn_depth is not None:
            base += f"_icnnd={self.icnn_depth}"
        if self.icnn_hidden_dims is not None:
            base += f"_icnndims={list(self.icnn_hidden_dims)}"
        return base


@dataclass
class ScoreTrainConfig:
    epsilon: float = 1e-3
    grad_clip_norm: float | None = 5.0
    max_epochs: int = 300
    patience: int = 40
    batch_size: int = 256
    coverage_tolerance: float = 0.02
    # Lambda-search selection when no candidate is coverage-admissible (lambda_search.py).
    lambda_fallback_rule: str = "undercoverage_gamma_distance"
    # "staged" = coordinate descent over the gamma grid; "grid" = train gamma_grid_miss x gamma_grid_cov.
    lambda_search_mode: str = "staged"
    gamma_grid_miss: tuple = ()
    gamma_grid_cov: tuple = ()
    seed: int = 0
    miss_type: str = MISS_SCORE_SPACE
    lambda_miss: float | None = None     # None: resolved from the reference scale
    lambda_miss_gamma: float = 1.0
    kappa_miss: float = 10.0
    size_objective: str = "raw_mean_volume"
    train_tau_mode: str = "quantile"
    train_tau_fixed: float = 1.0
    coverage_temperature_mode: str = "fixed"
    score_space_miss_form: str = "relative_relu"
    optimizer: str = "adamw"
    optimizer_weight_decay: float = 0.0
    coverage_temperature_multiplier: float = 1.0
    size_reference_scale: float = 1.0
    checkpoint_selection_metric: str = "calibrated_validation_area"
    checkpoint_every: int = 5
    min_selection_epoch: int = 0

    def __post_init__(self):
        if self.miss_type != MISS_SCORE_SPACE:
            raise ValueError(f"miss_type must be {MISS_SCORE_SPACE!r}, got {self.miss_type!r}")


@dataclass
class ScoreTrainResult:
    model: torch.nn.Module
    hyperparams: ScoreHyperparams
    train_width_history: list[float] = field(default_factory=list)
    train_cov_history: list[float] = field(default_factory=list)
    train_l2_history: list[float] = field(default_factory=list)
    train_total_history: list[float] = field(default_factory=list)
    val_width_history: list[float] = field(default_factory=list)
    val_smoothed_coverage_history: list[float] = field(default_factory=list)
    val_total_history: list[float] = field(default_factory=list)
    best_epoch: int = -1
    selected_epoch: int = -1
    selection_rule: str = "validation_total_loss"
    diagnostics_history: list[dict] = field(default_factory=list)
    checkpoint_history: list[dict] = field(default_factory=list)
    final_val_width: float = float("nan")
    final_val_smoothed_coverage: float = float("nan")
    # Calibrated coverage and size on the held-out half of the validation split.
    final_val_real_coverage: float = float("nan")
    final_val_real_width: float = float("nan")
    lambda_miss_used: float = float("nan")
    lambda_miss_diagnostics: dict = field(default_factory=dict)


def score_learning_loss(
    model: torch.nn.Module,
    x: torch.Tensor,
    r: torch.Tensor,
    alpha: float,
    beta: float,
    lambda_cov: float,
    lambda_l2: float,
    miss_type: str = MISS_SCORE_SPACE,
    lambda_miss: float | None = None,
    kappa_miss: float = 10.0,
    train_tau_mode: str = "quantile",
    train_tau_fixed: float = 1.0,
    coverage_temperature_mode: str = "fixed",
    score_space_miss_form: str = "relative_relu",
    coverage_temperature_multiplier: float = 1.0,
    size_reference_scale: float = 1.0,
    size_objective: str = "raw_mean_volume",
    return_parts: bool = False,
):
    """`prediction_set_loss` with the (total, size, coverage, l2) return shape the loop uses."""
    if lambda_miss is None:
        raise ScoreTrainingError("lambda_miss was not supplied")
    cfg = LossConfig(
        miss_type=miss_type,
        lambda_miss=lambda_miss,
        lambda_coverage=lambda_cov,
        lambda_l2=lambda_l2,
        kappa_miss=kappa_miss,
        kappa_coverage=beta,
        size_objective=size_objective,
        train_tau_mode=train_tau_mode,
        train_tau_fixed=train_tau_fixed,
        coverage_temperature_mode=coverage_temperature_mode,
        score_space_miss_form=score_space_miss_form,
        coverage_temperature_multiplier=coverage_temperature_multiplier,
        size_reference_scale=size_reference_scale,
    )
    try:
        total, parts = prediction_set_loss(model, x, r, alpha, cfg)
    except ValueError as exc:
        raise ScoreTrainingError(str(exc)) from exc

    size_t = parts["size_term"]
    coverage_t = parts["weighted_coverage_penalty"]
    l2_t = parts["weighted_regularization"]
    parts["width"] = size_t
    parts["coverage_term"] = coverage_t
    parts["outside_miss"] = parts["miss_term"]
    if return_parts:
        return total, size_t, coverage_t, l2_t, parts
    return total, size_t, coverage_t, l2_t


def _resolve_lambda_miss(model, X_train, r_train, alpha: float,
                          config: "ScoreTrainConfig", device: str = "cpu") -> tuple[float, dict]:
    """The miss weight: as supplied, or lambda_miss_gamma * lambda_ref from the reference scale."""
    if config.lambda_miss is not None:
        lam = float(config.lambda_miss)
        if not np.isfinite(lam) or lam < 0:
            raise ScoreTrainingError(f"lambda_miss must be finite and nonnegative, got {lam}")
        return lam, {"lambda_miss_source": "explicit", "lambda_miss": lam,
                     "lambda_miss_gamma": float("nan"), "lambda_ref": float("nan"),
                     "reference_size_term": float("nan"), "reference_miss_term": float("nan")}

    probe_cfg = LossConfig(
        miss_type=config.miss_type,
        lambda_miss=0.0,
        kappa_miss=config.kappa_miss,
        size_objective=config.size_objective,
        train_tau_mode=config.train_tau_mode,
        train_tau_fixed=config.train_tau_fixed,
        coverage_temperature_mode=config.coverage_temperature_mode,
        score_space_miss_form=getattr(config, "score_space_miss_form", "relative_relu"),
        coverage_temperature_multiplier=config.coverage_temperature_multiplier,
        size_reference_scale=getattr(config, 'size_reference_scale', 1.0),
    )
    ref = reference_scale(model, X_train, r_train, alpha, probe_cfg,
                          batch_size=config.batch_size, seed=config.seed, device=device)
    lam = float(config.lambda_miss_gamma) * ref.lambda_ref
    logger.info("lambda_miss resolved from the reference scale: S_ref=%.6g M_ref=%.6g "
                "lambda_ref=%.6g gamma=%.6g -> lambda_miss=%.6g",
                ref.s_ref, ref.m_ref, ref.lambda_ref, config.lambda_miss_gamma, lam)
    return lam, {"lambda_miss_source": "reference_scale", "lambda_miss": lam,
                 "lambda_miss_gamma": float(config.lambda_miss_gamma),
                 "lambda_ref": ref.lambda_ref, "reference_size_term": ref.s_ref,
                 "reference_miss_term": ref.m_ref}


def _init_qmc_schedule(model, seed: int) -> None:
    """Key the d > 2 training QMC scramble to the experiment seed."""
    if hasattr(model, "set_qmc_epoch"):
        model.qmc_base_seed = int(seed) * 100_003
        model.set_qmc_epoch(0)


def train_one_config(
    model_factory: Callable[[], torch.nn.Module],
    X_train: np.ndarray,
    r_train: np.ndarray,
    X_val: np.ndarray,
    r_val: np.ndarray,
    alpha: float,
    hp: ScoreHyperparams,
    config: ScoreTrainConfig,
    device: str = "cpu",
) -> ScoreTrainResult:
    torch.manual_seed(config.seed)
    model = model_factory().to(device)

    train_ds = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(r_train, dtype=torch.float32).reshape(r_train.shape[0], -1),
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    r_val_t = torch.as_tensor(r_val, dtype=torch.float32, device=device).reshape(r_val.shape[0], -1)

    # A rotated score (the spline at d >= 2) keeps its angle head in its own parameter group.
    rot_params = list(model.rotation_parameters()) if hasattr(model, "rotation_parameters") else []
    if rot_params:
        rot_ids = {id(p) for p in rot_params}
        rest = [p for p in model.parameters() if id(p) not in rot_ids]
        mult = float(getattr(model, "rotation_lr_multiplier",
                             getattr(getattr(model, "config", None), "lr_multiplier", 1.0)))
        optimizer = _make_optimizer(
            config, [{"params": rest, "lr": hp.learning_rate},
                     {"params": rot_params, "lr": hp.learning_rate * mult, "name": "rotation"}],
            hp.learning_rate)
    else:
        optimizer = _make_optimizer(config, model.parameters(), hp.learning_rate)
    stopper = EarlyStopper(patience=config.patience, min_delta=1e-6)

    logger.info("Training tau mode: %s", config.train_tau_mode)
    logger.info("Training tau level: 1 - alpha = %.4f", 1.0 - alpha)
    selection_metric = validate_selection_metric(config.checkpoint_selection_metric)
    # The validation split is halved once into inner-calibration / inner-evaluation parts,
    # shared by the periodic checkpoints and the final calibrated numbers.
    _inner = make_internal_validation_split(np.arange(len(X_val)), 0.5, seed=config.seed)
    X_inner_cal_t = torch.as_tensor(X_val[_inner.train_idx], dtype=torch.float32, device=device)
    r_inner_cal_t = torch.as_tensor(
        r_val[_inner.train_idx], dtype=torch.float32, device=device).reshape(len(_inner.train_idx), -1)
    X_inner_eval_t = torch.as_tensor(X_val[_inner.val_idx], dtype=torch.float32, device=device)
    r_inner_eval_t = torch.as_tensor(
        r_val[_inner.val_idx], dtype=torch.float32, device=device).reshape(len(_inner.val_idx), -1)
    tracker = CalibratedCheckpointTracker(alpha=alpha, coverage_tolerance=config.coverage_tolerance,
                                          min_selection_epoch=config.min_selection_epoch)
    consecutive_pinned = 0

    result = ScoreTrainResult(model=model, hyperparams=hp)

    lambda_miss, lambda_miss_diag = _resolve_lambda_miss(
        model, X_train, r_train, alpha, config, device=device)
    result.lambda_miss_used = float("nan") if lambda_miss is None else lambda_miss
    result.lambda_miss_diagnostics = lambda_miss_diag

    _init_qmc_schedule(model, config.seed)
    for epoch in range(config.max_epochs):
        if hasattr(model, "set_qmc_epoch"):
            model.set_qmc_epoch(epoch)
        model.train()
        epoch_total, epoch_width, epoch_cov, epoch_l2 = [], [], [], []
        for xb, rb in train_loader:
            xb, rb = xb.to(device), rb.to(device)
            optimizer.zero_grad()
            total, width_t, cov_t, l2_t, parts_b = score_learning_loss(
                model, xb, rb, alpha, hp.beta, hp.lambda_cov,
                hp.lambda_l2,
                miss_type=config.miss_type, lambda_miss=lambda_miss,
                kappa_miss=config.kappa_miss, size_objective=config.size_objective,
                train_tau_mode=config.train_tau_mode, train_tau_fixed=config.train_tau_fixed,
                coverage_temperature_mode=config.coverage_temperature_mode,
                coverage_temperature_multiplier=config.coverage_temperature_multiplier,
                size_reference_scale=getattr(config, 'size_reference_scale', 1.0),
                score_space_miss_form=getattr(config, 'score_space_miss_form', 'relative_relu'),
                return_parts=True,
            )
            total.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            epoch_total.append(total.item())
            epoch_width.append(width_t.item())
            epoch_cov.append(cov_t.item())
            epoch_l2.append(l2_t.item())

        result.train_total_history.append(float(np.mean(epoch_total)))
        result.train_width_history.append(float(np.mean(epoch_width)))
        result.train_cov_history.append(float(np.mean(epoch_cov)))
        result.train_l2_history.append(float(np.mean(epoch_l2)))

        model.eval()
        with torch.no_grad():
            val_total, val_width_t, val_cov_t, _, val_parts = score_learning_loss(
                model, X_val_t, r_val_t, alpha, hp.beta, hp.lambda_cov,
                hp.lambda_l2,
                miss_type=config.miss_type, lambda_miss=lambda_miss,
                kappa_miss=config.kappa_miss, size_objective=config.size_objective,
                train_tau_mode=config.train_tau_mode, train_tau_fixed=config.train_tau_fixed,
                coverage_temperature_mode=config.coverage_temperature_mode,
                coverage_temperature_multiplier=config.coverage_temperature_multiplier,
                size_reference_scale=getattr(config, 'size_reference_scale', 1.0),
                score_space_miss_form=getattr(config, 'score_space_miss_form', 'relative_relu'),
                return_parts=True,
            )
            diag = threshold_diagnostics(
                model.score(X_val_t, r_val_t), val_parts["tau_used"], val_parts["temperature_used"])
        val_smoothed_cov = diag.smooth_coverage

        val_total = val_total.detach()
        result.val_total_history.append(float(val_total.item()))
        result.val_width_history.append(float(val_width_t.detach().item()))
        result.val_smoothed_coverage_history.append(float(val_smoothed_cov))
        row = {"epoch": epoch, "coverage_term": float(val_parts["coverage_term"].detach()),
               "outside_miss_term": float(val_parts["outside_miss"].detach()),
               "width_or_area_term": float(val_width_t.detach()),
               "total_loss": float(val_total.item()), **diag.__dict__}
        for _k, _v in val_parts.items():
            if _k.startswith("miss_") and torch.is_tensor(_v) and _v.ndim == 0:
                row[_k] = float(_v.detach())
        result.diagnostics_history.append(row)

        if epoch % max(1, config.checkpoint_every) == 0:
            logger.info("epoch %d: tau_train=%.6g (mode=%s) temperature=%.6g smooth_cov=%.4f",
                        epoch, diag.tau_used, config.train_tau_mode, diag.temperature_used,
                        diag.smooth_coverage)

        consecutive_pinned = consecutive_pinned + 1 if diag.saturated else 0
        if consecutive_pinned in (10, 50, 200):
            logger.warning(
                "Coverage term pinned at %.3f for %d consecutive epochs (tau=%.4g, T=%.4g, "
                "sigmoid args [%.3g, %.3g]) -- it is contributing no gradient",
                diag.smooth_coverage, consecutive_pinned, diag.tau_used, diag.temperature_used,
                diag.sigmoid_argument_min, diag.sigmoid_argument_max)

        if selection_metric == "calibrated_validation_area" and (
                epoch % max(1, config.checkpoint_every) == 0 or epoch == config.max_epochs - 1):
            tracker.record(evaluate_calibrated_checkpoint(
                model, X_inner_cal_t, r_inner_cal_t, X_inner_eval_t, r_inner_eval_t, alpha, epoch,
                surrogate_loss=float(val_total.item()), surrogate_area=float(val_width_t.detach()),
                smooth_coverage=float(val_smoothed_cov)), model)

        should_stop = stopper.step(float(val_total.item()), epoch, model)
        if should_stop:
            logger.info(
                "Score model (%s) early stopping at epoch %d (best=%d)", hp.key(), epoch, stopper.best_epoch
            )
            break

    result.best_epoch = stopper.best_epoch
    result.checkpoint_history = tracker.rows()
    if selection_metric == "calibrated_validation_area" and tracker.restore_best(model):
        result.selected_epoch = tracker.best.epoch
        result.selection_rule = tracker.selection_rule
        logger.info(
            "Restored checkpoint by calibrated validation area: epoch %d "
            "(area=%.4g, coverage=%.4f) instead of surrogate-best epoch %d [%s]",
            tracker.best.epoch, tracker.best.actual_validation_mean_area,
            tracker.best.actual_validation_coverage, stopper.best_epoch, tracker.selection_rule)
    else:
        stopper.restore_best(model)
        result.selected_epoch = stopper.best_epoch
        result.selection_rule = "validation_total_loss"

    model.eval()
    with torch.no_grad():
        size_fn = deterministic_size_fn(model)
        final_scores = model.score(X_val_t, r_val_t)
        final_temp = resolve_temperature(
            final_scores, config.coverage_temperature_mode, hp.beta,
            config.coverage_temperature_multiplier)
        final_tau = compute_training_tau(final_scores, alpha, config.train_tau_mode,
                                         fixed_tau=config.train_tau_fixed,
                                         temperature=final_temp)
        result.final_val_width = float(size_fn(X_val_t, tau=float(final_tau)).mean().item())
        result.final_val_smoothed_coverage = float(
            torch.sigmoid((final_tau - final_scores) / final_temp).mean().item()
        )

        # Real split conformal on the two halves of the validation split.
        X_val_cal_t, r_val_cal_t = X_inner_cal_t, r_inner_cal_t
        X_val_test_t, r_val_test_t = X_inner_eval_t, r_inner_eval_t

        val_cal_scores = model.score(X_val_cal_t, r_val_cal_t).cpu().numpy().reshape(-1)
        q_hat_val = conformal_threshold(val_cal_scores, alpha)
        val_test_scores = model.score(X_val_test_t, r_val_test_t).cpu().numpy().reshape(-1)
        result.final_val_real_coverage = float(np.mean(val_test_scores <= q_hat_val))
        result.final_val_real_width = (
            float("inf") if math.isinf(q_hat_val)
            else float(size_fn(X_val_test_t, tau=q_hat_val).mean().item())
        )
    result.model.to("cpu")
    return result
