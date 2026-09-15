"""The training objective of every learned conformity score (CoCoNet_ICNN and CoCoNet_Spline).

    L = L_size + lambda_miss * L_miss + lambda_cov * L_cov + lambda_l2 * R(theta)

evaluated at one threshold tau per batch (tau = 1 in the paper):

    L_size  size of the tau-sublevel set, from the model's own geometry API
            (paper: root_mean_volume = (E[V])^(1/d), computed through logsumexp)
    L_miss  score-space hinge on points outside the set: mean [ (s - tau) / (|tau| + eps) ]_+
    L_cov   (mean sigmoid((tau - s) / T) - (1 - alpha))^2,  T = 1 / kappa_cov
    R       L2 penalty on the non-bias weights (lambda_l2 = 0 in the paper)

The two loss weights are searched as multiples gamma of reference scales measured on the
freshly initialised model (`reference_scale`): lambda_miss = gamma_miss * lambda_miss_ref and
lambda_cov = gamma_cov * lambda_cov_ref, so the same gamma grid means the same thing on every
dataset and dimension.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from coconet.training.score_thresholds import compute_training_tau, resolve_temperature

logger = logging.getLogger(__name__)

MISS_SCORE_SPACE = "score_space"
MISS_TYPES = (MISS_SCORE_SPACE,)

SIZE_OBJECTIVE_RAW_MEAN_VOLUME = "raw_mean_volume"
SIZE_OBJECTIVE_LOG_MEAN_VOLUME = "log_mean_volume"
SIZE_OBJECTIVE_MEAN_LOG_VOLUME = "mean_log_volume"
SIZE_OBJECTIVE_MEAN_ROOT_VOLUME = "mean_root_volume"
SIZE_OBJECTIVE_ROOT_MEAN_VOLUME = "root_mean_volume"
SIZE_OBJECTIVE_LOG1P_MEAN_VOLUME = "log1p_mean_volume"
SIZE_OBJECTIVE_MEAN_LOG1P_VOLUME = "mean_log1p_volume"
SIZE_OBJECTIVES = (
    SIZE_OBJECTIVE_RAW_MEAN_VOLUME, SIZE_OBJECTIVE_LOG_MEAN_VOLUME,
    SIZE_OBJECTIVE_MEAN_LOG_VOLUME, SIZE_OBJECTIVE_MEAN_ROOT_VOLUME,
    SIZE_OBJECTIVE_LOG1P_MEAN_VOLUME, SIZE_OBJECTIVE_MEAN_LOG1P_VOLUME,
    SIZE_OBJECTIVE_ROOT_MEAN_VOLUME,
)
LOG_SIZE_OBJECTIVES = (SIZE_OBJECTIVE_LOG_MEAN_VOLUME, SIZE_OBJECTIVE_MEAN_LOG_VOLUME,
                       SIZE_OBJECTIVE_LOG1P_MEAN_VOLUME, SIZE_OBJECTIVE_MEAN_LOG1P_VOLUME)

REFERENCE_EPSILON = 1e-8
SCORE_SPACE_MISS_EPS = 1e-8
SCORE_SPACE_MISS_FORM_RELATIVE = "relative_relu"
SCORE_SPACE_MISS_FORM_SOFTPLUS = "softplus_absolute"
SCORE_SPACE_MISS_FORMS = (SCORE_SPACE_MISS_FORM_RELATIVE, SCORE_SPACE_MISS_FORM_SOFTPLUS)

# Gamma grid of the staged (coordinate-descent) search used by the synthetic experiment.
PRODUCTION_GAMMA_GRID_FULL = (0.0, 0.01, 0.03, 0.1, 1.0, 10.0)
PRODUCTION_GAMMA_GRID_STAGE1 = (0.0, 0.01, 0.03, 0.1, 1.0, 10.0)
PRODUCTION_GAMMA_EXPANSION_LOWER: tuple[float, ...] = ()


def production_stage2_gammas(selected: float,
                             evaluated: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    """Neighbours of `selected` in PRODUCTION_GAMMA_GRID_FULL that were not evaluated yet."""
    grid = PRODUCTION_GAMMA_GRID_FULL
    done = [float(g) for g in evaluated]
    pos = next((i for i, g in enumerate(grid) if np.isclose(g, selected)), None)
    if pos is None:
        return ()
    neighbours = []
    if pos > 0:
        neighbours.append(grid[pos - 1])
    if pos < len(grid) - 1:
        neighbours.append(grid[pos + 1])
    return tuple(g for g in neighbours if not any(np.isclose(g, d) for d in done))


def production_boundary_expansion(selected: float,
                                  evaluated: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    """Extra gammas to try when the winner sits on a grid boundary (none for the paper grid)."""
    done = [float(g) for g in evaluated]

    def _todo(candidates):
        return tuple(g for g in candidates if not any(np.isclose(g, d) for d in done))

    if np.isclose(selected, 100.0):
        return _todo((300.0,))
    if np.isclose(selected, 300.0):
        return _todo((1000.0,))
    if np.isclose(selected, 0.01) and any(np.isclose(0.0, d) for d in done):
        return _todo(PRODUCTION_GAMMA_EXPANSION_LOWER)
    return ()


@dataclass
class LossConfig:
    miss_type: str = MISS_SCORE_SPACE
    lambda_miss: float = float("nan")      # must be set explicitly
    lambda_coverage: float = 1.0
    lambda_l2: float = 0.0
    kappa_miss: float = 10.0               # softplus form only
    score_space_miss_form: str = SCORE_SPACE_MISS_FORM_RELATIVE
    kappa_coverage: float = 10.0
    size_objective: str = SIZE_OBJECTIVE_RAW_MEAN_VOLUME
    size_reference_scale: float = 1.0
    train_tau_mode: str = "quantile"
    train_tau_fixed: float = 1.0
    coverage_temperature_mode: str = "fixed"
    coverage_temperature_multiplier: float = 1.0

    def __post_init__(self):
        if self.miss_type not in MISS_TYPES:
            raise ValueError(f"Unknown miss_type {self.miss_type!r}; choose from {MISS_TYPES}")
        if self.size_objective not in SIZE_OBJECTIVES:
            raise ValueError(f"size_objective must be one of {SIZE_OBJECTIVES}, "
                             f"got {self.size_objective!r}")


def prediction_set_loss(model, x: torch.Tensor, residual: torch.Tensor, alpha: float,
                         config: LossConfig) -> tuple[torch.Tensor, dict]:
    """Returns (total_loss, diagnostics)."""
    miss_type = config.miss_type
    lam = float(config.lambda_miss)
    if not np.isfinite(lam):
        raise ValueError("lambda_miss must be set explicitly (got NaN)")
    if lam < 0:
        raise ValueError(f"lambda_miss must be nonnegative, got {lam}")

    tau, temperature = _resolve_tau_and_temperature(model, x, residual, alpha, config)

    size_loss = size_term(model, x, tau, config.size_objective,
                          getattr(config, "size_reference_scale", 1.0))

    scores = model.score(x, residual).reshape(-1)
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite score encountered in prediction_set_loss")

    miss = compute_miss_loss(model=model, x=x, residual=residual, scores=scores, tau=tau,
                             miss_type=miss_type, config=config)
    miss_loss = miss["miss"]

    coverage_loss_value, smooth_coverage = coverage_loss(scores, tau, temperature, alpha)

    reg_loss = model.l2_penalty()

    total = (size_loss
             + config.lambda_miss * miss_loss
             + config.lambda_coverage * coverage_loss_value
             + config.lambda_l2 * reg_loss)
    if not torch.isfinite(total):
        raise ValueError(f"Non-finite total loss for miss_type={miss_type!r}")

    diagnostics = {
        "size_term": size_loss,
        "miss_term": miss_loss,
        "weighted_miss_term": config.lambda_miss * miss_loss,
        "coverage_penalty": coverage_loss_value,
        "weighted_coverage_penalty": config.lambda_coverage * coverage_loss_value,
        "regularization": reg_loss,
        "weighted_regularization": config.lambda_l2 * reg_loss,
        "lambda_miss": torch.as_tensor(lam),
        "smooth_coverage": smooth_coverage,
        "miss_type": miss_type,
        "tau_used": tau.detach() if torch.is_tensor(tau) else torch.as_tensor(float(tau)),
        "temperature_used": temperature.detach(),
        "total": total,
    }
    for k, v in miss.items():
        if k in ("miss", "per_observation"):
            continue
        diagnostics[f"miss_{k}"] = v
    return total, diagnostics


def compute_miss_loss(model, x: torch.Tensor, residual: torch.Tensor, scores: torch.Tensor,
                      tau, miss_type: str, config: LossConfig | None = None) -> dict:
    cfg = config or LossConfig(lambda_miss=0.0)
    if miss_type == MISS_SCORE_SPACE:
        if getattr(cfg, "score_space_miss_form", SCORE_SPACE_MISS_FORM_RELATIVE) \
                == SCORE_SPACE_MISS_FORM_SOFTPLUS:
            return score_space_miss_softplus(scores, tau, kappa=cfg.kappa_miss)
        return score_space_miss(scores, tau, kappa=cfg.kappa_miss)
    raise ValueError(f"Unknown miss_type {miss_type!r}; choose from {MISS_TYPES}")


def _residual_dim(model) -> int:
    return int(getattr(model, "output_dim", None) or getattr(model, "residual_dim", 1))


def _as_float(tau) -> float:
    return float(tau.detach()) if torch.is_tensor(tau) else float(tau)


def _resolve_tau_and_temperature(model, x, r, alpha: float, cfg: LossConfig):
    """One (tau, T) per loss evaluation, shared by the size, miss and coverage terms."""
    tau = torch.as_tensor(float(cfg.train_tau_fixed), dtype=x.dtype, device=x.device)
    temperature = torch.as_tensor(1.0 / cfg.kappa_coverage, dtype=x.dtype, device=x.device)
    if cfg.train_tau_mode != "fixed" or cfg.coverage_temperature_mode != "fixed":
        s_probe = model.score(x, r)
        if not torch.isfinite(s_probe).all():
            raise ValueError("Non-finite score encountered while resolving the training threshold")
        temperature = resolve_temperature(s_probe, cfg.coverage_temperature_mode,
                                          cfg.kappa_coverage, cfg.coverage_temperature_multiplier)
        tau = compute_training_tau(s_probe, alpha, cfg.train_tau_mode,
                                   fixed_tau=cfg.train_tau_fixed, temperature=temperature)
    return tau, temperature


def size_term(model, x: torch.Tensor, tau, size_objective: str,
              reference_scale: float = 1.0) -> torch.Tensor:
    """The size term, from the model's own width/area/volume hooks (log space for d > 2)."""
    if size_objective not in SIZE_OBJECTIVES:
        raise ValueError(f"size_objective must be one of {SIZE_OBJECTIVES}, got {size_objective!r}")
    tau_f = _as_float(tau)
    if size_objective == SIZE_OBJECTIVE_RAW_MEAN_VOLUME:
        return model.prediction_set_size(x, tau=tau_f).mean()
    if not hasattr(model, "log_prediction_set_size"):
        raise AttributeError(
            f"size_objective={size_objective!r} requires model.log_prediction_set_size(x, tau), "
            f"which {type(model).__name__} does not implement -- fall back to "
            f"size_objective='{SIZE_OBJECTIVE_RAW_MEAN_VOLUME}' for this model.")
    log_v = model.log_prediction_set_size(x, tau=tau_f).reshape(-1)
    if size_objective == SIZE_OBJECTIVE_LOG_MEAN_VOLUME:
        return torch.logsumexp(log_v, dim=0) - math.log(log_v.shape[0])
    if size_objective == SIZE_OBJECTIVE_MEAN_LOG_VOLUME:
        return log_v.mean()
    if size_objective == SIZE_OBJECTIVE_LOG1P_MEAN_VOLUME:
        return F.softplus(torch.logsumexp(log_v, dim=0) - math.log(log_v.shape[0]))
    if size_objective == SIZE_OBJECTIVE_MEAN_LOG1P_VOLUME:
        return F.softplus(log_v).mean()
    if size_objective == SIZE_OBJECTIVE_ROOT_MEAN_VOLUME:
        log_mean = torch.logsumexp(log_v, dim=0) - math.log(log_v.shape[0])
        return torch.exp(log_mean / _residual_dim(model)) / reference_scale
    return torch.exp(log_v / _residual_dim(model)).mean()


def coverage_loss(scores: torch.Tensor, tau, temperature: torch.Tensor,
                   alpha: float, mode: str = "symmetric") -> tuple[torch.Tensor, torch.Tensor]:
    """(mean sigmoid((tau - s) / T) - (1 - alpha))^2 and the smooth coverage itself."""
    smooth_coverage = torch.sigmoid((tau - scores) / temperature).mean()
    if mode != "symmetric":
        raise ValueError(f"coverage_penalty_mode must be 'symmetric', got {mode!r}")
    return (smooth_coverage - (1.0 - alpha)) ** 2, smooth_coverage


def score_space_miss_softplus(scores: torch.Tensor, tau, kappa: float = 10.0) -> dict:
    """softplus(kappa (s - tau)) / kappa, averaged over the batch."""
    tau_t = tau if torch.is_tensor(tau) else torch.as_tensor(
        float(tau), dtype=scores.dtype, device=scores.device)
    miss_i = torch.nn.functional.softplus(kappa * (scores - tau_t)) / kappa
    inside = scores <= _as_float(tau)
    return {"miss": miss_i.mean(),
            "n_outside": int((~inside).sum()),
            "zero_gradient_fraction": 0.0}


def score_space_miss(scores: torch.Tensor, tau, kappa: float = 10.0,
                     eps: float = SCORE_SPACE_MISS_EPS) -> dict:
    """[ (s - tau) / (stopgrad|tau| + eps) ]_+, averaged over the batch (the paper's hinge)."""
    tau_t = tau if torch.is_tensor(tau) else torch.as_tensor(
        float(tau), dtype=scores.dtype, device=scores.device)
    excess = scores - tau_t
    miss_i = (excess / (tau_t.detach().abs() + eps)).clamp_min(0.0)
    inside = scores <= _as_float(tau)
    q = torch.quantile(scores.detach(),
                       torch.tensor([0.25, 0.5, 0.75], dtype=scores.dtype, device=scores.device))
    return {
        "miss": miss_i.mean(),
        "per_observation": miss_i,
        "mean_score_excess": excess.mean().detach(),
        "mean_positive_score_excess": excess.clamp_min(0.0).mean().detach(),
        "fraction_outside": (~inside).to(scores.dtype).mean().detach(),
        "median_score": q[1].detach(),
        "score_interquantile_range": (q[2] - q[0]).detach(),
        "tau_used": torch.as_tensor(_as_float(tau)),
    }


@dataclass
class ReferenceScale:
    s_ref: float
    m_ref: float
    lambda_ref: float
    miss_type: str
    n_batches: int
    c_ref: float = float("nan")
    lambda_cov_ref: float = float("nan")


def reference_scale(model, x_score_train: np.ndarray, r_score_train: np.ndarray, alpha: float,
                     cfg: LossConfig, n_batches: int = 5, batch_size: int = 256,
                     seed: int = 0, device: str = "cpu") -> ReferenceScale:
    """Reference scales measured once on the initialised model, from D_score-train batches:
    lambda_miss_ref = |S_ref| / M_ref and lambda_cov_ref = |S_ref| / C_ref with S_ref, M_ref,
    C_ref the median size, miss and coverage terms over `n_batches` batches."""
    rng = np.random.default_rng(seed)
    n = len(x_score_train)
    take = min(batch_size, n)
    size_vals, miss_vals, cov_vals = [], [], []
    dtype = next((q.dtype for q in model.parameters()), torch.get_default_dtype())
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(n_batches):
            idx = rng.choice(n, size=take, replace=False)
            xb = torch.as_tensor(x_score_train[idx], dtype=dtype, device=device)
            rb = torch.as_tensor(r_score_train[idx], dtype=dtype,
                                 device=device).reshape(len(idx), -1)
            tau, temperature = _resolve_tau_and_temperature(model, xb, rb, alpha, cfg)
            size_vals.append(float(size_term(model, xb, tau, cfg.size_objective,
                                             getattr(cfg, "size_reference_scale", 1.0))))
            sb = model.score(xb, rb).reshape(-1)
            miss = compute_miss_loss(model=model, x=xb, residual=rb, scores=sb, tau=tau,
                                     miss_type=cfg.miss_type, config=cfg)
            miss_vals.append(float(miss["miss"]))
            cov_term, _ = coverage_loss(sb, tau, temperature, alpha)
            cov_vals.append(float(cov_term))
    if was_training:
        model.train()

    s_ref = float(np.median(size_vals))
    m_ref = float(np.median(miss_vals))
    size_scale = abs(s_ref)
    if cfg.size_objective in LOG_SIZE_OBJECTIVES:
        size_scale = max(size_scale, 1.0)

    # At a fixed threshold no point may sit outside the initial set (M_ref = 0); measure the
    # miss at the empirical (1 - alpha) quantile instead so the scale stays finite.
    m_ref_degenerate = m_ref <= REFERENCE_EPSILON
    if m_ref_degenerate:
        fallback = []
        with torch.no_grad():
            for _ in range(n_batches):
                idx = rng.choice(n, size=take, replace=False)
                xb = torch.as_tensor(x_score_train[idx], dtype=dtype, device=device)
                rb = torch.as_tensor(r_score_train[idx], dtype=dtype,
                                     device=device).reshape(len(idx), -1)
                sb = model.score(xb, rb).reshape(-1)
                tau_q = float(torch.quantile(sb, 1.0 - alpha))
                fallback.append(float(compute_miss_loss(
                    model=model, x=xb, residual=rb, scores=sb, tau=tau_q,
                    miss_type=cfg.miss_type, config=cfg)["miss"]))
        m_ref_fb = float(np.median(fallback))
        if m_ref_fb > REFERENCE_EPSILON:
            logger.warning("reference_scale: m_ref was 0 at the fixed threshold (no points "
                           "outside at init); using the (1-alpha)-quantile probe instead, "
                           "m_ref=%.6g", m_ref_fb)
            m_ref = m_ref_fb
        else:
            logger.warning("reference_scale: m_ref is 0 even at the (1-alpha) quantile; "
                           "lambda_miss will be effectively unbounded for this cell")
    lambda_ref = size_scale / max(m_ref, REFERENCE_EPSILON)
    c_ref = float(np.median(cov_vals))
    lambda_cov_ref = size_scale / max(c_ref, REFERENCE_EPSILON)
    return ReferenceScale(s_ref=s_ref, m_ref=m_ref, lambda_ref=lambda_ref,
                          miss_type=cfg.miss_type, n_batches=n_batches,
                          c_ref=c_ref, lambda_cov_ref=lambda_cov_ref)
