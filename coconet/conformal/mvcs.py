"""Minimum-Volume Covering Sets (Braun, Aolaritei, Jordan & Bach, 2025), the "MVCS_conv"
baseline: a locally adaptive p-norm ball around the frozen point prediction,

    C(x; q) = { y : || Lambda(x) (y - F(x)) ||_p <= q },   Lambda(x) = A(x) A(x)^T + ridge I,

with A(x) from an MLP and one global learned p. The paper's method allows any p > 0; this
paper restricts p >= 1 (projected after every optimizer step) so the set stays convex,
which is what the "conv" subscript means. The centre F is the shared frozen predictor, so the
model consumes residuals r = y - F(x) directly.

Training objective (paper eq. 24) on a batch of B residuals:

    L = log( sum_i 1 / det Lambda(x_i) ) + d log sigma_alpha{ s_i } + log V_{d,p}

where sigma_alpha is the batch order statistic of the scores and V_{d,p} the unit p-ball
volume. Norms are evaluated in log space so any p > 0 is numerically safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

_EPS = 1e-12

PAPER_LITERAL = "paper_literal"
TARGET_COVERAGE = "target_coverage"
CONVEX_P_MIN = 1.0


def unit_ball_log_volume(p: torch.Tensor, k: int) -> torch.Tensor:
    """log V_{k,p} = k log 2 + k lgamma(1 + 1/p) - lgamma(1 + k/p)   (paper eq. 13)."""
    if not torch.is_tensor(p):
        p = torch.as_tensor(float(p))
    return k * math.log(2.0) + k * torch.lgamma(1.0 + 1.0 / p) - torch.lgamma(1.0 + k / p)


def p_norm(v: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """||v||_p along the last axis in log space; an all-zero row returns exactly 0."""
    log_abs = torch.log(v.abs().clamp_min(_EPS))
    result = torch.exp(torch.logsumexp(p * log_abs, dim=-1) / p)
    is_zero_row = (v == 0).all(dim=-1)
    return torch.where(is_zero_row, torch.zeros_like(result), result)


def order_statistic(
    scores: torch.Tensor, alpha: float, convention: str = PAPER_LITERAL,
) -> tuple[torch.Tensor, int, float]:
    """Batch threshold sigma_alpha, its rank from the largest, and its empirical inclusion rate.

    paper_literal    the r-th largest score, r = floor(alpha B)  (the paper's text)
    target_coverage  the j-th smallest score, j = ceil((1 - alpha) B)  (the authors' code)
    """
    b = int(scores.shape[0])
    if b == 0:
        raise ValueError("order_statistic requires a non-empty score vector")
    if convention == PAPER_LITERAL:
        r = int(min(max(math.floor(alpha * b), 1), b))
        threshold = torch.topk(scores, r, largest=True).values[-1]
        rank_from_largest = r
    elif convention == TARGET_COVERAGE:
        j = int(min(max(math.ceil((1.0 - alpha) * b), 1), b))
        threshold = torch.kthvalue(scores, j).values
        rank_from_largest = b - j + 1
    else:
        raise ValueError(
            f"order_statistic_convention must be {PAPER_LITERAL!r} or {TARGET_COVERAGE!r}, "
            f"got {convention!r}"
        )
    empirical = float((scores <= threshold).float().mean())
    return threshold, rank_from_largest, empirical


@dataclass
class MvcsLossParts:
    total_loss: torch.Tensor
    volume_term: torch.Tensor
    order_statistic_term: torch.Tensor
    unit_ball_term: torch.Tensor
    training_threshold: float
    training_empirical_coverage: float
    order_statistic_rank: int
    order_statistic_convention: str


def _mlp(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(prev, hidden_dim), nn.ReLU()]
        prev = hidden_dim
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class MvcsSingleNormModel(nn.Module):
    """Matrix net A(x), one global p (a Parameter, projected onto p >= 1), context
    normalization buffers. The centre is external (residual inputs)."""

    def __init__(
        self,
        context_dim: int,
        output_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 3,
        ridge: float = 1e-6,
        p_init: float = 2.0,
        p_min: float = 0.1,
    ) -> None:
        super().__init__()
        if p_min <= 0:
            raise ValueError(f"p_min must be positive, got {p_min}")
        if p_init < CONVEX_P_MIN:
            raise ValueError(f"convex MVCS needs p_init >= 1, got {p_init}")

        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.ridge = float(ridge)
        self.learn_p = True
        self.learn_mu = False
        self.p_min = float(p_min)
        self.p_parameterization = "absolute"
        self.convex_only = True
        self.convex_parameterization = "projected"
        self.p_projection_count = 0
        self.p_step_count = 0
        self._min_preprojection_p = float("inf")
        self._violation_sum = 0.0
        self._violation_max = 0.0
        self._preprojection_p_sum_when_projected = 0.0
        self.center_mode = "fixed_center"

        self.matrix_net = _mlp(context_dim, output_dim * output_dim, hidden_dim, n_layers)
        # Zero weights and an identity bias: Lambda(x) = I at initialization.
        head = self.matrix_net[-1]
        nn.init.zeros_(head.weight)
        head.bias.data.copy_(torch.eye(output_dim).reshape(-1))

        self.center_net = None

        # Projected parameterization: p_raw is p itself.
        self.p_raw = nn.Parameter(torch.tensor(float(p_init)))
        self.register_buffer("mu", torch.zeros(output_dim))

        self.register_buffer("x_mean", torch.zeros(1, context_dim))
        self.register_buffer("x_std", torch.ones(1, context_dim))

    # --- p ---------------------------------------------------------------------------------
    def effective_p(self) -> torch.Tensor:
        return self.p_raw

    def project_p(self) -> bool:
        """Clamp p back to p >= 1 right after an optimizer step; returns whether it acted."""
        self.p_step_count += 1
        with torch.no_grad():
            proposed = float(self.p_raw.item())
            self._min_preprojection_p = min(self._min_preprojection_p, proposed)
            violation = max(CONVEX_P_MIN - proposed, 0.0)
            violated = violation > 0.0
            if violated:
                self.p_raw.clamp_(min=CONVEX_P_MIN)
                self.p_projection_count += 1
                self._violation_sum += violation
                self._violation_max = max(self._violation_max, violation)
                self._preprojection_p_sum_when_projected += proposed
        return violated

    def p_projection_fraction(self) -> float:
        return (self.p_projection_count / self.p_step_count) if self.p_step_count else 0.0

    # --- context ---------------------------------------------------------------------------
    def set_context_normalization(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float).reshape(len(x), -1)
        self.x_mean.copy_(torch.as_tensor(x.mean(axis=0, keepdims=True), dtype=self.x_mean.dtype))
        self.x_std.copy_(torch.as_tensor(
            np.maximum(x.std(axis=0, keepdims=True), 1e-8), dtype=self.x_std.dtype))

    def normalize_context(self, xt: torch.Tensor) -> torch.Tensor:
        return (xt - self.x_mean) / self.x_std

    # --- geometry --------------------------------------------------------------------------
    def lambda_matrix(self, xt: torch.Tensor) -> torch.Tensor:
        """Lambda(x) = A(x) A(x)^T + ridge I."""
        a = self.matrix_net(self.normalize_context(xt)).reshape(-1, self.output_dim, self.output_dim)
        eye = torch.eye(self.output_dim, device=a.device, dtype=a.dtype)
        return a @ a.transpose(-1, -2) + self.ridge * eye

    def log_determinant(self, xt: torch.Tensor, strict: bool = True) -> torch.Tensor:
        """log det Lambda(x). A non-positive sign only comes from non-finite A(x); strict=True
        raises (evaluation), strict=False returns -inf so the training loop skips the batch."""
        sign, logdet = torch.linalg.slogdet(self.lambda_matrix(xt))
        if not bool(torch.all(sign > 0)):
            if strict:
                raise RuntimeError(
                    "Lambda(x) has a non-positive determinant; it must be positive definite. "
                    "This indicates non-finite (NaN/Inf) values in A(x) -- ridge cannot fix a "
                    "NaN. If this fires on a FROZEN model, investigate upstream numerical "
                    "stability (input scale, learning rate); it should not happen on a "
                    "successfully-trained checkpoint."
                )
            return torch.full_like(logdet, float("-inf"))
        return logdet

    def score_tensor(self, xt: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """s = ||Lambda(x) (residual - mu)||_p."""
        lam = self.lambda_matrix(xt)
        return p_norm(torch.einsum("nij,nj->ni", lam, residual - self.mu), self.effective_p())

    # --- objective -------------------------------------------------------------------------
    def loss(
        self, xt: torch.Tensor, residual: torch.Tensor, alpha: float,
        convention: str = PAPER_LITERAL, average_volume_term: bool = False,
    ) -> MvcsLossParts:
        """Paper eq. (24), component-wise."""
        k = self.output_dim
        logdet = self.log_determinant(xt, strict=False)
        s = p_norm(
            torch.einsum("nij,nj->ni", self.lambda_matrix(xt), residual - self.mu),
            self.effective_p(),
        )
        threshold, rank, empirical = order_statistic(s, alpha, convention)

        volume_term = torch.logsumexp(-logdet, dim=0)
        if average_volume_term:
            volume_term = volume_term - math.log(int(xt.shape[0]))
        order_term = k * torch.log(threshold.clamp_min(_EPS))
        unit_term = unit_ball_log_volume(self.effective_p(), k)

        return MvcsLossParts(
            total_loss=volume_term + order_term + unit_term,
            volume_term=volume_term, order_statistic_term=order_term, unit_ball_term=unit_term,
            training_threshold=float(threshold.detach()),
            training_empirical_coverage=empirical,
            order_statistic_rank=rank, order_statistic_convention=convention,
        )


@dataclass
class MvcsScore:
    """numpy-facing view of a trained MvcsSingleNormModel: score / contains / volume."""

    model: MvcsSingleNormModel
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def _xt(self, x: np.ndarray) -> torch.Tensor:
        arr = np.asarray(x, dtype=float).reshape(len(x), -1)
        if arr.shape[1] != self.model.context_dim:
            raise ValueError(
                f"context has {arr.shape[1]} columns, model expects {self.model.context_dim}")
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    def _rt(self, r: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(r, dtype=float), dtype=torch.float32, device=self.device)

    @property
    def output_dim(self) -> int:
        return self.model.output_dim

    @property
    def p(self) -> float:
        with torch.no_grad():
            return float(self.model.effective_p())

    @property
    def p_raw(self) -> float:
        return float(self.model.p_raw.detach())

    @property
    def mu(self) -> np.ndarray:
        return self.model.mu.detach().cpu().numpy().astype(float)

    def _residual(self, x, y_or_residual, center) -> np.ndarray:
        """center=None: the input already is the residual y - F(x)."""
        arr = np.asarray(y_or_residual, dtype=float)
        if center is None:
            return arr
        return arr - np.asarray(center, dtype=float)

    def score(self, x: np.ndarray, r: np.ndarray, center=None) -> np.ndarray:
        with torch.no_grad():
            s = self.model.score_tensor(self._xt(x), self._rt(self._residual(x, r, center)))
        return s.cpu().numpy().astype(float)

    def contains(self, x: np.ndarray, y_or_residual: np.ndarray, q_hat: float, center=None) -> np.ndarray:
        """Exact membership s(x, y) <= q_hat."""
        return self.score(x, y_or_residual, center=center) <= float(q_hat)

    def get_transformation_matrix(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.model.lambda_matrix(self._xt(x)).cpu().numpy().astype(float)

    def get_log_determinant(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.model.log_determinant(self._xt(x)).cpu().numpy().astype(float)

    def log_volume(self, x: np.ndarray, q_hat: float) -> np.ndarray:
        """log Vol = log V_{k,p} + k log q - log det Lambda(x)   (paper eq. 4)."""
        logdet = self.get_log_determinant(x).astype(np.float64)
        log_unit = float(unit_ball_log_volume(
            torch.tensor(self.p, dtype=torch.float64), self.model.output_dim))
        return log_unit + self.model.output_dim * np.log(max(float(q_hat), _EPS)) - logdet

    def volume(self, x: np.ndarray, q_hat: float) -> np.ndarray:
        return np.exp(self.log_volume(x, q_hat))

    def n_params(self) -> int:
        return sum(int(t.numel()) for t in self.model.parameters())

    def eval(self):
        self.model.eval()
        return self
