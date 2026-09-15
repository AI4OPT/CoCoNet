"""Empirical-copula conformal prediction (Messoudi, Destercke & Rousseau, 2021), the
"Emp. Cop." baseline.

An axis-aligned hyper-rectangle centred on the point prediction with half-widths
q_j * sigma_j(x). sigma_j(x) = exp(mu_j(x)) + beta comes from an MLP fit on D_score to
log|r_j|; the per-target thresholds q_j are read off the empirical copula of the normalized
calibration scores at one common marginal level u chosen so that the joint calibration
coverage reaches 1 - alpha.

    D_score / D_validation  fit and early-stop the normalizer mu(x)
    D_cal                   normalized score matrix A, pseudo-observations U, u and q_j
    D_test                  evaluation only
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from coconet.training.train_point import PointTrainConfig, train_point_predictor
from coconet.utils import get_logger

logger = get_logger(__name__)

_MU_CLAMP_MAX = 30.0  # exp(30) ~ 1e13: keeps a pathological normalizer output finite


class EmpiricalCopulaError(RuntimeError):
    pass


@dataclass
class EmpiricalCopulaConfig:
    beta: float = 0.1
    log_epsilon: float = 1e-8
    normalized_scores: bool = True
    normalizer_hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    normalizer_epochs: int = 300
    normalizer_batch_size: int = 256
    normalizer_learning_rate: float = 1e-3
    normalizer_weight_decay: float = 1e-4
    normalizer_patience: int = 25
    seed: int = 0


def _empirical_cdf(sorted_values: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Fhat(a) = |{v <= a}| / n, max-rank convention for ties."""
    n = len(sorted_values)
    counts = np.searchsorted(sorted_values, queries, side="right")
    return counts.astype(np.float64) / n


def _empirical_inverse_cdf(sorted_values: np.ndarray, u: float) -> float:
    """Fhat^{-1}(u) = the order statistic at index ceil(n u) - 1 (no interpolation)."""
    n = len(sorted_values)
    index = int(math.ceil(n * u)) - 1
    index = max(0, min(index, n - 1))
    return float(sorted_values[index])


class EmpiricalCopulaScore:
    """fit_normalizer(D_score, D_val) -> calibrate(D_cal, alpha) -> contains / volume on D_test."""

    def __init__(self, output_dim: int, config: EmpiricalCopulaConfig | None = None) -> None:
        if output_dim < 2:
            raise EmpiricalCopulaError(
                f"EmpiricalCopulaScore is a multivariate baseline requiring output_dim >= 2, got {output_dim}"
            )
        self.output_dim = int(output_dim)
        self.config = config or EmpiricalCopulaConfig()
        self._normalizer: torch.nn.Module | None = None
        self._normalizer_fitted = False
        self._calibrated = False
        self.sorted_calibration_scores: np.ndarray | None = None  # (n_cal, m), each column sorted
        self.n_cal: int | None = None
        self.alpha: float | None = None
        self.u: float | None = None
        self.q: np.ndarray | None = None  # (m,) per-target thresholds
        self.calibration_joint_probability: float | None = None

    # -- normalizer mu(x) ~ log|r| ---------------------------------------------------------
    def fit_normalizer(self, x_score: np.ndarray, r_score: np.ndarray,
                        x_val: np.ndarray, r_val: np.ndarray) -> "EmpiricalCopulaScore":
        cfg = self.config
        if not cfg.normalized_scores:
            self._normalizer_fitted = True
            return self

        z_score = np.log(np.abs(np.asarray(r_score, dtype=np.float64)) + cfg.log_epsilon)
        z_val = np.log(np.abs(np.asarray(r_val, dtype=np.float64)) + cfg.log_epsilon)
        point_cfg = PointTrainConfig(
            hidden_dims=list(cfg.normalizer_hidden_dims), batch_size=cfg.normalizer_batch_size,
            learning_rate=cfg.normalizer_learning_rate, weight_decay=cfg.normalizer_weight_decay,
            max_epochs=cfg.normalizer_epochs, patience=cfg.normalizer_patience, seed=cfg.seed,
            output_dim=self.output_dim,
        )
        result = train_point_predictor(
            np.asarray(x_score, dtype=np.float32), z_score.astype(np.float32),
            np.asarray(x_val, dtype=np.float32), z_val.astype(np.float32),
            point_cfg, device="cpu",
        )
        result.model.eval()
        self._normalizer = result.model
        self._normalizer_fitted = True
        return self

    def _mu(self, x: np.ndarray) -> np.ndarray:
        if self._normalizer is None:
            return np.zeros((len(x), self.output_dim), dtype=np.float64)
        with torch.no_grad():
            xt = torch.as_tensor(np.asarray(x, dtype=np.float32))
            mu = self._normalizer(xt).numpy().astype(np.float64)
        return mu

    def sigma(self, x: np.ndarray) -> np.ndarray:
        """(n, m) positive scales exp(mu(x)) + beta (all ones when unnormalized)."""
        if not self.config.normalized_scores:
            return np.ones((len(np.asarray(x)), self.output_dim), dtype=np.float64)
        if not self._normalizer_fitted:
            raise EmpiricalCopulaError("sigma() called before fit_normalizer()")
        mu = self._mu(x)
        if not np.all(np.isfinite(mu)):
            raise EmpiricalCopulaError("Normalizer produced non-finite mu(x); cannot compute sigma")
        mu = np.clip(mu, -_MU_CLAMP_MAX, _MU_CLAMP_MAX)
        sigma = np.exp(mu) + self.config.beta
        return sigma

    # -- calibration -------------------------------------------------------------------------
    def calibrate(self, x_cal: np.ndarray, r_cal: np.ndarray, alpha: float) -> "EmpiricalCopulaScore":
        if not (0.0 < alpha < 1.0):
            raise EmpiricalCopulaError(f"alpha must be in (0, 1), got {alpha}")
        if not self._normalizer_fitted:
            raise EmpiricalCopulaError("calibrate() called before fit_normalizer()")

        r_cal = np.asarray(r_cal, dtype=np.float64)
        sigma_cal = self.sigma(x_cal)
        A = np.abs(r_cal) / sigma_cal
        if A.ndim != 2 or A.shape[1] != self.output_dim:
            raise EmpiricalCopulaError(f"calibration score matrix must be [n, {self.output_dim}], got {A.shape}")
        if not np.all(np.isfinite(A)):
            raise EmpiricalCopulaError("Calibration score matrix A contains non-finite values")
        if np.any(A < 0):
            raise EmpiricalCopulaError("Calibration score matrix A contains negative values")

        n_cal = A.shape[0]
        sorted_cols = np.sort(A, axis=0)

        # Pseudo-observations U_ij = Fhat_j(A_ij).
        U = np.stack([
            _empirical_cdf(sorted_cols[:, j], A[:, j]) for j in range(self.output_dim)
        ], axis=1)

        # Diagonal empirical copula C_E(u, ..., u) = mean(max_j U_ij <= u); the smallest u with
        # C_E >= 1 - alpha is the order statistic k = ceil(n_cal (1 - alpha)) of M = max_j U_ij.
        M = np.max(U, axis=1)
        sorted_M = np.sort(M)
        k = int(math.ceil(n_cal * (1.0 - alpha)))
        k = max(1, min(k, n_cal))
        u = float(sorted_M[k - 1])

        empirical_joint_probability = float(np.mean(M <= u))
        if empirical_joint_probability < 1.0 - alpha - 1e-9:
            raise EmpiricalCopulaError(
                f"Selected u={u} gives empirical joint probability {empirical_joint_probability} "
                f"< target {1.0 - alpha} -- order-statistic selection is broken"
            )

        q = np.array([_empirical_inverse_cdf(sorted_cols[:, j], u) for j in range(self.output_dim)])

        self.sorted_calibration_scores = sorted_cols
        self.n_cal = n_cal
        self.alpha = float(alpha)
        self.u = u
        self.q = q
        self.calibration_joint_probability = empirical_joint_probability
        self._calibrated = True
        logger.info(
            "empirical_copula calibrated: n_cal=%d alpha=%.4f u=%.6f achieved_joint_prob=%.6f q=%s",
            n_cal, alpha, u, empirical_joint_probability, np.round(q, 4).tolist(),
        )
        return self

    # -- prediction --------------------------------------------------------------------------
    def _require_calibrated(self) -> None:
        if not self._calibrated:
            raise EmpiricalCopulaError("Method has not been calibrated yet (call calibrate() first)")

    def predict_bounds(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Residual-space bounds (lower, upper), each (n, m), around r = 0."""
        self._require_calibrated()
        sigma = self.sigma(x)
        half_widths = self.q[None, :] * sigma
        return -half_widths, half_widths

    def contains(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        """Joint membership: every coordinate inside its own interval."""
        lower, upper = self.predict_bounds(x)
        r = np.asarray(r, dtype=np.float64)
        return np.all((r >= lower) & (r <= upper), axis=1)

    def volume(self, x: np.ndarray) -> np.ndarray:
        """prod_j (2 q_j sigma_j(x))."""
        lower, upper = self.predict_bounds(x)
        return np.prod(upper - lower, axis=1)

    def log_volume(self, x: np.ndarray) -> np.ndarray:
        lower, upper = self.predict_bounds(x)
        return np.sum(np.log(upper - lower), axis=1)

    def parameter_count(self) -> int:
        if self._normalizer is None:
            return 0
        return int(sum(p.numel() for p in self._normalizer.parameters()))

    def state_dict_summary(self) -> dict[str, Any]:
        return {
            "copula_type": "empirical",
            "copula_common_u": self.u,
            "copula_target_thresholds": None if self.q is None else self.q.tolist(),
            "copula_calibration_joint_probability": self.calibration_joint_probability,
            "copula_beta": self.config.beta,
            "copula_normalized_scores": self.config.normalized_scores,
            "calibration_size": self.n_cal,
            "output_dim": self.output_dim,
            "alpha": self.alpha,
        }
