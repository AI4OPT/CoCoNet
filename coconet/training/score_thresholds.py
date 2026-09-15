"""Training threshold tau and coverage-sigmoid temperature for the score-learning loss.

Every loss term is evaluated at one threshold tau (fixed at 1 in the paper, or the batch's
(1 - alpha) score quantile) with smooth coverage mean(sigmoid((tau - s) / T)), T = 1 / kappa_cov.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

TEMPERATURE_FLOOR = 1e-12
SATURATION_LIMIT = 40.0
PINNED_TOL = 1e-6
TRAIN_TAU_MODES = ("quantile", "fixed")


class ScoreThresholdError(ValueError):
    pass


@dataclass(frozen=True)
class ThresholdDiagnostics:
    """Numerical picture of the coverage term for one loss evaluation."""

    tau_used: float
    temperature_used: float
    score_min: float
    score_median: float
    score_mean: float
    score_max: float
    score_std: float
    score_iqr: float
    sigmoid_argument_min: float
    sigmoid_argument_median: float
    sigmoid_argument_max: float
    smooth_coverage: float
    hard_coverage_at_tau: float

    @property
    def saturated(self) -> bool:
        """Smooth coverage numerically pinned at 0 or 1 (no usable gradient)."""
        return self.smooth_coverage < PINNED_TOL or self.smooth_coverage > 1.0 - PINNED_TOL


def _flat(scores: torch.Tensor) -> torch.Tensor:
    if scores.numel() == 0:
        raise ScoreThresholdError("Cannot resolve a threshold from zero scores")
    return scores.reshape(-1)


def compute_training_tau(
    scores: torch.Tensor, alpha: float, mode: str = "quantile", fixed_tau: float = 1.0,
    temperature: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """One scalar threshold for the batch: the detached (1 - alpha) quantile, or `fixed_tau`."""
    flat = _flat(scores)
    if mode == "quantile":
        return torch.quantile(flat.detach(), q=1.0 - alpha).detach()
    if mode == "fixed":
        return torch.as_tensor(float(fixed_tau), device=flat.device, dtype=flat.dtype)
    raise ScoreThresholdError(f"Unknown train_tau_mode: {mode}")


def resolve_temperature(
    scores: torch.Tensor, mode: str, kappa_cov: float, multiplier: float = 1.0,
) -> torch.Tensor:
    """Sigmoid temperature in score units; mode "fixed" gives 1 / kappa_cov."""
    flat = _flat(scores)
    if mode == "fixed":
        if kappa_cov <= 0:
            raise ScoreThresholdError(f"kappa_cov must be positive, got {kappa_cov}")
        return torch.as_tensor(1.0 / kappa_cov, dtype=flat.dtype, device=flat.device)
    raise ScoreThresholdError(f"coverage_temperature_mode must be 'fixed', got {mode!r}")


def score_statistics(scores: torch.Tensor) -> dict[str, float]:
    d = _flat(scores).detach()
    q = torch.quantile(d, torch.tensor([0.25, 0.75], dtype=d.dtype, device=d.device))
    return {
        "score_min": float(d.min()), "score_median": float(d.median()),
        "score_mean": float(d.mean()), "score_max": float(d.max()),
        "score_std": float(d.std(unbiased=False)), "score_iqr": float(q[1] - q[0]),
    }


def threshold_diagnostics(
    scores: torch.Tensor, tau: torch.Tensor, temperature: torch.Tensor,
) -> ThresholdDiagnostics:
    d = _flat(scores).detach()
    tau_d, t_d = tau.detach(), temperature.detach()
    z = (tau_d - d) / t_d
    stats = score_statistics(d)
    return ThresholdDiagnostics(
        tau_used=float(tau_d), temperature_used=float(t_d),
        sigmoid_argument_min=float(z.min()), sigmoid_argument_median=float(z.median()),
        sigmoid_argument_max=float(z.max()),
        smooth_coverage=float(torch.sigmoid(z).mean()),
        hard_coverage_at_tau=float((d <= tau_d).to(d.dtype).mean()),
        **stats,
    )
