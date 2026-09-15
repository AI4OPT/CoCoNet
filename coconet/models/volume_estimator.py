"""Converged test-set volume of an ICNN sublevel set by iteratively refined whitening.

    V(x) = V_d * E_{u ~ S^{d-1}}[ rho(x, u)^d ]

where rho(x, u) is the radial distance to the set boundary along direction u. rho^d is highly
concentrated in high dimension, so the directions are drawn whitened by a per-point matrix
Sigma that is itself refined over a few rounds from the estimator's own weights. For any fixed
SPD Sigma with unit determinant the estimate is unbiased:

    rho_z(u) = rho_r(w) / ||Sigma^{1/2} u||,   w = Sigma^{1/2} u / ||Sigma^{1/2} u||
    log V    = log V_d + logsumexp(d log rho_z) - log M

Sigma is seeded per test point, and the budgets M and 2M come from one nested Sobol pass, so
the reported V(2M) and its convergence certificate |V(2M)^(1/d) / V(M)^(1/d) - 1| cost one pass.
"""
from __future__ import annotations

import math

import torch

__all__ = ["sobol_unit_blocks", "refine_sigma_seeded", "log_volume_whitened_nested",
           "log_volume_converged_nested"]


def _unit_det(sig: torch.Tensor) -> torch.Tensor:
    """Scale each (d, d) matrix in a batch to determinant 1."""
    d = sig.shape[-1]
    sign, logabsdet = torch.linalg.slogdet(sig)
    bad = (sign <= 0) | ~torch.isfinite(logabsdet)
    if bad.any():
        eye = torch.eye(d, dtype=sig.dtype, device=sig.device)
        sig = torch.where(bad[:, None, None], eye.expand_as(sig), sig)
        logabsdet = torch.where(bad, torch.zeros_like(logabsdet), logabsdet)
    return sig / torch.exp(logabsdet / d)[:, None, None]


def _half(sig: torch.Tensor, shrink: float) -> torch.Tensor:
    d = sig.shape[-1]
    if shrink > 0.0:
        iso = torch.diagonal(sig, dim1=1, dim2=2).sum(1) / d
        eye = torch.eye(d, dtype=sig.dtype, device=sig.device)
        sig = (1.0 - shrink) * sig + shrink * iso[:, None, None] * eye
    ev, q = torch.linalg.eigh(sig)
    ev = ev.clamp_min(1e-30)
    ev = ev / torch.exp(torch.log(ev).mean(dim=1, keepdim=True))
    return (q * ev.sqrt().unsqueeze(1)) @ q.transpose(1, 2)


def sobol_unit_blocks(d: int, seed: int, block: int, n_total: int, dtype, device):
    """Consecutive blocks of one scrambled-Sobol sphere sequence (the same sequence as
    `_sample_qmc_sphere_directions(d, n_total, seed)`), from one persistent engine."""
    engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=int(seed))
    eps = torch.finfo(torch.float64).eps
    done = 0
    while done < n_total:
        m = min(block, n_total - done)
        z = engine.draw(m).to(torch.float64).clamp(eps, 1.0 - eps)
        g = torch.erfinv(2.0 * z - 1.0) * math.sqrt(2.0)
        yield done, m, (g / g.norm(dim=1, keepdim=True).clamp_min(1e-300)).to(dtype=dtype, device=device)
        done += m


@torch.no_grad()
def refine_sigma_seeded(model, x: torch.Tensor, tau: float, point_seeds, rounds: int = 24,
                        dirs: int = 2048, shrink: float = 0.05) -> torch.Tensor:
    """Per-point preconditioner Sigma (batch, d, d), unit determinant, one RNG seed per point.
    Round 0 draws isotropically; every later round draws under the previous round's Sigma."""
    d = model.residual_dim
    batch = x.shape[0]
    if len(point_seeds) != batch:
        raise ValueError(f"point_seeds has {len(point_seeds)} entries for {batch} points")
    dev, dt = x.device, torch.float64
    sig = torch.eye(d, dtype=dt, device=dev).expand(batch, d, d).clone()
    g = torch.Generator(device="cpu")
    for t in range(rounds):
        us = []
        for sd in point_seeds:
            g.manual_seed(int(sd) * 7919 + t)
            u = torch.randn(dirs, d, generator=g, dtype=dt).to(dev)
            us.append(u / u.norm(dim=1, keepdim=True))
        u = torch.stack(us)                                    # (batch, dirs, d)
        shalf = _half(sig, shrink)
        v = torch.einsum("bmd,bde->bme", u, shalf)
        nv = v.norm(dim=2).clamp_min(1e-30)
        w = (v / nv.unsqueeze(2)).to(x.dtype)
        rho = model._solve_rays(x.repeat_interleave(dirs, dim=0), w.reshape(-1, d),
                                tau).view(batch, dirs).to(dt)
        pts = (rho / nv).unsqueeze(2) * v
        lw = d * torch.log((rho / nv).clamp_min(1e-300))
        wt = torch.softmax(lw, dim=1)
        sig = _unit_det(torch.einsum("bm,bmd,bme->bde", wt, pts, pts)
                        + 1e-12 * torch.eye(d, dtype=dt, device=dev))
    return sig


@torch.no_grad()
def log_volume_whitened_nested(model, x: torch.Tensor, tau: float, m_list, sigma: torch.Tensor,
                               block: int = 1 << 13) -> dict[int, torch.Tensor]:
    """log V(x; tau) at every budget M in m_list, under a fixed per-point Sigma, from one pass."""
    d = model.residual_dim
    batch = x.shape[0]
    dev, dt = x.device, torch.float64
    want = sorted({int(m) for m in m_list})
    block = min(int(block), want[0])
    shalf = _half(sigma.to(dt), 0.0)
    log_v_d = (d / 2.0) * math.log(math.pi) - math.lgamma(d / 2.0 + 1.0)
    lse = torch.full((batch,), -float("inf"), dtype=dt, device=dev)
    out = {}
    for done, m, u in sobol_unit_blocks(d, int(model.eval_qmc_seed), block, want[-1], dt, dev):
        v = torch.einsum("md,bde->bme", u, shalf)
        nv = v.norm(dim=2).clamp_min(1e-30)
        w = (v / nv.unsqueeze(2)).to(x.dtype)
        rho = model._solve_rays(x.repeat_interleave(m, dim=0), w.reshape(-1, d),
                                tau).view(batch, m).to(dt)
        lse = torch.logaddexp(lse, torch.logsumexp(
            d * torch.log((rho / nv).clamp_min(1e-300)), dim=1))
        if done + m in want:
            out[done + m] = log_v_d + lse - math.log(done + m)
    missing = [m for m in want if m not in out]
    if missing:
        raise ValueError(f"budgets {missing} do not fall on block boundaries (block={block})")
    return out


@torch.no_grad()
def log_volume_converged_nested(model, x: torch.Tensor, tau: float, m_list, point_seeds,
                                rounds: int = 24, refine_dirs: int = 2048, shrink: float = 0.05,
                                block: int = 1 << 13) -> dict[int, torch.Tensor]:
    """The test evaluator: per-point-seeded Sigma refinement, then nested budgets."""
    sig = refine_sigma_seeded(model, x, tau, point_seeds, rounds=rounds, dirs=refine_dirs, shrink=shrink)
    return log_volume_whitened_nested(model, x, tau, m_list, sig, block=block)
