"""Test-set volume of an ICNN sublevel set, with a convergence check.

    V(x) = V_d * E_u[ rho(x, u)^d ],   u uniform on the sphere

rho^d is very concentrated in high dimension, so the directions are drawn whitened by a per-point
matrix Sigma that is refined over a few rounds from the estimator's own weights (unbiased for any
fixed unit-determinant Sigma). The budgets M and 2M come from one nested Sobol pass, so V(2M) and
the doubling check |V(2M)^(1/d) / V(M)^(1/d) - 1| cost a single pass.
"""

import math

import numpy as np
import torch


def _unit_det(sig):
    """Scale every (d, d) matrix of the batch to determinant 1."""
    d = sig.shape[-1]
    sign, logabsdet = torch.linalg.slogdet(sig)
    bad = (sign <= 0) | ~torch.isfinite(logabsdet)
    if bad.any():
        eye = torch.eye(d, dtype=sig.dtype, device=sig.device)
        sig = torch.where(bad[:, None, None], eye.expand_as(sig), sig)
        logabsdet = torch.where(bad, torch.zeros_like(logabsdet), logabsdet)
    return sig / torch.exp(logabsdet / d)[:, None, None]


def _half(sig, shrink):
    d = sig.shape[-1]
    if shrink > 0.0:
        iso = torch.diagonal(sig, dim1=1, dim2=2).sum(1) / d
        eye = torch.eye(d, dtype=sig.dtype, device=sig.device)
        sig = (1.0 - shrink) * sig + shrink * iso[:, None, None] * eye
    ev, q = torch.linalg.eigh(sig)
    ev = ev.clamp_min(1e-30)
    ev = ev / torch.exp(torch.log(ev).mean(dim=1, keepdim=True))
    return (q * ev.sqrt().unsqueeze(1)) @ q.transpose(1, 2)


def _sobol_blocks(d, seed, block, n_total, dtype, device):
    """Consecutive blocks of one scrambled-Sobol sphere sequence from a single engine."""
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
def refine_sigma(model, x, tau, point_seeds, rounds, dirs, shrink=0.05):
    """Per-point preconditioner Sigma (batch, d, d) with unit determinant. Round 0 draws
    isotropically, every later round under the previous Sigma; one RNG seed per point."""
    d = model.residual_dim
    batch = x.shape[0]
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
        v = torch.einsum("bmd,bde->bme", u, _half(sig, shrink))
        nv = v.norm(dim=2).clamp_min(1e-30)
        w = (v / nv.unsqueeze(2)).to(x.dtype)
        rho = model._solve_rays(x.repeat_interleave(dirs, dim=0), w.reshape(-1, d), tau).view(batch, dirs).to(dt)
        pts = (rho / nv).unsqueeze(2) * v
        wt = torch.softmax(d * torch.log((rho / nv).clamp_min(1e-300)), dim=1)
        sig = _unit_det(torch.einsum("bm,bmd,bme->bde", wt, pts, pts)
                        + 1e-12 * torch.eye(d, dtype=dt, device=dev))
    return sig


@torch.no_grad()
def log_volume_nested(model, x, tau, budgets, sigma, block=1 << 13):
    """log V(x; tau) at every budget in `budgets` under a fixed per-point Sigma, one pass."""
    d = model.residual_dim
    batch = x.shape[0]
    dev, dt = x.device, torch.float64
    want = sorted({int(m) for m in budgets})
    block = min(int(block), want[0])
    shalf = _half(sigma.to(dt), 0.0)
    log_v_d = (d / 2.0) * math.log(math.pi) - math.lgamma(d / 2.0 + 1.0)
    lse = torch.full((batch,), -float("inf"), dtype=dt, device=dev)
    out = {}
    for done, m, u in _sobol_blocks(d, int(model.eval_qmc_seed), block, want[-1], dt, dev):
        v = torch.einsum("md,bde->bme", u, shalf)
        nv = v.norm(dim=2).clamp_min(1e-30)
        w = (v / nv.unsqueeze(2)).to(x.dtype)
        rho = model._solve_rays(x.repeat_interleave(m, dim=0), w.reshape(-1, d), tau).view(batch, m).to(dt)
        lse = torch.logaddexp(lse, torch.logsumexp(d * torch.log((rho / nv).clamp_min(1e-300)), dim=1))
        if done + m in want:
            out[done + m] = log_v_d + lse - math.log(done + m)
    return out


@torch.no_grad()
def log_volume_converged(model, x, tau, budgets, point_seeds, rounds=24, refine_dirs=2048):
    sig = refine_sigma(model, x, tau, point_seeds, rounds=rounds, dirs=refine_dirs)
    return log_volume_nested(model, x, tau, budgets, sig)


# ------------------------------------------------------------------------------------------------
# The same estimator for any region given by a gauge function (used for every method but the ICNN,
# whose test volume above is part of its own evaluation).
# ------------------------------------------------------------------------------------------------
GROUP = 30          # test points per estimator call, in test-row order


class Gauge:
    """A region {r : score(x, r) <= tau} seen through the ICNN's ray solver. `score(x_rows, r)`
    returns one value per row; x_rows are whatever `rows` holds (inputs or test-row indices)."""

    def __init__(self, score, d, set_boundary):
        from coconet.models.icnn import ICNN
        self._solve_rays = ICNN._solve_rays.__get__(self)
        self._newton_step = ICNN._newton_step.__get__(self)
        self.score, self.residual_dim, self.eval_qmc_seed = score, int(d), 0
        sb = set_boundary
        self.ray_initial_radius = float(sb["initial_radius"])
        self.ray_growth = float(sb["bracket_growth"])
        self.ray_max_radius = float(sb["max_radius"])
        self.ray_tolerance = float(sb["tolerance"])
        self.ray_bisection_steps = int(sb["bisection_steps"])

    def _tangent(self, x_rows):
        return None, x_rows, None

    def _score_core(self, r, h, G0, x_rows):
        return self.score(x_rows, r).reshape(-1, 1)


def estimate_log_volumes(gauge, rows, tau, budgets, rounds=24, refine_dirs=2048):
    """Per-point log volume at every budget: {budget: array (n,)} in standardized units."""
    n = len(rows)
    out = {b: [] for b in budgets}
    for i in range(0, n, GROUP):
        idx = list(range(i, min(i + GROUP, n)))
        lv = log_volume_converged(gauge, rows[i:i + len(idx)], tau, budgets, point_seeds=idx,
                                  rounds=rounds, refine_dirs=refine_dirs)
        for b in budgets:
            out[b].append(lv[b].reshape(-1).cpu().numpy())
    return {b: np.concatenate(v) for b, v in out.items()}


def pick_budget(mean_v, d, reported=1 << 15, tolerance_pct=0.1):
    """The reported budget if its doubling changed (mean V)^(1/d) by less than the tolerance, else the
    first later doubling that did. Returns (budget, converged)."""
    change = lambda b: abs((mean_v[b] / mean_v[b // 2]) ** (1.0 / d) - 1.0) * 100
    if reported in mean_v and reported // 2 in mean_v and change(reported) < tolerance_pct:
        return reported, True
    for b in sorted(mean_v):
        if b > reported and b // 2 in mean_v and change(b) < tolerance_pct:
            return b, True
    return max(mean_v), False
