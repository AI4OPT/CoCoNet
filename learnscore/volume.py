"""Root-finding and volume/width/area estimators for any conformity score with a `.score(x, r)`
method whose sublevel sets are convex and bounded, but no closed-form inverse -- this is what
the ICNN needs (the asymmetric-linear and spline scores have exact closed forms instead, see
their own model files).

Since there's no closed-form inverse, the boundary of {r : S(x, r) <= tau} is found by
root-finding along rays from the origin: bracket-and-double, then bisect. The returned root is
differentiable via one Newton correction wrapped in a straight-through estimator: the *value*
is the bisection root, the *gradient* is the implicit-function derivative -F(u*)/F'(u*), so
nothing is backpropagated through the bisection loop itself.

By dimension:
  d = 1  bisect along +/-1                  -> width()
  d = 2  equally spaced angular quadrature   -> area()
  d > 2  scrambled-Sobol QMC over the sphere, in log space -> log_volume()
"""

from __future__ import annotations

import math

import torch


class RootFindingError(RuntimeError):
    pass


def qmc_sphere_directions(d: int, n: int, seed: int, dtype, device) -> torch.Tensor:
    """n quasi-uniform directions on S^{d-1} from a scrambled Sobol sequence: draw n Sobol
    points, map them through the inverse normal CDF (giving standard Gaussian vectors), then
    normalize -- a Gaussian's direction is uniform on the sphere by rotational symmetry, and
    the low-discrepancy sequence spreads those directions far more evenly than i.i.d. draws."""
    engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=int(seed))
    z = engine.draw(n).to(torch.float64).clamp(1e-12, 1 - 1e-12)
    g = torch.erfinv(2 * z - 1) * math.sqrt(2.0)
    unit = g / g.norm(dim=1, keepdim=True).clamp_min(1e-300)
    return unit.to(dtype=dtype, device=device)


def solve_rays(model, x: torch.Tensor, directions: torch.Tensor, tau: float,
                initial_radius: float = 1.0, bracket_growth: float = 2.0,
                max_radius: float = 1.0e4, tolerance: float = 1.0e-5,
                bisection_steps: int = 60) -> torch.Tensor:
    """Solve model.score(x_i, rho_i * directions_i) = tau for rho_i >= 0, row-wise."""
    def f(rho: torch.Tensor) -> torch.Tensor:
        return model.score(x, rho * directions) - tau

    n = x.shape[0]
    with torch.no_grad():
        lo = torch.zeros(n, 1, dtype=x.dtype, device=x.device)
        hi = torch.full((n, 1), initial_radius, dtype=x.dtype, device=x.device)
        f_hi = f(hi)
        while True:
            grow = (f_hi < 0) & (hi < max_radius)
            if not torch.any(grow):
                break
            hi = torch.where(grow, torch.clamp(hi * bracket_growth, max=max_radius), hi)
            f_hi = f(hi)
        unbounded = f_hi < 0
        if torch.any(unbounded):
            n_bad = int(unbounded.sum())
            raise RootFindingError(
                f"{n_bad}/{n} rays never bracketed score=tau within max_radius={max_radius}")
        for _ in range(bisection_steps):
            mid = (lo + hi) / 2.0
            go_right = f(mid) < 0
            lo = torch.where(go_right, mid, lo)
            hi = torch.where(go_right, hi, mid)
            if (hi - lo).max().item() < tolerance:
                break
        rho_star = (lo + hi) / 2.0

    # straight-through Newton correction: value = bisection root, gradient = implicit-function
    # derivative. Some scores (e.g. the empirical-copula baseline's rank lookup) aren't
    # differentiable at all -- f_val then carries no grad_fn, so the correction is skipped and
    # the plain bisection value is returned; that value is still correct, just not usable to
    # backprop a training loss through, which those non-trained baselines never need anyway.
    with torch.enable_grad():
        rho_leaf = rho_star.clone().requires_grad_(True)
        f_val = f(rho_leaf)
        if not f_val.requires_grad:
            return rho_star
        (f_grad,) = torch.autograd.grad(f_val.sum(), rho_leaf, create_graph=True)
        denom = torch.where(f_grad.abs() < 1e-8, torch.full_like(f_grad, 1e-8), f_grad)
        correction = -f_val / denom
    return rho_star + (correction - correction.detach())


def width(model, x: torch.Tensor, tau: float, **kw) -> torch.Tensor:
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
    return solve_rays(model, x, ones, tau, **kw) - (-solve_rays(model, x, -ones, tau, **kw))


def interval(model, x: torch.Tensor, tau: float, **kw) -> tuple[torch.Tensor, torch.Tensor]:
    """(lower_offset, upper_offset): interval = y_hat + [lower_offset, upper_offset]."""
    ones = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
    return -solve_rays(model, x, -ones, tau, **kw), solve_rays(model, x, ones, tau, **kw)


def polar_root(model, x: torch.Tensor, tau: float, n_directions: int, **kw):
    """Equally spaced angular directions in the plane, for residual_dim=2."""
    thetas = torch.linspace(0.0, 2 * math.pi, n_directions + 1, dtype=x.dtype, device=x.device)[:-1]
    directions = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=1)
    x_exp = x.repeat_interleave(n_directions, dim=0)
    dir_exp = directions.repeat(x.shape[0], 1)
    rho = solve_rays(model, x_exp, dir_exp, tau, **kw).view(x.shape[0], n_directions)
    return rho, directions


def area(model, x: torch.Tensor, tau: float, n_directions: int, **kw) -> torch.Tensor:
    """Polar-quadrature area estimate: A = (pi/M) * sum_m rho(theta_m)^2."""
    rho, _ = polar_root(model, x, tau, n_directions, **kw)
    return (math.pi / n_directions) * (rho ** 2).sum(dim=1, keepdim=True)


def prediction_set(model, x: torch.Tensor, tau: float, center: torch.Tensor,
                    n_directions: int, **kw) -> torch.Tensor:
    """Boundary vertices of the 2-D sublevel set, shape (batch, M, 2), CCW order."""
    rho, directions = polar_root(model, x, tau, n_directions, **kw)
    return rho.unsqueeze(-1) * directions.unsqueeze(0) + center.unsqueeze(1)


def log_volume(model, x: torch.Tensor, tau, residual_dim: int, n_directions: int, seed: int,
                **kw) -> torch.Tensor:
    """log V(x;tau) = log(vol unit ball) + logsumexp(d * log rho_m) - log(M), the QMC average
    of V_d * rho^d over sphere directions. Computed via logsumexp so rho^d is never formed at
    full scale (it overflows float32 well before d ~ 12)."""
    directions = qmc_sphere_directions(residual_dim, n_directions, seed, x.dtype, x.device)
    x_exp = x.repeat_interleave(n_directions, dim=0)
    dir_exp = directions.repeat(x.shape[0], 1)
    rho = solve_rays(model, x_exp, dir_exp, tau, **kw).view(x.shape[0], n_directions)
    log_rho = torch.log(rho.clamp_min(1e-12))
    d = residual_dim
    log_unit_ball = (d / 2.0) * math.log(math.pi) - math.lgamma(d / 2.0 + 1.0)
    return log_unit_ball + torch.logsumexp(d * log_rho, dim=1, keepdim=True) - math.log(n_directions)


def volume(model, x: torch.Tensor, tau, residual_dim: int, n_directions: int, seed: int,
           **kw) -> torch.Tensor:
    return torch.exp(log_volume(model, x, tau, residual_dim, n_directions, seed, **kw))
