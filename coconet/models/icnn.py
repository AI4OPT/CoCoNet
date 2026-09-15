"""The ICNN conformity score (CoCoNet_ICNN).

    S(x, r) = G(x, r) - G(x, 0) - grad_r G(x, 0)^T r + epsilon ||r||^2

G is an input-convex network in the residual r = y - F(x), conditioned on x through an
encoder h = enc(x):

    z_1 = phi(A_1(x) r + V_1 h + b_1)
    z_k = phi(softplus(W_k) z_{k-1} + A_k(x) r + V_k h + b_k),   k = 2..L
    G   = softplus(w)^T z_L

Hidden-to-hidden and output weights are softplus-reparameterised (nonnegative) and phi is
convex and nondecreasing, so G is convex in r. The tangent correction makes S(x, 0) = 0 with
zero gradient there; the epsilon term makes S strongly convex, so its sublevel sets are compact.

There is no closed-form inverse, so every boundary quantity comes from root finding on
S(x, rho u) = tau:
  d = 1  width/interval        bracket-and-bisect along +-1
  d = 2  area/prediction_set   equally spaced angular quadrature
  d > 2  volume/log_volume     scrambled-Sobol QMC over the sphere, in log space

The root is found by bracket-and-bisect under no_grad and made differentiable by one Newton
correction in a straight-through estimator: the value is the bisection root, the gradient is
the implicit-function derivative -F(u*)/F'(u*).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from coconet.models.param_groups import l2_penalty as _l2_penalty


def _inverse_softplus(y: float) -> float:
    return math.log(math.expm1(y))


DEFAULT_ICNN_DEPTH = 2
DEFAULT_ICNN_HIDDEN_DIM = 32


def resolve_icnn_hidden_dims(spec: int | Sequence[int] | None,
                             depth: int | None = None) -> tuple[int, ...]:
    """Per-layer widths of the convex body: an explicit sequence wins, otherwise [width] * depth."""
    if spec is not None and not isinstance(spec, int):
        dims = tuple(int(h) for h in spec)
        if len(dims) < 1:
            raise ValueError("the ICNN body needs at least 1 hidden layer, got an empty sequence")
        if any(h < 1 for h in dims):
            raise ValueError(f"all ICNN hidden widths must be >= 1, got {list(dims)}")
        if depth is not None and int(depth) != len(dims):
            raise ValueError(
                f"conflicting ICNN architecture: icnn_hidden_dims={list(dims)} has "
                f"{len(dims)} layer(s) but icnn_depth={depth}. Supply one or the other -- an "
                f"explicit sequence already fixes the depth.")
        return dims

    width = DEFAULT_ICNN_HIDDEN_DIM if spec is None else int(spec)
    n = DEFAULT_ICNN_DEPTH if depth is None else int(depth)
    if width < 1:
        raise ValueError(f"icnn_hidden_dim must be >= 1, got {width}")
    if n < 1:
        raise ValueError(f"icnn_depth must be >= 1, got {n}")
    return (width,) * n


_ICNN_BIAS_CONTAINER = "b"


def _icnn_excluded_names(model: nn.Module) -> frozenset[str]:
    """The additive shifts b_k, excluded from the L2 penalty like ordinary biases."""
    return frozenset(
        full for full, _ in model.named_parameters()
        if len(full.split(".")) >= 2 and full.split(".")[-2] == _ICNN_BIAS_CONTAINER)


DEFAULT_TRAIN_VOLUME_DIRECTIONS = 128
DEFAULT_EVAL_VOLUME_DIRECTIONS = 1024


def _sample_qmc_sphere_directions(
    d: int, n_directions: int, seed: int, dtype, device, antithetic: bool = False,
) -> torch.Tensor:
    """n quasi-uniform directions on S^{d-1}: scrambled Sobol points mapped through the inverse
    normal CDF and normalised. Uses its own engine, so the global torch RNG is untouched."""
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    if n_directions < 1:
        raise ValueError(f"n_directions must be >= 1, got {n_directions}")
    n_base = (n_directions + 1) // 2 if antithetic else n_directions
    engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=int(seed))
    z = engine.draw(n_base).to(torch.float64)
    eps = torch.finfo(torch.float64).eps
    z = z.clamp(eps, 1.0 - eps)
    g = torch.erfinv(2.0 * z - 1.0) * math.sqrt(2.0)
    unit = g / g.norm(dim=1, keepdim=True).clamp_min(1e-300)
    if antithetic:
        unit = torch.cat([unit, -unit], dim=0)[:n_directions]
    return unit.to(dtype=dtype, device=device)


class RootFindingError(RuntimeError):
    pass


class ICNN(nn.Module):
    """Conditional input-convex conformity score; see the module docstring.

    Encoder input is x. Subclasses below change only `encode` (and the encoder's input width) to
    feed [x, F(x)], F(x) alone, nothing, or the raw context without an encoder MLP.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: list[int] | None = None,
        icnn_hidden_dim: int | Sequence[int] = DEFAULT_ICNN_HIDDEN_DIM,
        epsilon: float = 1e-3,
        initial_bracket: float = 1.0,
        bracket_growth: float = 2.0,
        max_bracket_expansions: int = 40,
        bisection_tol: float = 1e-6,
        bisection_max_iter: int = 100,
        additive_shift: bool = True,
        residual_dim: int = 1,
        activation: str = "softplus",
        context_dependent_residual_maps: bool = True,
        icnn_hidden_dims: Sequence[int] | None = None,
        icnn_depth: int | None = None,
        train_area_directions: int = 32,
        eval_area_directions: int = 360,
        train_volume_directions: int = DEFAULT_TRAIN_VOLUME_DIRECTIONS,
        eval_volume_directions: int = DEFAULT_EVAL_VOLUME_DIRECTIONS,
        volume_whiten: bool = False,
        whiten_momentum: float = 0.05,
        whiten_shrink: float = 0.05,
        whiten_warmup_steps: int = 20,
        volume_antithetic: bool = False,
        qmc_base_seed: int = 0,
        eval_qmc_seed: int = 0,
        set_boundary_initial_radius: float = 1.0,
        set_boundary_bracket_growth: float = 2.0,
        set_boundary_max_radius: float = 1.0e4,
        set_boundary_tolerance: float = 1.0e-5,
        set_boundary_bisection_steps: int = 60,
        strict_rays: bool = True,
    ) -> None:
        super().__init__()
        if encoder_hidden_dims is None:
            encoder_hidden_dims = [64, 32]
        if residual_dim < 1:
            raise ValueError(f"residual_dim must be >= 1, got {residual_dim!r}")
        if activation not in ("softplus", "relu"):
            raise ValueError(f"activation must be 'softplus' or 'relu', got {activation!r}")

        self.epsilon = epsilon
        self.initial_bracket = initial_bracket
        self.bracket_growth = bracket_growth
        self.max_bracket_expansions = max_bracket_expansions
        self.bisection_tol = bisection_tol
        self.bisection_max_iter = bisection_max_iter
        self.residual_dim = residual_dim
        self.output_dim = residual_dim
        self.set_boundary_initial_radius = set_boundary_initial_radius
        self.set_boundary_bracket_growth = set_boundary_bracket_growth
        self.set_boundary_max_radius = set_boundary_max_radius
        self.set_boundary_tolerance = set_boundary_tolerance
        self.set_boundary_bisection_steps = set_boundary_bisection_steps
        self.strict_rays = bool(strict_rays)
        self._last_ray_failure_mask = None
        self._last_ray_failure_count = 0
        self.train_area_directions = int(train_area_directions)
        self.eval_area_directions = int(eval_area_directions)
        self.train_volume_directions = int(train_volume_directions)
        self.eval_volume_directions = int(eval_volume_directions)
        self.volume_antithetic = bool(volume_antithetic)
        self.volume_whiten = bool(volume_whiten)
        self.whiten_momentum = float(whiten_momentum)
        self.whiten_shrink = float(whiten_shrink)
        self.whiten_warmup_steps = int(whiten_warmup_steps)
        # Whitening and QMC state are plain attributes, not buffers, so they stay out of the
        # state_dict.
        self._whiten_sigma = None
        self._whiten_steps = 0
        self.qmc_base_seed = int(qmc_base_seed)
        self.eval_qmc_seed = int(eval_qmc_seed)
        self.qmc_epoch = 0

        dims = resolve_icnn_hidden_dims(
            icnn_hidden_dims if icnn_hidden_dims is not None else icnn_hidden_dim,
            depth=icnn_depth)
        self.icnn_hidden_dims = list(dims)
        self.n_hidden_layers = len(dims)
        self.icnn_depth = len(dims)
        self.icnn_hidden_dim = dims[0] if len(set(dims)) == 1 else None
        self.activation_name = activation
        self._phi = F.softplus if activation == "softplus" else F.relu
        self.context_dependent_residual_maps = context_dependent_residual_maps
        self.additive_shift = additive_shift

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in encoder_hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.encoder = nn.Sequential(*layers)
        encoder_output_dim = prev_dim
        self.encoder_output_dim = int(encoder_output_dim)

        # Parameter creation order fixes the seeded initialisation; do not reorder.
        if context_dependent_residual_maps:
            self.a_nets = nn.ModuleList(
                [nn.Linear(encoder_output_dim, h * residual_dim) for h in dims])
        else:
            self.a_fixed = nn.ParameterList(
                [nn.Parameter(torch.zeros(residual_dim, h)) for h in dims])
            for p in self.a_fixed:
                nn.init.normal_(p, std=0.1)
        if additive_shift:
            self.V = nn.ModuleList([nn.Linear(encoder_output_dim, h) for h in dims])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(1, h)) for h in dims])
        self.W_raw = nn.ParameterList(
            [nn.Parameter(torch.zeros(dims[k], dims[k - 1])) for k in range(1, len(dims))])
        self.w_raw = nn.Parameter(torch.zeros(dims[-1]))

        for p in self.b:
            nn.init.normal_(p, std=0.1)
        # softplus(W_raw) ~ 1/fan_in keeps every pre-activation O(1) regardless of width.
        for p in self.W_raw:
            nn.init.normal_(p, mean=_inverse_softplus(1.0 / p.shape[1]), std=0.1)
        nn.init.normal_(self.w_raw, std=0.1)

    def architecture(self) -> dict:
        return {
            "icnn_depth": self.icnn_depth,
            "icnn_hidden_dim": self.icnn_hidden_dim,
            "icnn_hidden_dims": list(self.icnn_hidden_dims),
            "icnn_n_parameters": int(sum(p.numel() for p in self.parameters())),
            "icnn_encoder_hidden_dims": [m.out_features for m in self.encoder
                                          if isinstance(m, nn.Linear)],
            "icnn_residual_dim": int(self.residual_dim),
            "icnn_activation": self.activation_name,
            "icnn_additive_shift": bool(self.additive_shift),
            "icnn_context_dependent_residual_maps": bool(self.context_dependent_residual_maps),
        }

    def _residual_projection(self, h: torch.Tensor, r: torch.Tensor, k: int) -> torch.Tensor:
        """A_k(x) r, contracted to (batch, H_k)."""
        if self.context_dependent_residual_maps:
            a = self.a_nets[k](h).view(-1, self.icnn_hidden_dims[k], self.residual_dim)
            return torch.einsum("bhd,bd->bh", a, r)
        return r @ self.a_fixed[k]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _apply_phi(self, u: torch.Tensor) -> torch.Tensor:
        return self._phi(u)

    def _body(self, r: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """G(x, r) given the precomputed context h = encode(x). Returns (batch, 1)."""
        phi = self._apply_phi

        shift = self.b[0] + (self.V[0](h) if self.additive_shift else 0.0)
        z = phi(self._residual_projection(h, r, 0) + shift)

        for k in range(1, self.n_hidden_layers):
            shift = self.b[k] + (self.V[k](h) if self.additive_shift else 0.0)
            W = F.softplus(self.W_raw[k - 1])
            z = phi(z @ W.T + self._residual_projection(h, r, k) + shift)

        w = F.softplus(self.w_raw)
        return z @ w.unsqueeze(-1)

    def _tangent(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(G(x,0), grad_r G(x,0), h): the tangent plane at the origin, reused by root finders
        that evaluate the score many times against the same x."""
        batch = x.shape[0]
        h = self.encode(x)
        r0 = torch.zeros(batch, self.residual_dim, dtype=x.dtype, device=x.device, requires_grad=True)
        with torch.enable_grad():
            G0 = self._body(r0, h)
            (tangent,) = torch.autograd.grad(G0.sum(), r0, create_graph=True)
        return G0, tangent, h

    def _score_core(self, r: torch.Tensor, h: torch.Tensor, G0: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
        G_r = self._body(r, h)
        supporting_plane = (r * tangent).sum(dim=1, keepdim=True)
        quadratic = self.epsilon * (r ** 2).sum(dim=1, keepdim=True)
        return G_r - G0 - supporting_plane + quadratic

    def score(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Nonnegative score, shape (batch, 1). r = y - F(x), shape (batch, residual_dim)."""
        G0, tangent, h = self._tangent(x)
        return self._score_core(r, h, G0, tangent)

    # -- root finding --------------------------------------------------------------------

    def _bracket_and_bisect(
        self, f, batch: int, dtype: torch.dtype, device: torch.device,
        initial_bracket: float | None = None, bracket_growth: float | None = None,
        max_bracket_expansions: int | None = None, bisection_tol: float | None = None,
        bisection_max_iter: int | None = None,
    ) -> torch.Tensor:
        """u* >= 0 with f(u*) = 0 for an increasing f with f(0) <= 0. No grad."""
        initial_bracket = self.initial_bracket if initial_bracket is None else initial_bracket
        bracket_growth = self.bracket_growth if bracket_growth is None else bracket_growth
        max_bracket_expansions = self.max_bracket_expansions if max_bracket_expansions is None else max_bracket_expansions
        bisection_tol = self.bisection_tol if bisection_tol is None else bisection_tol
        bisection_max_iter = self.bisection_max_iter if bisection_max_iter is None else bisection_max_iter
        with torch.no_grad():
            lo = torch.zeros(batch, 1, dtype=dtype, device=device)
            hi = torch.full((batch, 1), initial_bracket, dtype=dtype, device=device)
            f_hi = f(hi)
            for _ in range(max_bracket_expansions):
                need_expand = f_hi < 0
                if not torch.any(need_expand):
                    break
                hi = torch.where(need_expand, hi * bracket_growth, hi)
                f_hi = f(hi)
            if torch.any(f_hi < 0):
                n_bad = int((f_hi < 0).sum().item())
                raise RootFindingError(
                    f"Bracket expansion failed to bound the root for {n_bad} sample(s) after "
                    f"{max_bracket_expansions} expansions (max hi={hi.max().item():.4g}); "
                    "increase icnn_conditional_slope.root_finding.max_bracket_expansions/initial_bracket/bracket_growth."
                )

            for _ in range(bisection_max_iter):
                mid = (lo + hi) / 2.0
                f_mid = f(mid)
                go_right = f_mid < 0
                lo = torch.where(go_right, mid, lo)
                hi = torch.where(go_right, hi, mid)
                if (hi - lo).max().item() < bisection_tol:
                    break
            return (lo + hi) / 2.0

    def _root_one_side(
        self, x: torch.Tensor, tau: float, sign: float, G0: torch.Tensor, tangent: torch.Tensor, h: torch.Tensor,
    ) -> torch.Tensor:
        batch = x.shape[0]

        def f(u: torch.Tensor) -> torch.Tensor:
            return self._score_core(sign * u, h, G0, tangent) - tau

        u_star = self._bracket_and_bisect(f, batch, x.dtype, x.device)

        with torch.enable_grad():
            u_leaf = u_star.clone().requires_grad_(True)
            f_val = f(u_leaf)
            (f_grad_u,) = torch.autograd.grad(f_val.sum(), u_leaf, create_graph=True, retain_graph=True)
            denom = torch.where(f_grad_u.abs() < 1e-8, torch.full_like(f_grad_u, 1e-8), f_grad_u)
            correction = -f_val / denom
        u_hat = u_star + (correction - correction.detach())
        return sign * u_hat

    def width(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        if self.residual_dim != 1:
            raise ValueError("width() is only defined for residual_dim=1; use area()/prediction_set_size() instead")
        G0, tangent, h = self._tangent(x)
        r_pos = self._root_one_side(x, tau, 1.0, G0, tangent, h)
        r_neg = self._root_one_side(x, tau, -1.0, G0, tangent, h)
        return r_pos - r_neg

    def interval(self, x: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
        """(lower_offset, upper_offset): interval = y_hat + [lower_offset, upper_offset]."""
        if self.residual_dim != 1:
            raise ValueError("interval() is only defined for residual_dim=1; use prediction_set() for residual_dim=2")
        if tau == float("inf"):
            inf = torch.full((x.shape[0], 1), float("inf"), dtype=x.dtype, device=x.device)
            return -inf, inf
        G0, tangent, h = self._tangent(x)
        r_pos = self._root_one_side(x, tau, 1.0, G0, tangent, h)
        r_neg = self._root_one_side(x, tau, -1.0, G0, tangent, h)
        return r_neg, r_pos

    def _polar_root(
        self, x: torch.Tensor, tau: float, n_directions: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Boundary radius along M equally spaced angles (residual_dim=2). Returns
        (rho (batch, M), directions (M, 2)), counter-clockwise."""
        if self.residual_dim != 2:
            raise ValueError("_polar_root is only defined for residual_dim=2")
        batch = x.shape[0]
        thetas = torch.linspace(
            0.0, 2.0 * math.pi, n_directions + 1, dtype=x.dtype, device=x.device,
        )[:-1]
        directions = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=1)  # (M, 2)

        x_expanded = x.repeat_interleave(n_directions, dim=0)  # (batch*M, input_dim)
        directions_expanded = directions.repeat(batch, 1)  # (batch*M, 2)
        rho_hat = self._solve_rays(x_expanded, directions_expanded, tau)

        return rho_hat.view(batch, n_directions), directions

    def _solve_rays(
        self, x_rows: torch.Tensor, dir_rows: torch.Tensor, tau: float,
        diagnostics: dict | None = None,
    ) -> torch.Tensor:
        """Solve score(x_i, rho_i d_i) = tau row-wise; returns rho (N, 1), differentiable via the
        straight-through Newton correction. Rays that never bracket tau within
        set_boundary_max_radius raise when strict_rays is set."""
        G0_exp, tangent_exp, h_exp = self._tangent(x_rows)
        directions_expanded = dir_rows

        def f(rho: torch.Tensor) -> torch.Tensor:
            r = rho * directions_expanded
            return self._score_core(r, h_exp, G0_exp, tangent_exp) - tau

        x = x_rows
        n_rows = x_rows.shape[0]
        with torch.no_grad():
            lo = torch.zeros(n_rows, 1, dtype=x.dtype, device=x.device)
            hi = torch.full((n_rows, 1), self.set_boundary_initial_radius, dtype=x.dtype, device=x.device)
            f_hi = f(hi)
            n_expansion_steps = 0
            while True:
                need_expand = (f_hi < 0) & (hi < self.set_boundary_max_radius)
                if not torch.any(need_expand):
                    break
                hi = torch.where(
                    need_expand, torch.clamp(hi * self.set_boundary_bracket_growth, max=self.set_boundary_max_radius), hi,
                )
                f_hi = f(hi)
                n_expansion_steps += 1
            still_unbounded = f_hi < 0
            n_bad = int(still_unbounded.sum().item())
            if n_bad > 0:
                warnings.warn(
                    f"Polar boundary trace: {n_bad}/{n_rows} (sample, direction) pair(s) did not "
                    f"reach the conformal threshold within set_boundary_max_radius="
                    f"{self.set_boundary_max_radius}; the ICNN sublevel set appears numerically "
                    "unbounded in that direction. This should not normally occur when "
                    "epsilon (strong convexity) > 0 -- consider increasing "
                    "set_boundary_max_radius or epsilon. These rays are marked FAILED: their "
                    "radius is NOT a root and carries no root gradient.", RuntimeWarning,
                )
                if self.strict_rays:
                    raise RootFindingError(
                        f"{n_bad}/{n_rows} rays did not bracket score = tau within "
                        f"set_boundary_max_radius={self.set_boundary_max_radius}. A capped "
                        f"radius is NOT a boundary point, so continuing would report a "
                        f"fabricated volume. Set strict_rays=False only for diagnostics.")
                hi = torch.where(still_unbounded, torch.full_like(hi, self.set_boundary_max_radius), hi)

            n_bisection_steps_used = 0
            for _ in range(self.set_boundary_bisection_steps):
                mid = (lo + hi) / 2.0
                f_mid = f(mid)
                go_right = f_mid < 0
                lo = torch.where(go_right, mid, lo)
                hi = torch.where(go_right, hi, mid)
                n_bisection_steps_used += 1
                if (hi - lo).max().item() < self.set_boundary_tolerance:
                    break
            rho_star = (lo + hi) / 2.0

            if diagnostics is not None:
                diagnostics["n_rows"] = n_rows
                diagnostics["n_expansion_steps"] = n_expansion_steps
                diagnostics["n_ray_failures"] = n_bad
                diagnostics["ray_failure_rate"] = n_bad / max(n_rows, 1)
                diagnostics["n_bisection_steps_used"] = n_bisection_steps_used
                diagnostics["max_radius_reached"] = float(hi.max().item())
                diagnostics["nonfinite_count"] = int((~torch.isfinite(rho_star)).sum().item())
                diagnostics["n_successful_rays"] = n_rows - n_bad
                diagnostics["fraction_failed_rays"] = n_bad / max(n_rows, 1)
                diagnostics["n_max_radius_hits"] = n_bad
                diagnostics["fraction_max_radius_hits"] = n_bad / max(n_rows, 1)
                diagnostics["n_max_expansion_hits"] = int(
                    ((hi >= self.set_boundary_max_radius) & still_unbounded).sum().item()
                )
                finite_rho = rho_star[torch.isfinite(rho_star)]
                if finite_rho.numel() > 0:
                    diagnostics["min_solved_radius"] = float(finite_rho.min().item())
                    diagnostics["median_solved_radius"] = float(finite_rho.median().item())
                    diagnostics["mean_solved_radius"] = float(finite_rho.mean().item())
                    diagnostics["max_solved_radius"] = float(finite_rho.max().item())
                else:
                    diagnostics["min_solved_radius"] = float("nan")
                    diagnostics["median_solved_radius"] = float("nan")
                    diagnostics["mean_solved_radius"] = float("nan")
                    diagnostics["max_solved_radius"] = float("nan")

        # Failed rays are not roots, so they get no implicit-function gradient.
        with torch.enable_grad():
            rho_leaf = rho_star.clone().requires_grad_(True)
            f_val = f(rho_leaf)
            (f_grad,) = torch.autograd.grad(f_val.sum(), rho_leaf, create_graph=True, retain_graph=True)
            denom = torch.where(f_grad.abs() < 1e-8, torch.full_like(f_grad, 1e-8), f_grad)
            correction = -f_val / denom
            correction = torch.where(still_unbounded, torch.zeros_like(correction), correction)
        self._last_ray_failure_mask = still_unbounded.detach()
        self._last_ray_failure_count = n_bad
        return rho_star + (correction - correction.detach())

    def ray_root(
        self, x: torch.Tensor, tau: float, directions: torch.Tensor,
        diagnostics: dict | None = None,
    ) -> torch.Tensor:
        """Boundary radius along one unit direction per sample; shape (batch, 1)."""
        expected = (x.shape[0], self.residual_dim)
        if directions.shape != expected:
            raise ValueError(f"directions must have shape {expected}, got {tuple(directions.shape)}")
        with torch.no_grad():
            norms = directions.norm(dim=1)
            if bool((norms < 1e-8).any()):
                raise ValueError("ray_root: directions must be non-zero (found a near-zero-norm row)")
            if not bool(torch.allclose(norms, torch.ones_like(norms), atol=1e-4)):
                raise ValueError(
                    f"ray_root: directions must be unit-norm, got norms in "
                    f"[{norms.min().item():.4g}, {norms.max().item():.4g}]"
                )
        return self._solve_rays(x, directions, tau, diagnostics=diagnostics)

    def area(self, x: torch.Tensor, tau: float, n_directions: int | None = None) -> torch.Tensor:
        """Polar-quadrature area (pi/M) sum_m rho(theta_m)^2 at residual_dim=2, shape (batch, 1)."""
        if self.residual_dim != 2:
            raise ValueError("area() is only defined for residual_dim=2; use width() for residual_dim=1")
        M = self.train_area_directions if n_directions is None else n_directions
        rho, _directions = self._polar_root(x, tau, M)
        return (math.pi / M) * (rho ** 2).sum(dim=1, keepdim=True)

    # -- d > 2 volume -------------------------------------------------------------------

    def set_qmc_epoch(self, epoch: int) -> None:
        """Training scramble index; the training loop sets it once per epoch so successive
        epochs integrate over different Sobol sets. Evaluation always uses eval_qmc_seed."""
        self.qmc_epoch = int(epoch)

    def training_qmc_seed(self) -> int:
        return self.qmc_base_seed + self.qmc_epoch

    def _volume_directions(self, n_directions: int, seed: int, dtype, device) -> torch.Tensor:
        return _sample_qmc_sphere_directions(
            self.residual_dim, n_directions, seed, dtype, device,
            antithetic=self.volume_antithetic)

    def _log_volume_core(self, x: torch.Tensor, tau, n_directions: int | None, seed: int | None,
                         diagnostics: dict | None = None) -> torch.Tensor:
        """log V(x; tau) = log V_d + logsumexp(d log rho_m) - log M over M sphere directions."""
        d = self.residual_dim
        M = self.eval_volume_directions if n_directions is None else n_directions
        seed = self.eval_qmc_seed if seed is None else seed
        directions = self._volume_directions(M, seed, x.dtype, x.device)
        m_actual = directions.shape[0]
        batch = x.shape[0]
        x_expanded = x.repeat_interleave(m_actual, dim=0)
        dir_expanded = directions.repeat(batch, 1)
        rho = self._solve_rays(x_expanded, dir_expanded, tau,
                               diagnostics=diagnostics).view(batch, m_actual)
        if diagnostics is not None:
            diagnostics["residual_dim"] = d
            diagnostics["n_directions"] = m_actual
            diagnostics["qmc_seed"] = int(seed)
        log_rho = torch.log(rho.clamp_min(1e-12))
        log_v_d = (d / 2.0) * math.log(math.pi) - math.lgamma(d / 2.0 + 1.0)
        return log_v_d + torch.logsumexp(d * log_rho, dim=1, keepdim=True) - math.log(m_actual)

    # Whitened size term. With z = Sigma^{-1/2} r, rho_z(u) = rho_r(w) / ||Sigma^{1/2} u|| for
    # w = Sigma^{1/2} u / ||Sigma^{1/2} u||, and
    #     log V_r = log V_d + logsumexp(d log rho_z) - log M      (Sigma has unit determinant)
    # exactly, for any fixed SPD Sigma; Sigma only changes the variance of the estimate. Sigma is
    # an EMA of the rho_z^d-weighted second moment of the boundary points solved by the size term
    # itself, so directions are drawn where the set carries its volume.
    def _whiten_half(self, dtype, device) -> torch.Tensor | None:
        """Unit-determinant Sigma^{1/2}, or None while Sigma is unset or still warming up."""
        if self._whiten_sigma is None or self._whiten_steps < self.whiten_warmup_steps:
            return None
        d = self.residual_dim
        sig = self._whiten_sigma.to(device=device, dtype=torch.float64)
        if self.whiten_shrink > 0.0:
            iso = torch.diagonal(sig).sum() / d
            sig = (1.0 - self.whiten_shrink) * sig + self.whiten_shrink * iso * torch.eye(
                d, dtype=torch.float64, device=device)
        ev, q = torch.linalg.eigh(sig)
        ev = ev.clamp_min(1e-30)
        ev = ev / torch.exp(torch.log(ev).mean())
        return ((q * ev.sqrt()) @ q.transpose(0, 1))

    @torch.no_grad()
    def _update_whiten_sigma(self, rho: torch.Tensor, dirs: torch.Tensor,
                             log_weight: torch.Tensor) -> None:
        d = self.residual_dim
        batch, m = rho.shape
        pts = (rho.unsqueeze(2).double() * dirs.unsqueeze(0).double())          # (batch, m, d)
        wt = torch.softmax(log_weight.reshape(-1).double(), dim=0).view(batch, m)
        sig = torch.einsum("bm,bmd,bme->de", wt, pts, pts)
        sig = sig + 1e-9 * (torch.diagonal(sig).sum() / d).clamp_min(1e-30) * torch.eye(
            d, dtype=torch.float64, device=sig.device)
        det = torch.linalg.slogdet(sig)
        if not torch.isfinite(det.logabsdet) or det.sign <= 0:
            return
        sig = sig / torch.exp(det.logabsdet / d)
        if self._whiten_sigma is None:
            self._whiten_sigma = sig
        else:
            mom = self.whiten_momentum
            self._whiten_sigma = (1.0 - mom) * self._whiten_sigma.to(sig) + mom * sig
        self._whiten_steps += 1

    def _log_volume_core_whitened(self, x: torch.Tensor, tau, n_directions: int | None,
                                  seed: int | None, update: bool = True,
                                  diagnostics: dict | None = None) -> torch.Tensor:
        d = self.residual_dim
        m = self.train_volume_directions if n_directions is None else n_directions
        seed = self.training_qmc_seed() if seed is None else seed
        u = self._volume_directions(m, seed, x.dtype, x.device)
        m_actual = u.shape[0]
        shalf = self._whiten_half(x.dtype, x.device)
        if shalf is None:
            dirs, log_nv = u, None
        else:
            v = u.double() @ shalf
            nv = v.norm(dim=1).clamp_min(1e-30)
            dirs = (v / nv.unsqueeze(1)).to(x.dtype)
            log_nv = torch.log(nv)
        batch = x.shape[0]
        rho = self._solve_rays(x.repeat_interleave(m_actual, dim=0), dirs.repeat(batch, 1),
                               tau, diagnostics=diagnostics).view(batch, m_actual)
        log_rho_z = torch.log(rho.clamp_min(1e-12))
        if log_nv is not None:
            log_rho_z = log_rho_z - log_nv.to(log_rho_z.dtype).unsqueeze(0)
        summand = d * log_rho_z
        if update:
            self._update_whiten_sigma(rho.detach(), dirs.detach(), summand.detach())
        if diagnostics is not None:
            diagnostics["whitened"] = shalf is not None
            diagnostics["whiten_steps"] = int(self._whiten_steps)
            diagnostics["n_directions"] = m_actual
        log_v_d = (d / 2.0) * math.log(math.pi) - math.lgamma(d / 2.0 + 1.0)
        return log_v_d + torch.logsumexp(summand, dim=1, keepdim=True) - math.log(m_actual)

    def log_volume(self, x: torch.Tensor, tau, n_directions: int | None = None,
                   seed: int | None = None, diagnostics: dict | None = None) -> torch.Tensor:
        return self._log_volume_core(x, tau, n_directions, seed, diagnostics)

    def volume(self, x: torch.Tensor, tau, n_directions: int | None = None,
               seed: int | None = None, diagnostics: dict | None = None) -> torch.Tensor:
        return torch.exp(self._log_volume_core(x, tau, n_directions, seed, diagnostics))

    def prediction_set_size(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Training size hook: width (d=1), area (d=2) or the QMC volume at the training scramble."""
        if self.residual_dim == 1:
            return self.width(x, tau=tau)
        if self.residual_dim == 2:
            return self.area(x, tau=tau)
        if self.volume_whiten:
            return torch.exp(self._log_volume_core_whitened(
                x, tau, self.train_volume_directions, self.training_qmc_seed(),
                update=self.training))
        return self.volume(x, tau=tau, n_directions=self.train_volume_directions,
                           seed=self.training_qmc_seed())

    def log_prediction_set_size(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        if self.residual_dim == 1:
            return torch.log(self.width(x, tau=tau).clamp_min(1e-300))
        if self.residual_dim == 2:
            return torch.log(self.area(x, tau=tau).clamp_min(1e-300))
        if self.volume_whiten:
            # Sigma is only updated while training, so validation never feeds the preconditioner.
            return self._log_volume_core_whitened(
                x, tau, self.train_volume_directions, self.training_qmc_seed(),
                update=self.training)
        return self.log_volume(x, tau=tau, n_directions=self.train_volume_directions,
                               seed=self.training_qmc_seed())

    def prediction_set(self, x: torch.Tensor, tau: float, center: torch.Tensor, n_directions: int | None = None) -> torch.Tensor:
        """Boundary vertices of the 2-D set, shape (batch, M, 2), counter-clockwise."""
        if self.residual_dim != 2:
            raise ValueError("prediction_set() is only defined for residual_dim=2; use interval() for residual_dim=1")
        M = self.eval_area_directions if n_directions is None else n_directions
        rho, directions = self._polar_root(x, tau, M)
        points = rho.unsqueeze(-1) * directions.unsqueeze(0)  # (batch, M, 2)
        return points + center.unsqueeze(1)

    def membership(self, x: torch.Tensor, r: torch.Tensor, tau: float) -> torch.Tensor:
        return (self.score(x, r) <= tau).reshape(-1)

    def l2_penalty(self) -> torch.Tensor:
        return _l2_penalty(self, extra_excluded_names=_icnn_excluded_names(self))


def _predict_frozen(pm, x: torch.Tensor) -> torch.Tensor:
    """Evaluate the frozen point predictor (a torch module, or anything with .predict)."""
    if callable(pm) and not hasattr(pm, "predict"):
        return pm(x)
    if hasattr(pm, "predict"):
        arr = pm.predict(x.detach().cpu().numpy())
        return torch.as_tensor(arr, dtype=x.dtype, device=x.device)
    return pm(x)


class PredictionAugmentedICNN(ICNN):
    """ICNN whose encoder sees [x, F(x)], F the frozen point predictor (the paper's CoCoNet_ICNN).

    F is held in a plain list so it is not a submodule: it never enters parameters(), the
    optimizer or the state_dict, and it is evaluated under no_grad.
    """

    def __init__(self, input_dim: int, point_predictor=None, prediction_dim: int | None = None,
                 **kw):
        if point_predictor is None:
            raise ValueError("PredictionAugmentedICNN requires a fitted `point_predictor`")
        if prediction_dim is None:
            prediction_dim = int(kw.get("residual_dim", 1))
        super().__init__(input_dim=input_dim + int(prediction_dim), **kw)
        self._frozen_predictor = [point_predictor]
        self.covariate_dim = int(input_dim)
        self.prediction_dim = int(prediction_dim)

    def _apply(self, fn, *args, **kwargs):
        """Carry the frozen predictor along with .to()/.cuda() without registering it."""
        out = super()._apply(fn, *args, **kwargs)
        pm = self._frozen_predictor[0]
        if isinstance(pm, torch.nn.Module):
            pm._apply(fn, *args, **kwargs)
        return out

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pm = self._frozen_predictor[0]
        with torch.no_grad():
            f = _predict_frozen(pm, x).reshape(x.shape[0], -1).to(x.dtype)
        if f.shape[1] != self.prediction_dim:
            raise ValueError(
                f"point predictor returned {f.shape[1]} outputs, expected {self.prediction_dim}")
        return self.encoder(torch.cat([x, f], dim=1))

    def architecture(self) -> dict:
        a = dict(super().architecture())
        a.update({"icnn_encoder_input": "[x, F(x)]",
                  "icnn_covariate_dim": self.covariate_dim,
                  "icnn_prediction_dim": self.prediction_dim})
        return a


class PredictionOnlyICNN(PredictionAugmentedICNN):
    """Encoder ablation: the encoder sees F(x) only. The parent's first encoder layer is
    replaced in place by a `prediction_dim`-wide one."""

    def __init__(self, input_dim: int, point_predictor=None, prediction_dim: int | None = None,
                 **kw):
        super().__init__(input_dim=input_dim, point_predictor=point_predictor,
                         prediction_dim=prediction_dim, **kw)
        first = self.encoder[0]
        if not isinstance(first, nn.Linear):
            raise TypeError("PredictionOnlyICNN expects encoder[0] to be nn.Linear")
        self.encoder[0] = nn.Linear(self.prediction_dim, first.out_features)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pm = self._frozen_predictor[0]
        with torch.no_grad():
            f = _predict_frozen(pm, x).reshape(x.shape[0], -1).to(x.dtype)
        if f.shape[1] != self.prediction_dim:
            raise ValueError(
                f"point predictor returned {f.shape[1]} outputs, expected {self.prediction_dim}")
        return self.encoder(f)

    def architecture(self) -> dict:
        a = dict(super().architecture())
        a.update({"icnn_encoder_input": "[F(x)]"})
        return a


class NoContextICNN(ICNN):
    """Encoder ablation: no context at all, one global convex set. The encoder is still built
    (so the seeded initialisation of the body matches the other arms) but its output is
    replaced by zeros."""

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], self.encoder_output_dim, dtype=x.dtype, device=x.device)

    def architecture(self) -> dict:
        a = dict(super().architecture())
        a.update({"icnn_encoder_input": "none"})
        return a


class IdentityXICNN(ICNN):
    """Encoder ablation: the raw covariates x are the context (no encoder MLP)."""
    def __init__(self, input_dim: int, **kw):
        kw["encoder_hidden_dims"] = []
        super().__init__(input_dim=input_dim, **kw)
    def architecture(self) -> dict:
        a = dict(super().architecture()); a.update({"icnn_encoder_input": "[x] identity (no encoder)"}); return a


class IdentityXFICNN(PredictionAugmentedICNN):
    """Encoder ablation: raw [x, F(x)] is the context (no encoder MLP)."""
    def __init__(self, input_dim: int, point_predictor=None, prediction_dim: int | None = None, **kw):
        kw["encoder_hidden_dims"] = []
        super().__init__(input_dim=input_dim, point_predictor=point_predictor, prediction_dim=prediction_dim, **kw)
    def architecture(self) -> dict:
        a = dict(super().architecture()); a.update({"icnn_encoder_input": "[x, F(x)] identity (no encoder)"}); return a


class IdentityFICNN(PredictionAugmentedICNN):
    """Encoder ablation: raw F(x) is the context (no encoder MLP)."""
    def __init__(self, input_dim: int, point_predictor=None, prediction_dim: int | None = None, **kw):
        kw["encoder_hidden_dims"] = []
        if prediction_dim is None:
            prediction_dim = int(kw.get("residual_dim", 1))
        super().__init__(input_dim=0, point_predictor=point_predictor, prediction_dim=prediction_dim, **kw)
        self.covariate_dim = int(input_dim)
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pm = self._frozen_predictor[0]
        with torch.no_grad():
            f = _predict_frozen(pm, x).reshape(x.shape[0], -1).to(x.dtype)
        if f.shape[1] != self.prediction_dim:
            raise ValueError(f"point predictor returned {f.shape[1]} outputs, expected {self.prediction_dim}")
        return f
    def architecture(self) -> dict:
        a = dict(super().architecture()); a.update({"icnn_encoder_input": "[F(x)] identity (no encoder)"}); return a
