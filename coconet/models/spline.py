"""CoCoNet_Spline: a convex, coordinate-wise piecewise-linear conformity score.

For a residual r in R^d the score is

    s(x, r) = max_j max( psi_{j,+}(x, z_j^+), psi_{j,-}(x, z_j^-) ),   z = Q(x)^T r,
    z_j^+ = max(z_j, 0),  z_j^- = max(-z_j, 0),

with Q(x) an orthogonal matrix (matrix exponential of a skew-symmetric generator predicted from
x; identity at d = 1) and each psi a convex, increasing, zero-anchored piecewise-linear branch
whose K segment slopes are a cumulative sum of positive increments predicted from x
(`ConvexSplineBranch`). Every branch is convex, a max of convex functions is convex, so each
sublevel set {r : s(x, r) <= tau} is a rotated hyperrectangle whose volume is a product of
per-coordinate widths, computed in closed form (`log_volume`).

Knot locations are the empirical quantiles of the D_score residual magnitudes, one set per
coordinate and sign (`compute_score_training_knots`). The encoder input is [x, F(x)] with F the
frozen point predictor (`PredictionAugmentedSplineScore`), the same input the ICNN uses.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from coconet.models.icnn import _predict_frozen
from coconet.models.param_groups import l2_penalty as _l2_penalty
from coconet.models.param_groups import split_regularized_params


class SplineScoreError(ValueError):
    pass


# --------------------------------------------------------------------------------------------
# orthogonal transformation Q(x) = matrix_exp(skew(theta(x)))
# --------------------------------------------------------------------------------------------
def givens_pairs(d: int) -> list[tuple[int, int]]:
    """Coordinate pairs (p, q), p < q, in lexicographic order: d(d-1)/2 of them."""
    return [(p, q) for p in range(d) for q in range(p + 1, d)]


def compose_matrix_exp_rotation(d: int, params: torch.Tensor) -> torch.Tensor:
    """Q = matrix_exp(A) with A skew-symmetric, A[p, q] = params[pair index]. params: (batch,
    d(d-1)/2). Returns (batch, d, d), orthogonal with det +1; params = 0 gives Q = I."""
    pairs = givens_pairs(d)
    if params.shape[-1] != len(pairs):
        raise SplineScoreError(f"expected {len(pairs)} parameters for d={d}, got {params.shape[-1]}")
    batch = params.shape[0]
    skew = torch.zeros(batch, d, d, dtype=params.dtype, device=params.device)
    for idx, (p, q) in enumerate(pairs):
        skew[:, p, q] = params[:, idx]
        skew[:, q, p] = -params[:, idx]
    return torch.linalg.matrix_exp(skew)


# --------------------------------------------------------------------------------------------
# knots
# --------------------------------------------------------------------------------------------
def compute_score_training_knots(
    r_score: np.ndarray, num_knots: int, min_gap: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """(knots_plus, knots_minus) for one residual coordinate: each a strictly increasing array
    of length num_knots + 1 starting at 0, from the quantiles of the positive residuals and of
    the magnitudes of the negative residuals in D_score. A sign with no residuals at all gets
    all-zero knots (no margin on that side)."""
    r_score = np.asarray(r_score, dtype=float)
    pos = r_score[r_score >= 0]
    neg_abs = -r_score[r_score < 0]

    if 0 < len(pos) < num_knots or 0 < len(neg_abs) < num_knots:
        raise SplineScoreError(
            f"Not enough positive ({len(pos)}) / negative ({len(neg_abs)}) residuals in "
            f"D_score to fit {num_knots} knots per branch"
        )

    quantile_levels = np.linspace(0.0, 1.0, num_knots + 1)[1:]
    knots_plus = (np.zeros(num_knots + 1) if len(pos) == 0
                 else np.concatenate([[0.0], np.quantile(pos, quantile_levels)]))
    knots_minus = (np.zeros(num_knots + 1) if len(neg_abs) == 0
                  else np.concatenate([[0.0], np.quantile(neg_abs, quantile_levels)]))
    return _ensure_strictly_increasing(knots_plus, min_gap), _ensure_strictly_increasing(knots_minus, min_gap)


def _ensure_strictly_increasing(knots: np.ndarray, min_gap: float) -> np.ndarray:
    knots = knots.astype(float).copy()
    for i in range(1, len(knots)):
        if knots[i] <= knots[i - 1]:
            knots[i] = knots[i - 1] + min_gap
    return knots


def _normalize_knots(knots, output_dim: int, name: str) -> list[np.ndarray]:
    """One bare 1-D array at d = 1, a length-d sequence of per-coordinate arrays above."""
    if output_dim == 1:
        arr = np.asarray(knots, dtype=float)
        if arr.ndim != 1:
            raise SplineScoreError(f"{name} must be a 1-D array at output_dim=1, got shape {arr.shape}")
        return [arr]
    if isinstance(knots, np.ndarray):
        raise SplineScoreError(
            f"{name} must be a length-{output_dim} sequence of per-coordinate knot arrays at "
            f"output_dim={output_dim} (one array per response coordinate), not a single array"
        )
    knots_list = list(knots)
    if len(knots_list) != output_dim:
        raise SplineScoreError(f"{name} must have exactly {output_dim} entries (one per coordinate), got {len(knots_list)}")
    return [np.asarray(k, dtype=float) for k in knots_list]


# --------------------------------------------------------------------------------------------
# one convex branch psi on [0, inf)
# --------------------------------------------------------------------------------------------
class ConvexSplineBranch(nn.Module):
    """Convex, strictly increasing, zero-anchored piecewise-linear function on [0, inf).

    Segment slopes are a cumulative sum of positive increments,
        m_1 = m_min + softplus(c_1),   m_k = m_{k-1} + softplus(c_k),
    so m_1 <= ... <= m_K by construction; beyond the last knot psi extrapolates with m_K.
    """

    def __init__(self, knots: np.ndarray, m_min: float = 1e-4) -> None:
        super().__init__()
        knots_t = torch.as_tensor(knots, dtype=torch.float32)
        if knots_t.ndim != 1 or knots_t.shape[0] < 2:
            raise SplineScoreError(f"knots must be 1-D with at least 2 entries (u_0, u_1), got {knots_t.shape}")
        if knots_t[0].item() != 0.0:
            raise SplineScoreError(f"knots[0] must be 0.0 (u_0), got {knots_t[0].item()}")
        if torch.any(knots_t[1:] <= knots_t[:-1]):
            raise SplineScoreError("knots must be strictly increasing")
        if m_min <= 0:
            raise SplineScoreError(f"m_min must be positive, got {m_min}")
        self.register_buffer("knots", knots_t)
        self.num_knots = knots_t.shape[0] - 1  # K
        self.m_min = float(m_min)

    @staticmethod
    def raw_width(num_knots: int) -> int:
        return num_knots  # K raw increments

    def slopes(self, c_raw: torch.Tensor) -> torch.Tensor:
        """m_1 <= ... <= m_K, all > m_min. c_raw: (batch, K). Returns (batch, K)."""
        increments = nn.functional.softplus(c_raw)
        return self.m_min + torch.cumsum(increments, dim=-1)

    def knot_values(self, c_raw: torch.Tensor) -> torch.Tensor:
        """psi at every knot, v_0 = 0. Returns (batch, K+1)."""
        m = self.slopes(c_raw)
        seg_len = (self.knots[1:] - self.knots[:-1]).unsqueeze(0)
        v_rest = torch.cumsum(m * seg_len, dim=-1)
        v0 = torch.zeros(c_raw.shape[0], 1, dtype=c_raw.dtype, device=c_raw.device)
        return torch.cat([v0, v_rest], dim=-1)

    def forward(self, t: torch.Tensor, c_raw: torch.Tensor) -> torch.Tensor:
        """psi(t; x). t, output: (batch, 1) >= 0; c_raw: (batch, K)."""
        m = self.slopes(c_raw)                         # (batch, K)
        lo = self.knots[:-1].unsqueeze(0)               # (1, K)
        hi = self.knots[1:].unsqueeze(0)                # (1, K)
        contrib = (torch.minimum(t, hi) - torch.minimum(t, lo)).clamp_min(0.0)  # (batch, K)
        within_or_below = (m * contrib).sum(dim=-1, keepdim=True)
        beyond = (t - self.knots[-1]).clamp_min(0.0) * m[:, -1:]
        return within_or_below + beyond

    def inverse(self, tau, c_raw: torch.Tensor) -> torch.Tensor:
        """psi^{-1}(tau; x): find the segment, invert linearly. Returns (batch, 1) >= 0."""
        batch = c_raw.shape[0]
        tau_t = torch.as_tensor(tau, dtype=c_raw.dtype, device=c_raw.device)
        tau_t = tau_t.expand(batch, 1) if tau_t.ndim == 0 else tau_t.reshape(batch, 1)

        v = self.knot_values(c_raw)   # (batch, K+1)
        m = self.slopes(c_raw)        # (batch, K)
        v_K = v[:, -1:]

        beyond_mask = tau_t > v_K
        ge_mask = v[:, 1:] >= tau_t
        m_star_idx = torch.argmax(ge_mask.to(torch.float32), dim=-1, keepdim=True)

        v_m_minus_1 = torch.gather(v, 1, m_star_idx)
        m_m_star = torch.gather(m, 1, m_star_idx)
        kappa_m_minus_1 = self.knots[m_star_idx.squeeze(-1)].unsqueeze(-1)

        within_range = kappa_m_minus_1 + (tau_t - v_m_minus_1) / m_m_star
        beyond_value = self.knots[-1] + (tau_t - v_K) / m[:, -1:]
        return torch.where(beyond_mask, beyond_value, within_range)


# --------------------------------------------------------------------------------------------
# the score
# --------------------------------------------------------------------------------------------
class MonotoneSplineScore(nn.Module):
    """Shared encoder -> two coefficient heads (positive and negative branches of every
    coordinate) and, for output_dim >= 2, a rotation head. See the module docstring."""

    def __init__(
        self,
        input_dim: int,
        knots_plus,
        knots_minus,
        hidden_dims: list[int] | None = None,
        epsilon: float = 1e-3,
        output_dim: int = 1,
        transformation_lr_multiplier: float = 10.0,
        minimum_slope: float = 1e-4,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        if output_dim < 1:
            raise SplineScoreError(f"output_dim must be >= 1, got {output_dim!r}")
        self.epsilon = epsilon
        self.conditional = True
        self.output_dim = output_dim
        self.enforce_convexity = True
        self.minimum_slope = float(minimum_slope)
        self.aggregation = "max"
        # Rotation is undefined for a scalar residual.
        self.transformation_mode = "none" if output_dim == 1 else "conditional"
        self.transformation_lr_multiplier = float(transformation_lr_multiplier)

        knots_plus_list = _normalize_knots(knots_plus, output_dim, "knots_plus")
        knots_minus_list = _normalize_knots(knots_minus, output_dim, "knots_minus")
        self.branches_plus = nn.ModuleList(ConvexSplineBranch(k, m_min=minimum_slope) for k in knots_plus_list)
        self.branches_minus = nn.ModuleList(ConvexSplineBranch(k, m_min=minimum_slope) for k in knots_minus_list)

        num_knots_values = {b.num_knots for b in list(self.branches_plus) + list(self.branches_minus)}
        if len(num_knots_values) != 1:
            raise SplineScoreError(
                f"all knot arrays (every coordinate, both directions) must define the same "
                f"number of knots K, got {sorted(num_knots_values)}"
            )
        self.num_knots = num_knots_values.pop()
        self._coeff_width = ConvexSplineBranch.raw_width(self.num_knots)
        head_width = output_dim * self._coeff_width

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.encoder = nn.Sequential(*layers)
        self.head_plus = nn.Linear(prev_dim, head_width)
        self.head_minus = nn.Linear(prev_dim, head_width)

        # Rotation head, zero-initialised so training starts axis-aligned. Its default init
        # still draws from the global RNG; the state is restored around it so the random stream
        # seen by everything after construction is the same as at output_dim = 1.
        self.angle_head: nn.Module | None = None
        n_angles = len(givens_pairs(output_dim)) if output_dim >= 2 else 0
        _rng_state = torch.random.get_rng_state()
        if self.transformation_mode == "conditional":
            self.angle_head = nn.Linear(prev_dim, n_angles)
            nn.init.zeros_(self.angle_head.weight)
            nn.init.zeros_(self.angle_head.bias)
        if self.transformation_mode != "none":
            torch.random.set_rng_state(_rng_state)

    # --- rotation ---------------------------------------------------------------------------
    def angles(self, x: torch.Tensor) -> torch.Tensor:
        """theta(x), shape (batch, d(d-1)/2): the skew generator coordinates of Q(x)."""
        batch = x.shape[0]
        n_angles = len(givens_pairs(self.output_dim)) if self.output_dim >= 2 else 0
        if self.transformation_mode == "none":
            return torch.zeros(batch, n_angles, dtype=x.dtype, device=x.device)
        h = self.encoder(x)
        return self.angle_head(h)

    def rotation_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Q(x) = matrix_exp(skew(-theta(x))); at d = 2 this is [[cos, -sin], [sin, cos]]."""
        return compose_matrix_exp_rotation(self.output_dim, -self.angles(x))

    def _to_rotated(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """z = Q(x)^T r (r itself at d = 1)."""
        if self.transformation_mode == "none":
            return r
        q = self.rotation_matrix(x)
        return torch.einsum("bij,bi->bj", q, r)

    def _from_rotated(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """r = Q(x) z, for z of shape (batch, d) or (batch, M, d)."""
        if self.transformation_mode == "none":
            return z
        q = self.rotation_matrix(x)
        if z.dim() == 3:
            return torch.einsum("bij,bmj->bmi", q, z)
        return torch.einsum("bij,bj->bi", q, z)

    def rotation_parameters(self) -> list[nn.Parameter]:
        """Parameters of the rotation head only (their own optimizer group)."""
        if self.transformation_mode == "conditional":
            return list(self.angle_head.parameters())
        return []

    # --- coefficients / score ---------------------------------------------------------------
    def _coeffs_by_coordinate(self, x: torch.Tensor) -> list[tuple[tuple, tuple]]:
        """Per coordinate: ((c_raw_plus,), (c_raw_minus,)), the raw slope increments."""
        h = self.encoder(x)
        raw_plus = self.head_plus(h)
        raw_minus = self.head_minus(h)

        width = self._coeff_width
        result = []
        for j in range(self.output_dim):
            rp = raw_plus[:, j * width : (j + 1) * width]
            rm = raw_minus[:, j * width : (j + 1) * width]
            result.append(((rp,), (rm,)))
        return result

    def branch_slopes(self, x: torch.Tensor) -> torch.Tensor:
        """Segment slopes m_1..m_K for every (coordinate, direction): (batch, d, 2, K);
        direction 0 = negative, 1 = positive."""
        coeffs_by_coord = self._coeffs_by_coordinate(x)
        rows = []
        for j, ((rp,), (rm,)) in enumerate(coeffs_by_coord):
            m_plus = self.branches_plus[j].slopes(rp)
            m_minus = self.branches_minus[j].slopes(rm)
            rows.append(torch.stack([m_minus, m_plus], dim=1).unsqueeze(1))  # (batch,1,2,K)
        return torch.cat(rows, dim=1)  # (batch, output_dim, 2, K)

    def directional_scores(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Branch values for every transformed coordinate and direction, (batch, d, 2).
        The inactive direction of each coordinate evaluates to psi(0) = 0."""
        z = self._to_rotated(x, r)
        coeffs_by_coord = self._coeffs_by_coordinate(x)
        per_coordinate = []
        for j, (plus_args, minus_args) in enumerate(coeffs_by_coord):
            z_j = z[:, j : j + 1]
            g_plus = self.branches_plus[j](z_j.clamp(min=0.0), *plus_args)
            g_minus = self.branches_minus[j]((-z_j).clamp(min=0.0), *minus_args)
            per_coordinate.append(torch.stack([g_minus, g_plus], dim=-1))
        return torch.cat(per_coordinate, dim=1)

    def score(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """r: (batch, d). Returns the score, (batch, 1): the max over all branch values."""
        ds = self.directional_scores(x, r)
        return ds.reshape(ds.shape[0], -1).max(dim=1, keepdim=True).values

    # --- prediction-set geometry ------------------------------------------------------------
    def width(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        """Interval length at d = 1, (batch, 1)."""
        if self.output_dim != 1:
            raise SplineScoreError("width() is only defined for output_dim=1; use area()/prediction_set_size() instead")
        plus_args, minus_args = self._coeffs_by_coordinate(x)[0]
        upper = self.branches_plus[0].inverse(tau, *plus_args)
        lower_mag = self.branches_minus[0].inverse(tau, *minus_args)
        return upper + lower_mag

    def area(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        if self.output_dim != 2:
            raise SplineScoreError("area() is only defined for output_dim=2; use width() for output_dim=1")
        return self.volume(x, tau)

    def log_volume(self, x: torch.Tensor, tau) -> torch.Tensor:
        """log of the (rotated) hyperrectangle volume, sum_j log(u_{j,+} + u_{j,-}), (batch, 1).
        Rotation has |det Q| = 1 so it leaves the volume unchanged."""
        coeffs_by_coord = self._coeffs_by_coordinate(x)
        log_widths = []
        for j, (plus_args, minus_args) in enumerate(coeffs_by_coord):
            u_plus = self.branches_plus[j].inverse(tau, *plus_args)
            u_minus = self.branches_minus[j].inverse(tau, *minus_args)
            width = (u_plus + u_minus).clamp_min(1e-300)
            log_widths.append(torch.log(width))
        return torch.stack(log_widths, dim=0).sum(dim=0)

    def volume(self, x: torch.Tensor, tau, dtype: torch.dtype | None = None) -> torch.Tensor:
        log_v = self.log_volume(x, tau)
        if dtype is not None:
            log_v = log_v.to(dtype)
        return torch.exp(log_v)

    def volume_root(self, x: torch.Tensor, tau) -> torch.Tensor:
        return torch.exp(self.log_volume(x, tau) / self.output_dim)

    def prediction_set_size(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        return self.width(x, tau=tau) if self.output_dim == 1 else self.volume(x, tau=tau)

    def log_prediction_set_size(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        if self.output_dim == 1:
            return torch.log(self.width(x, tau=tau).clamp_min(1e-300))
        return self.log_volume(x, tau=tau)

    def interval(self, x: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
        """(lower_offset, upper_offset) around the point prediction at d = 1."""
        if self.output_dim != 1:
            raise SplineScoreError("interval() is only defined for output_dim=1; use prediction_set() for output_dim=2")
        plus_args, minus_args = self._coeffs_by_coordinate(x)[0]
        upper_offset = self.branches_plus[0].inverse(tau, *plus_args)
        lower_offset = -self.branches_minus[0].inverse(tau, *minus_args)
        return lower_offset, upper_offset

    def prediction_set(self, x: torch.Tensor, tau: float, center: torch.Tensor) -> torch.Tensor:
        """Vertices of the rotated rectangle at d = 2, (batch, 4, 2), around `center`."""
        if self.output_dim != 2:
            raise SplineScoreError("prediction_set() is only defined for output_dim=2; use interval() for output_dim=1")
        coeffs_by_coord = self._coeffs_by_coordinate(x)
        (plus1, minus1), (plus2, minus2) = coeffs_by_coord
        u1_plus = self.branches_plus[0].inverse(tau, *plus1)
        u1_minus = self.branches_minus[0].inverse(tau, *minus1)
        u2_plus = self.branches_plus[1].inverse(tau, *plus2)
        u2_minus = self.branches_minus[1].inverse(tau, *minus2)
        z_bottom_right = torch.cat([u1_plus, -u2_minus], dim=1)
        z_top_right = torch.cat([u1_plus, u2_plus], dim=1)
        z_top_left = torch.cat([-u1_minus, u2_plus], dim=1)
        z_bottom_left = torch.cat([-u1_minus, -u2_minus], dim=1)
        z_verts = torch.stack([z_bottom_right, z_top_right, z_top_left, z_bottom_left], dim=1)
        r_verts = self._from_rotated(x, z_verts)
        return r_verts + center.unsqueeze(1)

    def membership(self, x: torch.Tensor, r: torch.Tensor, tau: float) -> torch.Tensor:
        return (self.score(x, r) <= tau).reshape(-1)

    def contains(self, x: torch.Tensor, r: torch.Tensor, tau: float) -> torch.Tensor:
        return self.membership(x, r, tau)

    def l2_penalty(self) -> torch.Tensor:
        return _l2_penalty(self)

    def regularized_parameter_names(self) -> tuple[list[str], list[str]]:
        return split_regularized_params(self)


class PredictionAugmentedSplineScore(MonotoneSplineScore):
    """MonotoneSplineScore whose encoder input is [x, F(x)], F the frozen point predictor.
    F is held in a plain list so it is not a submodule (never trained or saved), and is moved
    along with the model by `_apply`."""

    def __init__(self, input_dim: int, point_predictor=None, prediction_dim: int | None = None, **kw):
        pm = point_predictor
        if pm is None:
            raise ValueError("PredictionAugmentedSplineScore requires a fitted point predictor")
        if prediction_dim is None:
            prediction_dim = int(kw.get("output_dim", 1))
        super().__init__(input_dim=int(input_dim) + int(prediction_dim), **kw)
        self._frozen_predictor = [pm]
        self.covariate_dim = int(input_dim)
        self.prediction_dim = int(prediction_dim)
        self.encoder_input = "[x, F(x)]"

    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        pm = self._frozen_predictor[0]
        if isinstance(pm, torch.nn.Module):
            pm._apply(fn, *args, **kwargs)
        return out

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.covariate_dim:
            raise ValueError(f"expected raw covariates of width {self.covariate_dim}, got {x.shape[1]} "
                             f"(already augmented?)")
        with torch.no_grad():
            f = _predict_frozen(self._frozen_predictor[0], x).reshape(x.shape[0], -1).to(x.dtype)
        if f.shape[1] != self.prediction_dim:
            raise ValueError(f"point predictor returned {f.shape[1]} outputs, expected {self.prediction_dim}")
        return torch.cat([x, f], dim=1)

    def angles(self, x: torch.Tensor) -> torch.Tensor:
        return super().angles(self._augment(x))

    def _coeffs_by_coordinate(self, x: torch.Tensor):
        return super()._coeffs_by_coordinate(self._augment(x))
