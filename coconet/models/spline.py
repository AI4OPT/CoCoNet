"""Convex piecewise-linear conformity score.

    s(x, r) = max_j max( psi_j+(x, z_j+), psi_j-(x, z_j-) ),   z = Q(x)^T r

Q(x) is an orthogonal matrix (matrix exponential of a skew-symmetric generator predicted from x;
identity at d = 1). Each psi is a convex, increasing, zero-anchored piecewise-linear function
whose K segment slopes are a cumulative sum of positive increments predicted from x. The sublevel
set {r : s(x, r) <= tau} is therefore a rotated box, with a closed-form volume.

Knots are empirical quantiles of the training residuals, one set per coordinate and sign. The
encoder sees [x, F(x)] with F the frozen point predictor.
"""

import numpy as np
import torch
from torch import nn


def compute_knots(r, num_knots, min_gap=1e-6):
    """(knots_plus, knots_minus) for one residual coordinate: increasing arrays of length K + 1
    starting at 0, from the quantiles of the positive residuals and of |negative residuals|."""
    r = np.asarray(r, dtype=float)
    pos = r[r >= 0]
    neg = -r[r < 0]
    if 0 < len(pos) < num_knots or 0 < len(neg) < num_knots:
        raise ValueError(f"not enough residuals of each sign for {num_knots} knots")
    levels = np.linspace(0.0, 1.0, num_knots + 1)[1:]

    def knots(values):
        k = np.zeros(num_knots + 1) if len(values) == 0 else np.concatenate([[0.0], np.quantile(values, levels)])
        for i in range(1, len(k)):
            if k[i] <= k[i - 1]:
                k[i] = k[i - 1] + min_gap
        return k

    return knots(pos), knots(neg)


def rotation_from_angles(d, params):
    """Q = matrix_exp(A), A skew-symmetric with A[p, q] = params[i] over the pairs p < q."""
    pairs = [(p, q) for p in range(d) for q in range(p + 1, d)]
    skew = torch.zeros(params.shape[0], d, d, dtype=params.dtype, device=params.device)
    for i, (p, q) in enumerate(pairs):
        skew[:, p, q] = params[:, i]
        skew[:, q, p] = -params[:, i]
    return torch.linalg.matrix_exp(skew)


class ConvexBranch(nn.Module):
    """Convex increasing piecewise-linear psi on [0, inf) with psi(0) = 0.

    Slopes m_1 = m_min + softplus(c_1), m_k = m_{k-1} + softplus(c_k) are nondecreasing by
    construction; beyond the last knot psi continues with slope m_K.
    """

    def __init__(self, knots, m_min):
        super().__init__()
        knots = torch.as_tensor(knots, dtype=torch.float32)
        if knots[0].item() != 0.0 or torch.any(knots[1:] <= knots[:-1]):
            raise ValueError("knots must start at 0 and increase strictly")
        self.register_buffer("knots", knots)
        self.num_knots = knots.shape[0] - 1
        self.m_min = float(m_min)

    def slopes(self, c_raw):
        return self.m_min + torch.cumsum(nn.functional.softplus(c_raw), dim=-1)

    def forward(self, t, c_raw):
        """psi(t) for t (batch, 1) >= 0 and raw increments c_raw (batch, K)."""
        m = self.slopes(c_raw)
        lo = self.knots[:-1].unsqueeze(0)
        hi = self.knots[1:].unsqueeze(0)
        inside = (torch.minimum(t, hi) - torch.minimum(t, lo)).clamp_min(0.0)
        beyond = (t - self.knots[-1]).clamp_min(0.0) * m[:, -1:]
        return (m * inside).sum(dim=-1, keepdim=True) + beyond

    def inverse(self, tau, c_raw):
        """psi^{-1}(tau): locate the segment, invert linearly. Returns (batch, 1)."""
        batch = c_raw.shape[0]
        tau_t = torch.as_tensor(tau, dtype=c_raw.dtype, device=c_raw.device)
        tau_t = tau_t.expand(batch, 1) if tau_t.ndim == 0 else tau_t.reshape(batch, 1)
        seg_len = (self.knots[1:] - self.knots[:-1]).unsqueeze(0)
        v = torch.cat([torch.zeros(batch, 1, dtype=c_raw.dtype, device=c_raw.device),
                       torch.cumsum(self.slopes(c_raw) * seg_len, dim=-1)], dim=-1)   # psi at the knots
        m = self.slopes(c_raw)     # evaluated again on purpose: keeps the gradient graph as trained
        v_last = v[:, -1:]
        idx = torch.argmax((v[:, 1:] >= tau_t).to(torch.float32), dim=-1, keepdim=True)
        v_lo = torch.gather(v, 1, idx)
        m_seg = torch.gather(m, 1, idx)
        knot_lo = self.knots[idx.squeeze(-1)].unsqueeze(-1)
        within = knot_lo + (tau_t - v_lo) / m_seg
        beyond = self.knots[-1] + (tau_t - v_last) / m[:, -1:]
        return torch.where(tau_t > v_last, beyond, within)


class SplineScore(nn.Module):
    def __init__(self, input_dim, output_dim, point_predictor, knots_plus, knots_minus,
                 encoder_hidden_dims=(64, 32), minimum_slope=1e-4):
        super().__init__()
        d = int(output_dim)
        self.output_dim = d
        self.rotate = d >= 2
        self._point_predictor = [point_predictor]     # not a submodule, see ICNN

        self.branches_plus = nn.ModuleList(ConvexBranch(k, minimum_slope) for k in knots_plus)
        self.branches_minus = nn.ModuleList(ConvexBranch(k, minimum_slope) for k in knots_minus)
        ks = {b.num_knots for b in list(self.branches_plus) + list(self.branches_minus)}
        if len(self.branches_plus) != d or len(self.branches_minus) != d or len(ks) != 1:
            raise ValueError("need one knot array per coordinate and sign, all with the same K")
        self.num_knots = ks.pop()

        layers = []
        prev = input_dim + d
        for h in encoder_hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.head_plus = nn.Linear(prev, d * self.num_knots)
        self.head_minus = nn.Linear(prev, d * self.num_knots)

        # The rotation head starts at zero (axis-aligned). Its default initialisation still draws
        # from the global RNG, so the RNG state is restored around it: everything constructed
        # after this model sees the same random stream at every d.
        self.angle_head = None
        if self.rotate:
            state = torch.random.get_rng_state()
            self.angle_head = nn.Linear(prev, d * (d - 1) // 2)
            nn.init.zeros_(self.angle_head.weight)
            nn.init.zeros_(self.angle_head.bias)
            torch.random.set_rng_state(state)

    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        pm = self._point_predictor[0]
        if isinstance(pm, nn.Module):
            pm._apply(fn, *args, **kwargs)
        return out

    def _encode(self, x):
        with torch.no_grad():
            f = self._point_predictor[0](x).reshape(x.shape[0], -1).to(x.dtype)
        return self.encoder(torch.cat([x, f], dim=1))

    def _coefficients(self, h):
        """Per coordinate j: (c_plus_j, c_minus_j), each (batch, K)."""
        raw_plus, raw_minus = self.head_plus(h), self.head_minus(h)
        K = self.num_knots
        return [(raw_plus[:, j * K:(j + 1) * K], raw_minus[:, j * K:(j + 1) * K])
                for j in range(self.output_dim)]

    def score(self, x, r):
        # The encoder runs once for the rotation and once for the slopes (two graph branches),
        # which is how the models were trained; sharing one pass changes the last bits.
        if self.rotate:
            q = rotation_from_angles(self.output_dim, -self.angle_head(self._encode(x)))
            z = torch.einsum("bij,bi->bj", q, r)       # Q(x)^T r
        else:
            z = r
        per_coordinate = []
        for j, (c_plus, c_minus) in enumerate(self._coefficients(self._encode(x))):
            z_j = z[:, j:j + 1]
            g_plus = self.branches_plus[j](z_j.clamp(min=0.0), c_plus)
            g_minus = self.branches_minus[j]((-z_j).clamp(min=0.0), c_minus)
            per_coordinate.append(torch.stack([g_minus, g_plus], dim=-1))
        values = torch.cat(per_coordinate, dim=1)                      # (batch, d, 2)
        return values.reshape(values.shape[0], -1).max(dim=1, keepdim=True).values

    def width(self, x, tau):
        (c_plus, c_minus), = self._coefficients(self._encode(x))
        return self.branches_plus[0].inverse(tau, c_plus) + self.branches_minus[0].inverse(tau, c_minus)

    def log_volume(self, x, tau):
        """log volume of the box: sum_j log(u_j+ + u_j-). Rotation has |det Q| = 1."""
        log_widths = []
        for j, (c_plus, c_minus) in enumerate(self._coefficients(self._encode(x))):
            w = self.branches_plus[j].inverse(tau, c_plus) + self.branches_minus[j].inverse(tau, c_minus)
            log_widths.append(torch.log(w.clamp_min(1e-300)))
        return torch.stack(log_widths, dim=0).sum(dim=0)

    def log_size(self, x, tau):
        if self.output_dim == 1:
            return torch.log(self.width(x, tau).clamp_min(1e-300))
        return self.log_volume(x, tau)

    def size(self, x, tau):
        if self.output_dim == 1:
            return self.width(x, tau)
        return torch.exp(self.log_volume(x, tau))
