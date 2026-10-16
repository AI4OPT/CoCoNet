"""Convex piecewise-linear conformity score.

    s(x, r) = max_j max( psi_j+(x, z_j+), psi_j-(x, z_j-) ),   z = Q(x)^T r          (norm="inf")
    s(x, r) = sum_j ( psi_j+(x, z_j+) + psi_j-(x, z_j-) )                             (norm="l1")

Q(x) is an orthogonal matrix (matrix exponential of a skew-symmetric generator predicted from x;
identity at d = 1). Each psi is a convex, increasing, zero-anchored piecewise-linear function
whose K segment slopes are a cumulative sum of positive increments predicted from x. The sublevel
set {r : s(x, r) <= tau} is therefore a rotated box (norm="inf") or an asymmetric polytope (norm="l1"),
both with a closed-form volume.

Knots are empirical quantiles of the training residuals, one set per coordinate and sign. The
encoder sees [x, F(x)] with F the frozen point predictor.
"""

import math

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
                 encoder_hidden_dims=(64, 32), minimum_slope=1e-4, norm="inf"):
        super().__init__()
        if norm not in ("inf", "l1"):
            raise ValueError("norm must be 'inf' or 'l1'")
        self.norm = norm
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
        if norm == "l1":
            # The paper's l_1 runs also built an (unused) exponent head here; drawing its initialisation keeps
            # the training random stream, and so the trained models, identical.
            nn.Linear(prev, 1)
        # l_1 volume grids: training, validation / checkpoint selection, reported test volume
        self.l1_train_grid, self.l1_size_grid, self.l1_eval_grid = 512, 2048, 2 ** 15

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
        if self.norm == "l1":
            return self._score_l1(x, r)
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

    def _score_l1(self, x, r):
        """sum_j psi_j(z_j), written as the l_1 runs computed it (one encoder pass, slopes before the rotation,
        sum scaled by the largest term), so a trained model gives the same numbers."""
        h = self._encode(x)
        coefficients = self._coefficients(h)
        z = torch.einsum("bij,bi->bj", rotation_from_angles(self.output_dim, -self.angle_head(h)), r) if self.rotate else r
        v = torch.stack([(self.branches_plus[j](z[:, j:j + 1].clamp_min(0.0), c_plus)
                          + self.branches_minus[j]((-z[:, j:j + 1]).clamp_min(0.0), c_minus)).reshape(-1)
                         for j, (c_plus, c_minus) in enumerate(coefficients)], dim=1)
        top = v.max(dim=-1).values.clamp_min(1e-9)
        return (top * (v / top.unsqueeze(-1)).clamp_min(1e-30).sum(-1)).reshape(-1, 1)

    def _l1_log_volume(self, x, tau, n):
        """Exact volume of {sum_j psi_j(z_j) <= tau}. Substituting u_j = psi_j(z_j) and merging the orthants gives
        V = int_{u >= 0, sum u <= tau} prod_j H_j(u_j) du, with H_j the summed inverse slopes of the two branches,
        a step function: H_j(u) = sum_i c_ji 1[u >= a_ji]. Each product integrates in closed form,
        V = (1/d!) sum_{i_1..i_d} prod_j c_{j i_j} (tau - sum_j a_{j i_j})_+^d, summed directly when the number of
        terms is small; otherwise V = int_0^tau (H_1 * ... * H_d)(t) dt by FFT convolution on n cells of [0, tau]
        with Richardson extrapolation 2 V(2n) - V(n). Differentiable in the slopes."""
        h = self._encode(x)
        A, C = [], []
        for j, (c_plus, c_minus) in enumerate(self._coefficients(h)):
            mp, mm = self.branches_plus[j].slopes(c_plus), self.branches_minus[j].slopes(c_minus)
            parts_a, parts_c = [], []
            for br, m in ((self.branches_plus[j], mp), (self.branches_minus[j], mm)):
                m = m.double(); kn = br.knots.double()
                parts_a.append(torch.cumsum(m[:, :-1] * torch.diff(kn)[: m.shape[1] - 1], dim=1))   # psi at inner knots
                parts_c.append(1.0 / m[:, 1:] - 1.0 / m[:, :-1])
            c0 = 1.0 / mp[:, :1].double() + 1.0 / mm[:, :1].double()
            A.append(torch.cat([torch.zeros_like(c0)] + parts_a, 1)); C.append(torch.cat([c0] + parts_c, 1))
        tau, d = float(tau), self.output_dim
        if math.prod(a.shape[1] for a in A) * x.shape[0] <= 4e6:
            s = torch.zeros_like(A[0][:, :1]); w = torch.ones_like(s)
            for a, c in zip(A, C):
                s = (s[:, :, None] + a[:, None, :]).flatten(1); w = (w[:, :, None] * c[:, None, :]).flatten(1)
            v = (w * torch.clamp(tau - s, min=0.0) ** d).sum(1) / math.factorial(d)
        else:
            def conv(n):
                e = torch.arange(n + 1, dtype=torch.float64, device=A[0].device) * (tau / n)
                g = None
                for a, c in zip(A, C):
                    cum = (c[:, None, :] * torch.clamp(e[None, :, None] - a[:, None, :], min=0.0)).sum(2)
                    hc = cum[:, 1:] - cum[:, :-1]                       # H_j integrated over each cell
                    g = hc if g is None else torch.fft.irfft(torch.fft.rfft(g, 2 * n) * torch.fft.rfft(hc, 2 * n), 2 * n)[:, :n]
                return g.sum(1)
            v = 2 * conv(2 * n) - conv(n)
        return torch.log(v.clamp_min(1e-300)).to(x.dtype).reshape(-1, 1)

    def _l1_log_volume_chunked(self, x, tau, n):
        return torch.cat([self._l1_log_volume(x[i:i + 64], tau, n) for i in range(0, x.shape[0], 64)], 0)

    def width(self, x, tau):
        (c_plus, c_minus), = self._coefficients(self._encode(x))
        return self.branches_plus[0].inverse(tau, c_plus) + self.branches_minus[0].inverse(tau, c_minus)

    def log_volume(self, x, tau):
        """log volume of the box: sum_j log(u_j+ + u_j-) (l_1: the exact polytope volume, on the fine grid).
        Rotation has |det Q| = 1."""
        if self.norm == "l1":
            return self._l1_log_volume_chunked(x, tau, self.l1_eval_grid)
        log_widths = []
        for j, (c_plus, c_minus) in enumerate(self._coefficients(self._encode(x))):
            w = self.branches_plus[j].inverse(tau, c_plus) + self.branches_minus[j].inverse(tau, c_minus)
            log_widths.append(torch.log(w.clamp_min(1e-300)))
        return torch.stack(log_widths, dim=0).sum(dim=0)

    def log_size(self, x, tau):
        if self.output_dim == 1:
            return torch.log(self.width(x, tau).clamp_min(1e-300))
        if self.norm == "l1":
            return self._l1_log_volume(x, tau, self.l1_train_grid)
        return self.log_volume(x, tau)

    def size(self, x, tau):
        if self.output_dim == 1:
            return self.width(x, tau)
        if self.norm == "l1":
            with torch.no_grad():
                return torch.exp(self._l1_log_volume_chunked(x, tau, self.l1_size_grid))
        return torch.exp(self.log_volume(x, tau))
