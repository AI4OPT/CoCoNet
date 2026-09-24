"""Input-convex conformity score.

    S(x, r) = G(x, r) - G(x, 0) - grad_r G(x, 0)^T r + eps ||r||^2

G is convex in the residual r = y - F(x): hidden-to-hidden weights are kept nonnegative through
a softplus and the activation is softplus. The context x enters through an encoder h(x) that
shifts and rotates every layer. The sublevel set {r : S(x, r) <= tau} has no closed form, so its
boundary is traced by root finding along rays:

    d = 1   interval length from two one-sided roots
    d = 2   area from M equally spaced angles
    d > 2   volume from M quasi-Monte Carlo (Sobol) directions, whitened, in log space

Root finding is bracket-and-bisect under no_grad plus one Newton step whose value is discarded
but whose gradient is kept, which is the implicit-function derivative of the root.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class RootFindingError(RuntimeError):
    pass


def _inverse_softplus(y):
    return math.log(math.expm1(y))


def sobol_directions(d, n, seed, dtype, device):
    """n quasi-uniform unit vectors in R^d from a scrambled Sobol sequence (own generator)."""
    engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=int(seed))
    z = engine.draw(n).to(torch.float64)
    eps = torch.finfo(torch.float64).eps
    g = torch.erfinv(2.0 * z.clamp(eps, 1.0 - eps) - 1.0) * math.sqrt(2.0)
    u = g / g.norm(dim=1, keepdim=True).clamp_min(1e-300)
    return u.to(dtype=dtype, device=device)


# Evaluation volumes are solved in blocks so that a block never holds more rays than this.
MAX_RAY_ELEMENTS = 1 << 23
MAX_RAY_ROWS = 640 * 1024

# What the encoder sees. "xf" is the main model; the others are the encoder ablation.
ENCODER_INPUTS = ("xf", "f", "x", "none", "id_xf", "id_x", "id_f")


class ICNN(nn.Module):
    def __init__(self, input_dim, residual_dim, point_predictor=None, encoder_input="xf",
                 encoder_hidden_dims=(64, 32), hidden_dim=64, depth=2, epsilon=1e-3,
                 train_area_directions=512, eval_area_directions=2048,
                 train_volume_directions=512, eval_volume_directions=4096,
                 whiten_momentum=0.05, whiten_shrink=0.05, whiten_warmup_steps=20,
                 root_finding=None, set_boundary=None):
        super().__init__()
        if encoder_input not in ENCODER_INPUTS:
            raise ValueError(f"encoder_input must be one of {ENCODER_INPUTS}, got {encoder_input!r}")
        uses_f = encoder_input in ("xf", "f", "id_xf", "id_f")
        if uses_f and point_predictor is None:
            raise ValueError(f"encoder_input={encoder_input!r} needs the point predictor")

        d = int(residual_dim)
        self.residual_dim = self.output_dim = d
        self.epsilon = float(epsilon)
        self.encoder_input = encoder_input
        # The frozen predictor lives in a plain list so it is not a submodule (no parameters,
        # no state_dict entry); .to() moves it along through _apply below.
        self._point_predictor = [point_predictor]

        rf = root_finding or {}
        self.initial_bracket = float(rf.get("initial_bracket", 1.0))
        self.bracket_growth = float(rf.get("bracket_growth", 2.0))
        self.max_bracket_expansions = int(rf.get("max_bracket_expansions", 40))
        self.bisection_tol = float(rf.get("bisection_tol", 1e-6))
        self.bisection_max_iter = int(rf.get("bisection_max_iter", 100))
        sb = set_boundary or {}
        self.ray_initial_radius = float(sb.get("initial_radius", 1.0))
        self.ray_growth = float(sb.get("bracket_growth", 2.0))
        self.ray_max_radius = float(sb.get("max_radius", 1e4))
        self.ray_tolerance = float(sb.get("tolerance", 1e-5))
        self.ray_bisection_steps = int(sb.get("bisection_steps", 60))

        self.train_area_directions = int(train_area_directions)
        self.eval_area_directions = int(eval_area_directions)
        self.train_volume_directions = int(train_volume_directions)
        self.eval_volume_directions = int(eval_volume_directions)
        self.whiten_momentum = float(whiten_momentum)
        self.whiten_shrink = float(whiten_shrink)
        self.whiten_warmup_steps = int(whiten_warmup_steps)
        # Whitening and direction-seed state are plain attributes, not buffers (kept out of the state_dict).
        self._whiten_sigma = None
        self._whiten_steps = 0
        self.qmc_base_seed = 0
        self.qmc_epoch = 0
        self.eval_qmc_seed = 0

        # Encoder. The identity arms have no encoder MLP; "f" starts with the [x, F(x)] width
        # and swaps its first layer at the end (this keeps the seeded initialisation of the
        # convex body identical across arms).
        context_dim = {"xf": input_dim + d, "f": input_dim + d, "x": input_dim, "none": input_dim,
                       "id_xf": input_dim + d, "id_x": input_dim, "id_f": d}[encoder_input]
        dims_enc = [] if encoder_input.startswith("id_") else list(encoder_hidden_dims)
        layers = []
        prev = context_dim
        for h in dims_enc:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.context_out = prev

        # Convex body. Parameter creation order fixes the seeded initialisation; do not reorder.
        dims = [int(hidden_dim)] * int(depth)
        self.hidden_dims = dims
        self.a_nets = nn.ModuleList([nn.Linear(prev, h * d) for h in dims])   # A_k(x)
        self.V = nn.ModuleList([nn.Linear(prev, h) for h in dims])            # V_k h(x)
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(1, h)) for h in dims])
        self.W_raw = nn.ParameterList([nn.Parameter(torch.zeros(dims[k], dims[k - 1]))
                                       for k in range(1, len(dims))])
        self.w_raw = nn.Parameter(torch.zeros(dims[-1]))
        for p in self.b:
            nn.init.normal_(p, std=0.1)
        for p in self.W_raw:   # softplus(W) ~ 1 / fan_in keeps pre-activations O(1)
            nn.init.normal_(p, mean=_inverse_softplus(1.0 / p.shape[1]), std=0.1)
        nn.init.normal_(self.w_raw, std=0.1)

        if encoder_input == "f":
            self.encoder[0] = nn.Linear(d, self.encoder[0].out_features)

    def _apply(self, fn, *args, **kwargs):
        out = super()._apply(fn, *args, **kwargs)
        pm = self._point_predictor[0]
        if isinstance(pm, nn.Module):
            pm._apply(fn, *args, **kwargs)
        return out

    # ---- score --------------------------------------------------------------------------
    def encode(self, x):
        mode = self.encoder_input
        if mode == "none":
            return torch.zeros(x.shape[0], self.context_out, dtype=x.dtype, device=x.device)
        if mode in ("x", "id_x"):
            return self.encoder(x)
        with torch.no_grad():
            f = self._point_predictor[0](x).reshape(x.shape[0], -1).to(x.dtype)
        if mode in ("f", "id_f"):
            return self.encoder(f)
        return self.encoder(torch.cat([x, f], dim=1))

    def _body(self, r, h):
        """G(x, r) for the encoded context h. Returns (batch, 1)."""
        # Order of the operations is kept as trained (it fixes the gradient accumulation order).
        z = None
        for k in range(len(self.hidden_dims)):
            shift = self.b[k] + self.V[k](h)
            if k == 0:
                z = F.softplus(self._project(h, r, 0) + shift)
            else:
                W = F.softplus(self.W_raw[k - 1])
                z = F.softplus(z @ W.T + self._project(h, r, k) + shift)
        w = F.softplus(self.w_raw)
        return z @ w.unsqueeze(-1)

    def _project(self, h, r, k):
        """A_k(x) r, with A_k(x) predicted from the context."""
        a = self.a_nets[k](h).view(-1, self.hidden_dims[k], self.residual_dim)
        return torch.einsum("bhd,bd->bh", a, r)

    def _tangent(self, x):
        """(G(x, 0), grad_r G(x, 0), h(x)), reused by every root finder for the same x."""
        h = self.encode(x)
        r0 = torch.zeros(x.shape[0], self.residual_dim, dtype=x.dtype, device=x.device,
                         requires_grad=True)
        with torch.enable_grad():
            G0 = self._body(r0, h)
            (tangent,) = torch.autograd.grad(G0.sum(), r0, create_graph=True)
        return G0, tangent, h

    def _score_core(self, r, h, G0, tangent):
        return (self._body(r, h) - G0 - (r * tangent).sum(dim=1, keepdim=True)
                + self.epsilon * (r ** 2).sum(dim=1, keepdim=True))

    def score(self, x, r):
        G0, tangent, h = self._tangent(x)
        return self._score_core(r, h, G0, tangent)

    # ---- root finding -------------------------------------------------------------------
    def _newton_step(self, f, u_star):
        """Newton correction with value u_star and gradient -f(u*) / f'(u*)."""
        with torch.enable_grad():
            u = u_star.clone().requires_grad_(True)
            f_val = f(u)
            (f_grad,) = torch.autograd.grad(f_val.sum(), u, create_graph=True, retain_graph=True)
            denom = torch.where(f_grad.abs() < 1e-8, torch.full_like(f_grad, 1e-8), f_grad)
            correction = -f_val / denom
        return u_star + (correction - correction.detach())

    def _one_sided_root(self, f, batch, dtype, device):
        """u >= 0 with f(u) = 0 for increasing f (d = 1)."""
        with torch.no_grad():
            lo = torch.zeros(batch, 1, dtype=dtype, device=device)
            hi = torch.full((batch, 1), self.initial_bracket, dtype=dtype, device=device)
            f_hi = f(hi)
            for _ in range(self.max_bracket_expansions):
                expand = f_hi < 0
                if not torch.any(expand):
                    break
                hi = torch.where(expand, hi * self.bracket_growth, hi)
                f_hi = f(hi)
            if torch.any(f_hi < 0):
                raise RootFindingError("bracket expansion did not bound the root")
            for _ in range(self.bisection_max_iter):
                mid = (lo + hi) / 2.0
                right = f(mid) < 0
                lo = torch.where(right, mid, lo)
                hi = torch.where(right, hi, mid)
                if (hi - lo).max().item() < self.bisection_tol:
                    break
            u_star = (lo + hi) / 2.0
        return self._newton_step(f, u_star)

    def _solve_rays(self, x_rows, dir_rows, tau):
        """rho (N, 1) with score(x_i, rho_i d_i) = tau, one ray per row."""
        G0, tangent, h = self._tangent(x_rows)

        def f(rho):
            return self._score_core(rho * dir_rows, h, G0, tangent) - tau

        n = x_rows.shape[0]
        with torch.no_grad():
            lo = torch.zeros(n, 1, dtype=x_rows.dtype, device=x_rows.device)
            hi = torch.full((n, 1), self.ray_initial_radius, dtype=x_rows.dtype, device=x_rows.device)
            f_hi = f(hi)
            while True:
                expand = (f_hi < 0) & (hi < self.ray_max_radius)
                if not torch.any(expand):
                    break
                hi = torch.where(expand, torch.clamp(hi * self.ray_growth, max=self.ray_max_radius), hi)
                f_hi = f(hi)
            n_bad = int((f_hi < 0).sum().item())
            if n_bad > 0:
                raise RootFindingError(
                    f"{n_bad}/{n} rays did not reach score = tau within radius {self.ray_max_radius}")
            for _ in range(self.ray_bisection_steps):
                mid = (lo + hi) / 2.0
                right = f(mid) < 0
                lo = torch.where(right, mid, lo)
                hi = torch.where(right, hi, mid)
                if (hi - lo).max().item() < self.ray_tolerance:
                    break
            rho_star = (lo + hi) / 2.0
        return self._newton_step(f, rho_star)

    # ---- d = 1 and d = 2 ----------------------------------------------------------------
    def width(self, x, tau):
        G0, tangent, h = self._tangent(x)
        r_pos = self._one_sided_root(lambda u: self._score_core(u, h, G0, tangent) - tau,
                                     x.shape[0], x.dtype, x.device)
        r_neg = self._one_sided_root(lambda u: self._score_core(-u, h, G0, tangent) - tau,
                                     x.shape[0], x.dtype, x.device)
        return r_pos + r_neg

    def polar_radii(self, x, tau, n_directions):
        """Boundary radius along M equally spaced angles: (rho (batch, M), directions (M, 2))."""
        batch = x.shape[0]
        thetas = torch.linspace(0.0, 2.0 * math.pi, n_directions + 1, dtype=x.dtype, device=x.device)[:-1]
        directions = torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=1)
        rho = self._solve_rays(x.repeat_interleave(n_directions, dim=0), directions.repeat(batch, 1), tau)
        return rho.view(batch, n_directions), directions

    def area(self, x, tau, n_directions):
        rho, _ = self.polar_radii(x, tau, n_directions)
        return (math.pi / n_directions) * (rho ** 2).sum(dim=1, keepdim=True)

    # ---- d > 2: whitened QMC volume -----------------------------------------------------
    # With z = Sigma^{-1/2} r the volume is log V_d + logsumexp(d log rho_z) - log M for any
    # symmetric positive definite Sigma of unit determinant; Sigma only changes the variance.
    # Sigma is a running average of the rho^d-weighted second moment of the boundary points
    # found during training.
    def set_qmc_epoch(self, epoch):
        self.qmc_epoch = int(epoch)

    def _whiten_half(self, device):
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
        return (q * ev.sqrt()) @ q.transpose(0, 1)

    @torch.no_grad()
    def _update_whiten_sigma(self, rho, dirs, log_weight):
        d = self.residual_dim
        batch, m = rho.shape
        pts = rho.unsqueeze(2).double() * dirs.unsqueeze(0).double()
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

    def log_volume(self, x, tau, n_directions, seed, update):
        d = self.residual_dim
        u = sobol_directions(d, n_directions, seed, x.dtype, x.device)
        m = u.shape[0]
        shalf = self._whiten_half(x.device)
        if shalf is None:
            dirs, log_nv = u, None
        else:
            v = u.double() @ shalf
            nv = v.norm(dim=1).clamp_min(1e-30)
            dirs = (v / nv.unsqueeze(1)).to(x.dtype)
            log_nv = torch.log(nv)
        batch = x.shape[0]
        rho = self._solve_rays(x.repeat_interleave(m, dim=0), dirs.repeat(batch, 1), tau).view(batch, m)
        log_rho = torch.log(rho.clamp_min(1e-12))
        if log_nv is not None:
            log_rho = log_rho - log_nv.to(log_rho.dtype).unsqueeze(0)
        summand = d * log_rho
        if update:
            self._update_whiten_sigma(rho.detach(), dirs.detach(), summand.detach())
        log_v_d = (d / 2.0) * math.log(math.pi) - math.lgamma(d / 2.0 + 1.0)
        return log_v_d + torch.logsumexp(summand, dim=1, keepdim=True) - math.log(m)

    # ---- the two size hooks ------------------------------------------------------------
    def log_size(self, x, tau):
        """Training size (log): directions re-scrambled every epoch, whitening updated."""
        if self.residual_dim == 1:
            return torch.log(self.width(x, tau).clamp_min(1e-300))
        if self.residual_dim == 2:
            return torch.log(self.area(x, tau, self.train_area_directions).clamp_min(1e-300))
        return self.log_volume(x, tau, self.train_volume_directions,
                               self.qmc_base_seed + self.qmc_epoch, update=self.training)

    def size(self, x, tau):
        """Evaluation size with a fixed direction set (comparable across epochs and models)."""
        if self.residual_dim == 1:
            return self.width(x, tau)
        if self.residual_dim == 2:
            return self.area(x, tau, self.eval_area_directions)
        d, M = self.residual_dim, self.eval_volume_directions
        per = max(1, min(MAX_RAY_ELEMENTS // (M * d), MAX_RAY_ROWS // M))
        blocks = [torch.exp(self.log_volume(x[i:i + per], tau, M, self.eval_qmc_seed, update=False))
                  for i in range(0, x.shape[0], per)]
        return torch.cat(blocks, dim=0)
