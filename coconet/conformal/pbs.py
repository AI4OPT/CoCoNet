"""Pullback conformal sets (PBS) on an amortized conditional neural optimal-transport model (AC-NQR_U).

Model. A potential phi(u, x), strongly convex in u: a partially input-convex network plus
(alpha / 2) ||u||^2 with alpha = exp(w) learned. The vector quantile is Q(u, x) = grad_u phi(u, x)
and the vector rank Q^{-1}(r, x) = argmax_u { u'r - phi(u, x) }, solved by a damped Newton method
warm-started by an amortizer network u~(r, x). Training minimises the semi-dual objective
E[phi(U, X)] + E[u_hat'R - phi(u_hat, X)] with U ~ N(0, I), AdamW, cosine decay to 0, gradient
clipping, last iterate kept; the amortizer is fitted to u_hat alongside.

Set. s(x, r) = ||Q^{-1}(r, x)||_2, calibrated by split conformal, C(x) = F(x) + Q(B(0, q), x). The set
is connected but neither convex nor star-shaped. Its volume is the change-of-variables integral
vol = int_{B(0, q)} det Hess_u phi(u, x) du, estimated with scrambled Sobol points in the ball.

Configuration selection. Each candidate (width, depth, batch size) is trained on the training part
of D_score; the kept one has the smallest mean validation coverage error of the plug-in sets
{||Q^{-1}|| <= chi_d(1 - a)} over a in {0.1, 0.2, 0.3}, ties broken by the validation objective.
"""

import copy
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import chi2


def _inv_softplus(v):
    return v + torch.log(-torch.expm1(-v))


class ActNorm(nn.Module):
    """Per-feature affine normalisation with a positive scale, initialised from the first batch."""

    def __init__(self, n):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n))
        self.log_scale = nn.Parameter(torch.zeros(n))
        self._pending = False

    def forward(self, z):
        if self._pending:
            with torch.no_grad():
                self.bias.copy_(-z.mean(0))
                self.log_scale.copy_(-torch.log(z.std(0) + 1e-6))
            self._pending = False
        return (z + self.bias) * torch.exp(self.log_scale)


class Potential(nn.Module):
    """phi(u, x) = PICNN(u, x) + 0.5 exp(w) ||u||^2, convex in u.

        c_{i+1} = ELU(W~_i c_i + b~_i),   c_0 = x
        z_{i+1} = softplus(ActNorm(W^z_i (z_i * softplus(W^zc_i c_i + b^z_i)) + W^u_i (u * (W^uc_i c_i + b^u_i))
                                   + W^c_i c_i + b^c_i)),   W^z_i = softplus(raw) >= 0
    """

    def __init__(self, d, dx, width, depth):
        super().__init__()
        self.d, self.depth = int(d), int(depth)
        cd = [dx] + [width] * depth
        zin = [0] + [width] * depth
        zout = [width] * depth + [1]
        self.ctx = nn.ModuleList([nn.Linear(cd[i], cd[i + 1]) for i in range(depth)])
        self.Wc = nn.ModuleList([nn.Linear(cd[i], zout[i]) for i in range(depth + 1)])
        self.Wuc = nn.ModuleList([nn.Linear(cd[i], d) for i in range(depth + 1)])
        self.Wu = nn.ModuleList([nn.Linear(d, zout[i], bias=False) for i in range(depth + 1)])
        self.Wzc = nn.ModuleList([nn.Linear(cd[i], zin[i]) if zin[i] else nn.Identity() for i in range(depth + 1)])
        self.Wz_raw = nn.ParameterList([nn.Parameter(torch.empty(zout[i], zin[i])) for i in range(1, depth + 1)])
        self.norms = nn.ModuleList([ActNorm(width) for _ in range(depth)])
        self.out_norm = ActNorm(1)
        self.log_alpha = nn.Parameter(torch.zeros(()))
        with torch.no_grad():
            for i in range(depth + 1):
                nn.init.ones_(self.Wuc[i].bias)
                if zin[i]:
                    nn.init.ones_(self.Wzc[i].bias)
            for p in self.Wz_raw:
                p.copy_(_inv_softplus(torch.rand_like(p) * (2.0 / p.shape[1]) + 1e-4))

    def begin_data_init(self):
        for n in list(self.norms) + [self.out_norm]:
            n._pending = True

    def forward(self, u, x):
        c, z = x, None
        for i in range(self.depth + 1):
            pre = self.Wc[i](c) + self.Wu[i](u * self.Wuc[i](c))
            if z is not None:
                pre = pre + F.linear(z * F.softplus(self.Wzc[i](c)), F.softplus(self.Wz_raw[i - 1]))
            if i < self.depth:
                z = F.softplus(self.norms[i](pre))
                c = F.elu(self.ctx[i](c))
            else:
                out = F.softplus(self.out_norm(pre)).squeeze(-1)
        return out + 0.5 * torch.exp(self.log_alpha) * (u * u).sum(-1)


class Amortizer(nn.Module):
    """u~(r, x) = MLP([r; x]) + W r + b, the identity map at initialisation."""

    def __init__(self, d, dx, width, depth):
        super().__init__()
        layers, h = [], d + dx
        for _ in range(max(depth, 1)):
            layers += [nn.Linear(h, width), nn.ELU()]
            h = width
        last = nn.Linear(h, d)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.mlp = nn.Sequential(*layers, last)
        self.skip = nn.Linear(d, d)
        with torch.no_grad():
            self.skip.weight.copy_(torch.eye(d))
            self.skip.bias.zero_()

    def forward(self, r, x):
        return self.mlp(torch.cat([r, x], dim=-1)) + self.skip(r)


def _grad_hess(phi, u, x, hess=True):
    d = u.shape[1]
    with torch.enable_grad():
        u = u.detach().requires_grad_(True)
        f = phi(u, x)
        (g,) = torch.autograd.grad(f.sum(), u, create_graph=hess)
        H = None
        if hess:
            H = torch.stack([torch.autograd.grad(g[:, k].sum(), u, retain_graph=(k < d - 1))[0] for k in range(d)], dim=1)
    return f.detach(), g.detach(), (H.detach() if H is not None else None)


@torch.no_grad()
def newton_inverse(phi, r, x, u0, tol=1e-5, max_iter=60):
    """argmin_u phi(u, x) - u'r row by row (damped Newton, Armijo backtracking); rows whose
    ||grad phi - r|| is below tol are frozen."""
    u = u0.detach().clone()
    active = torch.arange(u.shape[0], device=u.device)
    slack = 8.0 * torch.finfo(u.dtype).eps
    for _ in range(max_iter):
        if active.numel() == 0:
            break
        ua, ra, xa = u[active], r[active], x[active]
        f, g, H = _grad_hess(phi, ua, xa)
        g = g - ra
        keep = g.norm(dim=1) >= tol
        if not bool(keep.any()):
            break
        active, ua, ra, xa, f, g, H = (t[keep] for t in (active, ua, ra, xa, f, g, H))
        step = torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)
        obj0 = f - (ua * ra).sum(1)
        slope = (g * step).sum(1)
        t = torch.ones_like(obj0)
        accepted = torch.zeros_like(obj0, dtype=torch.bool)
        for _ in range(30):
            un = ua - t[:, None] * step
            objn = phi(un, xa) - (un * ra).sum(1)
            accepted = accepted | (objn <= obj0 - 1e-4 * t * slope + slack * (obj0.abs() + 1.0))
            if bool(accepted.all()):
                break
            t = torch.where(accepted, t, 0.5 * t)
        u[active] = ua - t[:, None] * step
    return u


@dataclass
class PbsConfig:
    width: int = 16
    depth: int = 4
    batch_size: int = 1024
    lr: float = 1e-2
    amort_lr: float = 1e-2
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    amort_restart_epochs: int = 10
    max_epochs: int = 800
    min_eval_every: int = 10
    newton_tol: float = 1e-5
    newton_max_iter: int = 60
    n_val_u: int = 32
    alpha_init: float = 1.0
    seed: int = 0


class AcNqr(nn.Module):
    def __init__(self, d, dx, width, depth):
        super().__init__()
        self.d = int(d)
        self.phi = Potential(d, dx, width, depth)
        self.amort = Amortizer(d, dx, width, depth)

    @torch.no_grad()
    def rank(self, r, x, tol=1e-5, max_iter=60):
        return newton_inverse(self.phi, r, x, self.amort(r, x), tol=tol, max_iter=max_iter)


def _semi_dual(phi, x, r, u_ref, u_hat):
    xr = x.repeat_interleave(u_ref.shape[0] // x.shape[0], dim=0) if u_ref.shape[0] != x.shape[0] else x
    return phi(u_ref, xr).mean() + ((u_hat * r).sum(1) - phi(u_hat, x)).mean()


def train(x_tr, r_tr, x_va, r_va, cfg):
    """Returns (model, info) with the validation coverage error and objective of the last iterate."""
    torch.manual_seed(cfg.seed)
    g = torch.Generator(device="cpu").manual_seed(cfg.seed)
    d, dx = r_tr.shape[1], x_tr.shape[1]
    Xtr, Rtr = torch.as_tensor(x_tr, dtype=torch.float32), torch.as_tensor(r_tr, dtype=torch.float32)
    Xva, Rva = torch.as_tensor(x_va, dtype=torch.float32), torch.as_tensor(r_va, dtype=torch.float32)
    n = len(Xtr)
    bs = min(cfg.batch_size, n)
    steps_per_epoch = int(math.ceil(n / bs))
    eval_every = max(cfg.min_eval_every, steps_per_epoch)
    total_steps = int(cfg.max_epochs) * steps_per_epoch
    model = AcNqr(d, dx, cfg.width, cfg.depth)
    U_val = torch.randn(len(Xva) * cfg.n_val_u, d, generator=g)
    with torch.no_grad():                                     # ActNorm initialisation on one batch
        model.phi.begin_data_init()
        ii = torch.randperm(n, generator=g)[:bs]
        model.phi(torch.randn(len(ii), d, generator=g), Xtr[ii])
        model.phi.log_alpha.fill_(math.log(cfg.alpha_init))
    opt = torch.optim.AdamW(model.phi.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=0.0)
    opt_a = torch.optim.AdamW(model.amort.parameters(), lr=cfg.amort_lr, weight_decay=cfg.weight_decay)
    sched_a = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_a, T_0=max(1, cfg.amort_restart_epochs * steps_per_epoch))

    def validation():
        with torch.no_grad():
            u_hat = model.rank(Rva, Xva, tol=cfg.newton_tol, max_iter=cfg.newton_max_iter)
            return float(_semi_dual(model.phi, Xva, Rva, U_val, u_hat)), u_hat.norm(dim=1).numpy()

    perm, pos, step = torch.randperm(n, generator=g), 0, 0
    while step < total_steps:
        if pos + bs > n:
            perm, pos = torch.randperm(n, generator=g), 0
        idx = perm[pos:pos + bs]
        pos += bs
        xb, rb = Xtr[idx], Rtr[idx]
        with torch.no_grad():
            u0 = model.amort(rb, xb)
        u_hat = newton_inverse(model.phi, rb, xb, u0, tol=cfg.newton_tol, max_iter=cfg.newton_max_iter)
        u_ref = torch.randn(len(idx), d, generator=g)
        loss = _semi_dual(model.phi, xb, rb, u_ref, u_hat.detach())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.phi.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        loss_a = ((model.amort(rb, xb) - u_hat.detach()) ** 2).sum(1).mean()
        opt_a.zero_grad(set_to_none=True)
        loss_a.backward()
        torch.nn.utils.clip_grad_norm_(model.amort.parameters(), cfg.grad_clip)
        opt_a.step()
        sched_a.step()
        step += 1
        if step % eval_every == 0 or step == total_steps:
            if not math.isfinite(validation()[0]):
                break                                         # training diverged: stop here
    model.eval()
    val_loss, norms = validation()
    cov_err = float(np.mean([abs(np.mean(norms <= math.sqrt(chi2.ppf(1 - a, d))) - (1 - a)) for a in (0.1, 0.2, 0.3)]))
    return model, {"val_coverage_error": cov_err, "val_loss": val_loss, "steps": step,
                   "alpha": float(torch.exp(model.phi.log_alpha)), "config": asdict(cfg)}


@torch.no_grad()
def scores(model, x, r, batch=8192):
    """s = ||Q^{-1}(r, x)||_2, solved in float64 to ||grad|| < 1e-9."""
    m64 = copy.deepcopy(model).double()
    out = []
    for i in range(0, len(x), batch):
        xt = torch.as_tensor(x[i:i + batch], dtype=torch.float64)
        rt = torch.as_tensor(r[i:i + batch], dtype=torch.float64)
        out.append(newton_inverse(m64.phi, rt, xt, m64.amort(rt, xt), tol=1e-9, max_iter=100).numpy())
    return np.linalg.norm(np.concatenate(out), axis=1)


@torch.no_grad()
def log_volume(model, x, q, budgets, seed=0, block=1 << 13):
    """log vol C(x) = log int_{B(0, q)} det Hess_u phi(u, x) du at every budget (one nested Sobol
    pass; the Hessian is computed in float64)."""
    d = model.d
    want = sorted({int(m) for m in budgets})
    log_ball = (d / 2.0) * math.log(math.pi) - math.lgamma(d / 2.0 + 1.0) + d * math.log(q)
    eng = torch.quasirandom.SobolEngine(dimension=d + 1, scramble=True, seed=int(seed))
    eps = torch.finfo(torch.float64).eps
    phi64 = copy.deepcopy(model.phi).double().eval()
    X64 = torch.as_tensor(x, dtype=torch.float32).double()
    acc = torch.zeros(len(X64), dtype=torch.float64)
    out, done = {}, 0
    while done < want[-1]:
        mb = min(block, want[-1] - done)
        z = eng.draw(mb).to(torch.float64).clamp(eps, 1 - eps)
        direction = torch.erfinv(2 * z[:, :d] - 1) * math.sqrt(2)
        u = (direction / direction.norm(dim=1, keepdim=True)) * (z[:, d:] ** (1.0 / d)) * q
        for i in range(len(X64)):
            _, _, H = _grad_hess(phi64, u, X64[i:i + 1].expand(mb, -1))
            sign, logdet = torch.linalg.slogdet(H)
            if not bool((sign > 0).all()):
                raise RuntimeError("Hessian determinant not positive")
            acc[i] += torch.exp(logdet).sum()
        done += mb
        if done in want:
            out[done] = log_ball + torch.log(acc / done)
    return out
