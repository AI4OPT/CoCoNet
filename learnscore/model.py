"""The ICNN conformity score.

    S(x, r) = G(x, r) - G(x, 0) - grad_r G(x, 0)^T r + epsilon * ||r||^2

G is a conditional input-convex network in the residual r = y - mu_hat(x):

    h      = encoder(x)
    z_1    = softplus(A_1(x) r + V_1(h) + b_1)
    z_k    = softplus(W_k^+ z_{k-1} + A_k(x) r + V_k(h) + b_k),  k = 2..L
    G(x,r) = w^+ . z_L

with W_k^+ = softplus(W_k_raw) and w^+ = softplus(w_raw), so a nonnegative combination
of convex, nondecreasing functions of r stays convex in r regardless of A_k(x)/V_k(h)/b_k,
which are unconstrained. The tangent correction makes S(x, 0) = 0 with zero gradient there;
the epsilon term makes S strongly convex, so its sublevel sets are compact.

There's no closed-form inverse for this score, so its width/area/volume are estimated by
root-finding -- see volume.py, which every dimension-generic hook below delegates to.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from learnscore import volume


class ICNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        residual_dim: int = 1,
        encoder_hidden_dims: list[int] | None = None,
        icnn_hidden_dims: list[int] | None = None,
        epsilon: float = 1e-3,
        train_area_directions: int = 32,
        eval_area_directions: int = 1024,
        train_volume_directions: int = 128,
        eval_volume_directions: int = 1024,
        qmc_base_seed: int = 0,
        eval_qmc_seed: int = 0,
    ) -> None:
        super().__init__()
        if residual_dim < 1:
            raise ValueError(f"residual_dim must be >= 1, got {residual_dim}")
        encoder_hidden_dims = encoder_hidden_dims or [64, 32]
        dims = icnn_hidden_dims or [32, 32]

        self.residual_dim = residual_dim
        self.output_dim = residual_dim  # name the training loss dispatches on
        self.epsilon = epsilon
        self.train_area_directions = train_area_directions
        self.eval_area_directions = eval_area_directions
        self.train_volume_directions = train_volume_directions
        self.eval_volume_directions = eval_volume_directions
        self.qmc_base_seed = qmc_base_seed
        self.eval_qmc_seed = eval_qmc_seed
        self.qmc_epoch = 0
        self.n_layers = len(dims)
        self.hidden_dims = dims

        layers: list[nn.Module] = []
        prev = input_dim
        for h in encoder_hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*layers)
        encoder_out = prev

        # parameter creation order matches the original repo, for reproducibility under a fixed seed
        self.a_nets = nn.ModuleList([nn.Linear(encoder_out, h * residual_dim) for h in dims])
        self.V = nn.ModuleList([nn.Linear(encoder_out, h) for h in dims])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(1, h)) for h in dims])
        self.W_raw = nn.ParameterList(
            [nn.Parameter(torch.zeros(dims[k], dims[k - 1])) for k in range(1, len(dims))])
        self.w_raw = nn.Parameter(torch.zeros(dims[-1]))

        for p in self.b:
            nn.init.normal_(p, std=0.1)
        for p in self.W_raw:
            # softplus(w) ~ 1/fan_in keeps the pre-activation O(1) regardless of width
            nn.init.normal_(p, mean=math.log(math.expm1(1.0 / p.shape[1])), std=0.1)
        nn.init.normal_(self.w_raw, std=0.1)

    # -- score -------------------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _residual_projection(self, h: torch.Tensor, r: torch.Tensor, k: int) -> torch.Tensor:
        a = self.a_nets[k](h).view(-1, self.hidden_dims[k], self.residual_dim)
        return torch.einsum("bhd,bd->bh", a, r)

    def _body(self, r: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        z = F.softplus(self._residual_projection(h, r, 0) + self.V[0](h) + self.b[0])
        for k in range(1, self.n_layers):
            W = F.softplus(self.W_raw[k - 1])
            z = F.softplus(z @ W.T + self._residual_projection(h, r, k) + self.V[k](h) + self.b[k])
        return z @ F.softplus(self.w_raw).unsqueeze(-1)

    def score(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """r = y - mu_hat(x), shape (batch, residual_dim). Nonnegative, shape (batch, 1)."""
        h = self.encode(x)
        r0 = torch.zeros_like(r, requires_grad=True)
        with torch.enable_grad():
            G0 = self._body(r0, h)
            (tangent,) = torch.autograd.grad(G0.sum(), r0, create_graph=True)
        G_r = self._body(r, h)
        plane = (r * tangent).sum(dim=1, keepdim=True)
        quad = self.epsilon * (r ** 2).sum(dim=1, keepdim=True)
        return G_r - G0 - plane + quad

    def membership(self, x: torch.Tensor, r: torch.Tensor, tau: float) -> torch.Tensor:
        return (self.score(x, r) <= tau).reshape(-1)

    def ray_root(self, x: torch.Tensor, tau: float, directions: torch.Tensor) -> torch.Tensor:
        return volume.solve_rays(self, x, directions, tau)

    # -- dimension-generic geometry hooks, delegated to volume.py ----------------------------

    def set_qmc_epoch(self, epoch: int) -> None:
        """Keys the training scramble to the epoch, so successive epochs integrate over
        different Sobol sets rather than overfitting one fixed finite direction grid."""
        self.qmc_epoch = int(epoch)

    def width(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        if self.residual_dim != 1:
            raise ValueError("width() is only defined for residual_dim=1")
        return volume.width(self, x, tau)

    def interval(self, x: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.residual_dim != 1:
            raise ValueError("interval() is only defined for residual_dim=1")
        return volume.interval(self, x, tau)

    def area(self, x: torch.Tensor, tau: float, n_directions: int | None = None) -> torch.Tensor:
        if self.residual_dim != 2:
            raise ValueError("area() is only defined for residual_dim=2")
        return volume.area(self, x, tau, n_directions or self.train_area_directions)

    def prediction_set(self, x: torch.Tensor, tau: float, center: torch.Tensor,
                        n_directions: int | None = None) -> torch.Tensor:
        return volume.prediction_set(self, x, tau, center, n_directions or self.eval_area_directions)

    def log_volume(self, x: torch.Tensor, tau, n_directions: int | None = None,
                    seed: int | None = None) -> torch.Tensor:
        return volume.log_volume(self, x, tau, self.residual_dim,
                                  n_directions or self.eval_volume_directions,
                                  self.eval_qmc_seed if seed is None else seed)

    def prediction_set_size(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        if self.residual_dim == 1:
            return self.width(x, tau)
        if self.residual_dim == 2:
            return self.area(x, tau)
        return volume.volume(self, x, tau, self.residual_dim, self.train_volume_directions,
                              self.qmc_base_seed + self.qmc_epoch)

    def log_prediction_set_size(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        if self.residual_dim == 1:
            return torch.log(self.width(x, tau).clamp_min(1e-300))
        if self.residual_dim == 2:
            return torch.log(self.area(x, tau).clamp_min(1e-300))
        return volume.log_volume(self, x, tau, self.residual_dim, self.train_volume_directions,
                                  self.qmc_base_seed + self.qmc_epoch)
