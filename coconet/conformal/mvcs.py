"""Minimum-volume covering set (MVCS) baseline, convex version: a p-norm ball around F(x),

    C(x; q) = { y : || Lambda(x) (y - F(x)) ||_p <= q },    Lambda(x) = A(x) A(x)^T + ridge I,

with A(x) an MLP and one learned p, projected onto p >= 1 after every step so the set is convex.
Training minimises, per batch of B residuals,

    log sum_i det Lambda(x_i)^-1  +  d log s_(j)  +  log V_{d,p},     j = ceil((1 - alpha) B),

where s_(j) is the j-th smallest score of the batch and V_{d,p} the unit p-ball volume.
"""

import math

import numpy as np
import torch
from torch import nn

EPS = 1e-12


def unit_ball_log_volume(p, k):
    """log V_{k,p} = k log 2 + k lgamma(1 + 1/p) - lgamma(1 + k/p)."""
    if not torch.is_tensor(p):
        p = torch.as_tensor(float(p))
    return k * math.log(2.0) + k * torch.lgamma(1.0 + 1.0 / p) - torch.lgamma(1.0 + k / p)


def p_norm(v, p):
    """||v||_p along the last axis in log space; an all-zero row gives exactly 0."""
    log_abs = torch.log(v.abs().clamp_min(EPS))
    result = torch.exp(torch.logsumexp(p * log_abs, dim=-1) / p)
    return torch.where((v == 0).all(dim=-1), torch.zeros_like(result), result)


def _mlp(in_dim, out_dim, hidden_dim, n_layers):
    layers = []
    prev = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(prev, hidden_dim), nn.ReLU()]
        prev = hidden_dim
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class MvcsModel(nn.Module):
    def __init__(self, context_dim, output_dim, hidden_dim, n_layers, ridge, p_init):
        super().__init__()
        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.ridge = float(ridge)
        self.matrix_net = _mlp(context_dim, output_dim * output_dim, hidden_dim, n_layers)
        head = self.matrix_net[-1]              # Lambda(x) = I at initialisation
        nn.init.zeros_(head.weight)
        head.bias.data.copy_(torch.eye(output_dim).reshape(-1))
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.register_buffer("x_mean", torch.zeros(1, context_dim))
        self.register_buffer("x_std", torch.ones(1, context_dim))

    def set_context_normalization(self, x):
        x = np.asarray(x, dtype=float).reshape(len(x), -1)
        self.x_mean.copy_(torch.as_tensor(x.mean(axis=0, keepdims=True), dtype=self.x_mean.dtype))
        self.x_std.copy_(torch.as_tensor(np.maximum(x.std(axis=0, keepdims=True), 1e-8), dtype=self.x_std.dtype))

    def project_p(self):
        with torch.no_grad():
            if float(self.p.item()) < 1.0:
                self.p.clamp_(min=1.0)

    def lambda_matrix(self, xt):
        a = self.matrix_net((xt - self.x_mean) / self.x_std).reshape(-1, self.output_dim, self.output_dim)
        eye = torch.eye(self.output_dim, device=a.device, dtype=a.dtype)
        return a @ a.transpose(-1, -2) + self.ridge * eye

    def scores(self, xt, residual):
        return p_norm(torch.einsum("nij,nj->ni", self.lambda_matrix(xt), residual), self.p)

    def loss(self, xt, residual, alpha):
        """The batch objective; -inf log-determinants (non-finite A) make it non-finite."""
        k = self.output_dim
        sign, logdet = torch.linalg.slogdet(self.lambda_matrix(xt))
        if not bool(torch.all(sign > 0)):
            logdet = torch.full_like(logdet, float("-inf"))
        # Lambda(x) is computed a second time for the scores: two graph branches, as trained.
        s = p_norm(torch.einsum("nij,nj->ni", self.lambda_matrix(xt), residual), self.p)
        b = int(s.shape[0])
        j = int(min(max(math.ceil((1.0 - alpha) * b), 1), b))
        threshold = torch.kthvalue(s, j).values
        return (torch.logsumexp(-logdet, dim=0) + k * torch.log(threshold.clamp_min(EPS))
                + unit_ball_log_volume(self.p, k))


class MvcsScore:
    """numpy-facing view of a trained model: score / contains / volume."""

    def __init__(self, model):
        self.model = model

    def _xt(self, x):
        arr = np.asarray(x, dtype=float).reshape(len(x), -1)
        return torch.as_tensor(arr, dtype=torch.float32)

    @property
    def p(self):
        return float(self.model.p.detach())

    def score(self, x, r):
        with torch.no_grad():
            rt = torch.as_tensor(np.asarray(r, dtype=float), dtype=torch.float32)
            return self.model.scores(self._xt(x), rt).numpy().astype(float)

    def contains(self, x, r, q):
        return self.score(x, r) <= float(q)

    def volume(self, x, q):
        """V = V_{d,p} q^d / det Lambda(x)."""
        d = self.model.output_dim
        with torch.no_grad():
            sign, logdet = torch.linalg.slogdet(self.model.lambda_matrix(self._xt(x)))
            if not bool(torch.all(sign > 0)):
                raise RuntimeError("Lambda(x) is not positive definite")
        log_unit = float(unit_ball_log_volume(torch.tensor(self.p, dtype=torch.float64), d))
        return np.exp(log_unit + d * np.log(max(float(q), EPS)) - logdet.numpy().astype(np.float64))

    def n_params(self):
        return sum(int(t.numel()) for t in self.model.parameters())
