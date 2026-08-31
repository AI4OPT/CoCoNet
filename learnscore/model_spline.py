"""Convex piecewise-linear ("spline") conformity score -- the third CLeS-family method,
alongside the ICNN and the asymmetric-linear score.

Each coordinate gets a signed branch, one convex piecewise-linear function per sign:

    psi(z) = sum_{k=1..K} m_k * clip(z - u_{k-1}, 0, u_k - u_{k-1})       (z >= 0, breakpoints
                                                                            u_0=0 < u_1 < ... < u_K=inf)

with slopes built by a cumulative sum of positive increments, m_1 = m_min + softplus(c_1),
m_k = m_{k-1} + softplus(c_k) -- this guarantees m_1 <= ... <= m_K, i.e. psi is convex (a
nondecreasing sequence of segment slopes). Breakpoints are fixed at u_k = k, so only the
slopes are learned (a simplification -- the original also learns breakpoint positions).

    psi_j(x, z) = psi_plus_j(x, z)    if z >= 0
                = psi_minus_j(x, -z)  if z < 0
    S(x, r)     = max_j psi_j(x, z_j),   z = Q(x)^T r  (rotated at residual_dim=2, else z = r)

Because psi is convex, strictly increasing and piecewise LINEAR, {z : psi(z) <= tau} is an
exact interval whose endpoint is found by inverting psi directly (a small closed-form
segment lookup) -- no root-finding solver needed, unlike the ICNN.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SplineScore(nn.Module):
    def __init__(self, input_dim: int, residual_dim: int = 1,
                 encoder_hidden_dims: list[int] | None = None, num_knots: int = 2,
                 min_slope: float = 1e-3) -> None:
        super().__init__()
        encoder_hidden_dims = encoder_hidden_dims or [64, 32]
        self.residual_dim = residual_dim
        self.output_dim = residual_dim
        self.num_segments = num_knots + 1
        self.min_slope = min_slope
        self.rotate = residual_dim == 2
        # fixed unit-spaced breakpoints u_1..u_{K-1}; the last segment extends to infinity
        self.register_buffer("breakpoints", torch.arange(1, self.num_segments, dtype=torch.float32))

        layers: list[nn.Module] = []
        prev = input_dim
        for h in encoder_hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.encoder = nn.Sequential(*layers)

        # K raw slope-increments per coordinate, one head for each sign
        self.head_plus = nn.Linear(prev, residual_dim * self.num_segments)
        self.head_minus = nn.Linear(prev, residual_dim * self.num_segments)
        if self.rotate:
            self.angle_head = nn.Linear(prev, 1)
            nn.init.zeros_(self.angle_head.weight)
            nn.init.zeros_(self.angle_head.bias)

    def _slopes(self, head: nn.Linear, h: torch.Tensor) -> torch.Tensor:
        """Cumulative nondecreasing slopes m_1 <= ... <= m_K, shape (batch, output_dim, K)."""
        raw = head(h).view(-1, self.output_dim, self.num_segments)
        increments = self.min_slope + F.softplus(raw)
        return torch.cumsum(increments, dim=-1)

    def _rotate_in(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        if not self.rotate:
            return r
        theta = self.angle_head(self.encoder(x)).squeeze(-1)
        cos, sin = torch.cos(theta), torch.sin(theta)
        return torch.stack([cos * r[:, 0] + sin * r[:, 1], -sin * r[:, 0] + cos * r[:, 1]], dim=1)

    def _psi(self, slopes: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """psi(z) for z >= 0, shape (batch, output_dim). `slopes` is (batch, output_dim, K)."""
        u = torch.cat([torch.zeros(1, device=z.device), self.breakpoints])  # (K,) breakpoint starts
        widths = torch.cat([self.breakpoints, torch.full((1,), float("inf"), device=z.device)]) - u
        raw = z.unsqueeze(-1) - u
        segment_len = torch.minimum(torch.maximum(raw, torch.zeros_like(raw)), widths)  # (batch, output_dim, K)
        return (slopes * segment_len).sum(dim=-1)

    def _psi_inverse(self, slopes: torch.Tensor, tau: float) -> torch.Tensor:
        """z >= 0 such that psi(z) = tau, shape (batch, output_dim). Closed-form: find which
        segment tau falls in from the cumulative value at each breakpoint, then invert linearly
        within that segment."""
        u = torch.cat([torch.zeros(1, device=slopes.device), self.breakpoints])
        widths = torch.cat([self.breakpoints, torch.full((1,), float("inf"), device=slopes.device)]) - u
        cum_value = torch.cumsum(slopes * widths, dim=-1)              # value AT the end of each segment
        value_before = cum_value - slopes * widths                     # value at the START of each segment
        in_segment = (tau > value_before) & (tau <= cum_value)
        in_segment[..., -1] |= tau > cum_value[..., -1]                # tau beyond the last breakpoint
        z_in_segment = u + (tau - value_before) / slopes
        return (z_in_segment * in_segment).sum(dim=-1)

    def score(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        z = self._rotate_in(x, r)
        h = self.encoder(x)
        m_plus, m_minus = self._slopes(self.head_plus, h), self._slopes(self.head_minus, h)
        per_coordinate = torch.where(z >= 0, self._psi(m_plus, z), self._psi(m_minus, -z))
        return per_coordinate.max(dim=-1, keepdim=True).values

    def membership(self, x: torch.Tensor, r: torch.Tensor, tau: float) -> torch.Tensor:
        return (self.score(x, r) <= tau).reshape(-1)

    def _side_lengths(self, x: torch.Tensor, tau: float) -> torch.Tensor:
        """Per-coordinate interval width tau maps to, shape (batch, output_dim): the sublevel
        set is an axis-aligned box (max of independent per-coordinate convex functions), whose
        side length is the sum of the two signed branches' inverses at tau."""
        h = self.encoder(x)
        m_plus, m_minus = self._slopes(self.head_plus, h), self._slopes(self.head_minus, h)
        return self._psi_inverse(m_plus, tau) + self._psi_inverse(m_minus, tau)

    def log_prediction_set_size(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        side = self._side_lengths(x, tau)
        return torch.log(side.clamp_min(1e-300)).sum(dim=1, keepdim=True)

    def prediction_set_size(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        return torch.exp(self.log_prediction_set_size(x, tau))

    def width(self, x: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        if self.residual_dim != 1:
            raise ValueError("width() is only defined for residual_dim=1")
        return self._side_lengths(x, tau)

    def interval(self, x: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
        if self.residual_dim != 1:
            raise ValueError("interval() is only defined for residual_dim=1")
        h = self.encoder(x)
        m_plus, m_minus = self._slopes(self.head_plus, h), self._slopes(self.head_minus, h)
        return -self._psi_inverse(m_minus, tau), self._psi_inverse(m_plus, tau)
