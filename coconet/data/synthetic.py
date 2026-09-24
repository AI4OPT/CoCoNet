"""Two-dimensional synthetic regression with a known optimal 90% region.

    x ~ U(-1, 1)^2,   y = mu(x) + r,   mu(x) = (1.5 x1 + 0.5 sin(pi x2),  -x2 + 0.5 x1^2)
    r = A u,  A = Rot(theta) diag(sx, sy),  u ~ Uniform(B) with probability 0.9,
                                            u ~ Uniform(1.6 B minus B) with probability 0.1

B is a convex body centred on its area centroid. The core is 9 (1.6^2 - 1) = 14 times denser than
the shell, so A B is the unique smallest region holding 90% of the residuals, and its area is V*.
Every method is given the exact mean mu(x) as its point prediction.
"""

import math

import numpy as np
import torch
from matplotlib.path import Path as MplPath

INNER_FRACTION = 0.90
OUTER_SCALE = 1.6


def _ellipse(a, b, n):
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack([a * np.cos(t), b * np.sin(t)], axis=1)


# name -> (body vertices, rotation in degrees, axis scales)
SHAPES = {
    "ellipse": (_ellipse(1.32, 0.62, 720), 93.0, (1.10, 0.85)),
    "rectangle": (np.array([[-1.3, -0.65], [1.3, -0.65], [1.3, 0.65], [-1.3, 0.65]]), -30.0, (1.0, 1.0)),
    "triangle": (np.array([[-1.3, -0.7], [1.8, -0.4], [-0.2, 1.4]]), 20.0, (1.0, 0.8)),
    "trapezoid": (np.array([[-1.6, -1.0], [1.4, -1.0], [0.7, 1.0], [-0.4, 1.0]]), 140.0, (0.85, 0.85)),
}


def true_mu(x):
    x = np.atleast_2d(np.asarray(x, dtype=float))
    x1, x2 = x[:, 0], x[:, 1]
    return np.stack([1.5 * x1 + 0.5 * np.sin(np.pi * x2), -x2 + 0.5 * x1 ** 2], axis=1)


class ExactMean(torch.nn.Module):
    """mu(x) as a torch module: the point predictor handed to the learned scores."""

    def forward(self, x):
        x1, x2 = x[..., 0], x[..., 1]
        return torch.stack([1.5 * x1 + 0.5 * torch.sin(math.pi * x2), -x2 + 0.5 * x1 ** 2], dim=-1)


def polygon_area(v):
    x, y = v[:, 0], v[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_centroid(v):
    x, y = v[:, 0], v[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a = 0.5 * float(cross.sum())
    return np.array([float(((x + xn) * cross).sum()) / (6.0 * a), float(((y + yn) * cross).sum()) / (6.0 * a)])


def sample_polygon(v, n, rng):
    """n points uniform in a convex polygon (fan triangulation from the first vertex)."""
    tris = np.stack([np.stack([v[0], v[i], v[i + 1]]) for i in range(1, len(v) - 1)])
    w = np.array([polygon_area(t) for t in tris])
    t = tris[rng.choice(len(tris), size=n, p=w / w.sum())]
    a, b = rng.random(n), rng.random(n)
    flip = (a + b) > 1.0
    a, b = np.where(flip, 1.0 - a, a), np.where(flip, 1.0 - b, b)
    return t[:, 0] + a[:, None] * (t[:, 1] - t[:, 0]) + b[:, None] * (t[:, 2] - t[:, 0])


class Shape:
    def __init__(self, name):
        raw, rotation_deg, scale = SHAPES[name]
        raw = np.asarray(raw, dtype=float)
        self.name = name
        self.body = raw - polygon_centroid(raw)
        t = np.deg2rad(rotation_deg)
        rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        self.A = rot @ np.diag(np.asarray(scale, dtype=float))

    def optimal_region(self):
        """A B: the exact 90% minimum-area region in the residual frame."""
        return self.body @ self.A.T

    def outer_region(self):
        return OUTER_SCALE * self.optimal_region()

    def sample_residuals(self, n, rng):
        u = sample_polygon(self.body, n, rng)
        shell = np.flatnonzero(rng.random(n) >= INNER_FRACTION)
        if len(shell):
            core = MplPath(self.body)
            ring = np.empty((0, 2))
            while len(ring) < len(shell):          # uniform on 1.6 B minus B by rejection
                z = OUTER_SCALE * sample_polygon(self.body, 2 * len(shell) + 16, rng)
                ring = np.vstack([ring, z[~core.contains_points(z)]])
            u[shell] = ring[:len(shell)]
        return u @ self.A.T

    def generate(self, n, rng):
        x = rng.uniform(-1.0, 1.0, size=(n, 2))
        r = self.sample_residuals(n, rng)
        return x, true_mu(x) + r, r
