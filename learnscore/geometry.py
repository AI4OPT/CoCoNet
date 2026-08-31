"""The actual shape of a calibrated prediction set, not just its coverage/volume.

For a test point x with center y_hat(x), the calibrated set is {y_hat + offset : score(x,
offset) <= q_hat}. This traces its boundary as a cloud of offset points along a fixed set of
directions from the center -- exact at d=1 (the two interval endpoints) and d=2 (a polygon,
directions evenly spaced by angle), and a QMC sample of the boundary at d>2, where a convex
region can't be written down exactly. Works for any model with a `.score(x, r)` method, via
the same generic root-finder the ICNN's own volume estimate uses (see volume.py) -- so it
applies unchanged to every method in this repo, closed-form or not.
"""

from __future__ import annotations

import math

import torch

from learnscore import volume


def _directions(d: int, n_directions: int, seed: int, dtype, device) -> torch.Tensor:
    if d == 1:
        return torch.tensor([[1.0], [-1.0]], dtype=dtype, device=device)
    if d == 2:
        thetas = torch.linspace(0.0, 2 * math.pi, n_directions + 1, dtype=dtype, device=device)[:-1]
        return torch.stack([torch.cos(thetas), torch.sin(thetas)], dim=1)
    return volume.qmc_sphere_directions(d, n_directions, seed, dtype, device)


def set_boundary(model, x: torch.Tensor, tau: float, n_directions: int = 64,
                  seed: int = 0) -> torch.Tensor:
    """Offsets from the center to the set's boundary, shape (batch, n_directions_used, d) --
    add the center (the point prediction) to get the boundary in response space."""
    d = model.residual_dim
    directions = _directions(d, n_directions, seed, x.dtype, x.device)
    m = directions.shape[0]
    x_exp = x.repeat_interleave(m, dim=0)
    dir_exp = directions.repeat(x.shape[0], 1)
    with torch.no_grad():
        rho = volume.solve_rays(model, x_exp, dir_exp, tau).view(x.shape[0], m, 1)
    return rho * directions.unsqueeze(0)
