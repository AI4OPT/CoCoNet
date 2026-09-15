"""A prediction-set size function with a fixed direction set, for validation and selection.

The training hook `model.prediction_set_size` re-scrambles its QMC directions every epoch.
Anything that compares checkpoints or candidates to each other must instead use one fixed
direction set (`eval_qmc_seed`, `eval_volume_directions`), which is what this returns. Large
point sets are solved in blocks under a fixed memory budget.
"""

from __future__ import annotations

from typing import Callable

import torch

MAX_RAY_ELEMENTS_PER_CALL = 1 << 23
MAX_RAY_ROWS_PER_CALL = 640 * 1024


def _in_blocks(one_block, x: torch.Tensor, tau, n_directions: int, d: int) -> torch.Tensor:
    m = max(1, int(n_directions))
    per = max(1, min(MAX_RAY_ELEMENTS_PER_CALL // (m * max(1, int(d))), MAX_RAY_ROWS_PER_CALL // m))
    n = x.shape[0]
    if n <= per:
        return one_block(x, tau)
    per_point_tau = torch.is_tensor(tau) and tau.numel() > 1
    return torch.cat([one_block(x[i:i + per], tau[i:i + per] if per_point_tau else tau)
                      for i in range(0, n, per)], dim=0)


def is_qmc_volume_model(model) -> bool:
    return (
        hasattr(model, "volume")
        and hasattr(model, "eval_qmc_seed")
        and int(getattr(model, "residual_dim", 1)) > 2
    )


def deterministic_size_fn(
    model, n_directions: int | None = None, seed: int | None = None,
) -> Callable[..., torch.Tensor]:
    if is_qmc_volume_model(model):
        fixed_seed = int(model.eval_qmc_seed if seed is None else seed)
        M = int(model.eval_volume_directions if n_directions is None else n_directions)

        if getattr(model, "volume_whiten", False):
            d = int(model.residual_dim)

            def _whitened_block(xb: torch.Tensor, tb) -> torch.Tensor:
                return torch.exp(model._log_volume_core_whitened(
                    xb, tb, M, fixed_seed, update=False))

            def _fixed_whitened_size(x: torch.Tensor, tau) -> torch.Tensor:
                return _in_blocks(_whitened_block, x, tau, M, d)

            return _fixed_whitened_size

        d = int(model.residual_dim)

        def _qmc_block(xb: torch.Tensor, tb) -> torch.Tensor:
            return model.volume(xb, tau=tb, n_directions=M, seed=fixed_seed)

        def _fixed_qmc_size(x: torch.Tensor, tau) -> torch.Tensor:
            return _in_blocks(_qmc_block, x, tau, M, d)

        return _fixed_qmc_size

    if hasattr(model, "area") and int(getattr(model, "residual_dim", 1)) == 2 \
            and hasattr(model, "eval_area_directions"):
        M = int(model.eval_area_directions if n_directions is None else n_directions)

        def _fixed_polar_size(x: torch.Tensor, tau) -> torch.Tensor:
            return model.area(x, tau=tau, n_directions=M)

        return _fixed_polar_size

    return model.prediction_set_size if hasattr(model, "prediction_set_size") else model.width
