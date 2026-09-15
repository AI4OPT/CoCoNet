"""L2 penalty over the non-bias parameters of a score model."""

from __future__ import annotations

import torch
from torch import nn


def _is_excluded(name: str, extra_excluded_names: frozenset[str]) -> bool:
    last = name.rsplit(".", 1)[-1]
    return last == "bias" or last in extra_excluded_names or name in extra_excluded_names


ENCODER_PREFIXES: tuple[str, ...] = ("encoder.",)


def is_encoder_param(name: str) -> bool:
    return any(name.startswith(pfx) for pfx in ENCODER_PREFIXES)


def split_regularized_params(
    model: nn.Module, extra_excluded_names: frozenset[str] = frozenset(),
    encoder_only: bool = False,
) -> tuple[list[str], list[str]]:
    """(regularized_names, excluded_names): biases (and any name in `extra_excluded_names`) are
    excluded; `encoder_only=True` additionally excludes every non-encoder parameter."""
    regularized, excluded = [], []
    for name, _p in model.named_parameters():
        drop = _is_excluded(name, extra_excluded_names) or (
            encoder_only and not is_encoder_param(name))
        (excluded if drop else regularized).append(name)
    return regularized, excluded


def l2_penalty(model: nn.Module, extra_excluded_names: frozenset[str] = frozenset(),
               encoder_only: bool = False) -> torch.Tensor:
    """Sum of squares over every parameter that `split_regularized_params` keeps."""
    excluded = set(split_regularized_params(model, extra_excluded_names, encoder_only)[1])
    total = sum((p**2).sum() for name, p in model.named_parameters() if name not in excluded)
    if isinstance(total, int):
        return torch.zeros((), dtype=torch.get_default_dtype())
    return total
