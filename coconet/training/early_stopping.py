"""Patience-based early stopping on a validation metric (lower is better)."""

from __future__ import annotations

import copy

import torch
from torch import nn


class EarlyStopper:
    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float("inf")
        self.best_epoch = -1
        self.num_bad_epochs = 0
        self.best_state: dict | None = None

    def step(self, value: float, epoch: int, model: nn.Module) -> bool:
        """Record this epoch's value; return True when training should stop."""
        if value < self.best_value - self.min_delta:
            self.best_value = value
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience

    def restore_best(self, model: nn.Module) -> nn.Module:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model
