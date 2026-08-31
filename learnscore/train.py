"""Train the ICNN score model.

Adam + gradient clipping on `prediction_set_loss`. Early stopping tracks the surrogate
validation loss only to decide *when to stop*; which epoch's weights are actually kept is
decided separately, by `CheckpointTracker`, on real calibrated coverage/area (see
checkpoint_selection.py) -- the surrogate loss is not a reliable proxy for "the model that
would deploy well".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from learnscore.checkpoint_selection import CheckpointTracker
from learnscore.early_stopping import EarlyStopper
from learnscore.loss import LossConfig, prediction_set_loss


@dataclass
class TrainConfig:
    learning_rate: float
    batch_size: int
    max_epochs: int = 300
    patience: int = 80
    grad_clip_norm: float = 5.0
    checkpoint_every: int = 5
    coverage_tolerance: float = 0.02
    min_selection_epoch: int = 10
    seed: int = 0


@dataclass
class TrainResult:
    model: torch.nn.Module
    selected_epoch: int
    train_loss_history: list[float]
    val_loss_history: list[float]


def _to_tensor(a: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(a, dtype=torch.float32, device=device)


def _inner_split(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Halve validation indices for the checkpoint tracker's inner calibrate/evaluate split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    half = n // 2
    return perm[:half], perm[half:]


def train_score_model(model_factory, X_train, r_train, X_val, r_val, loss_cfg: LossConfig,
                       train_cfg: TrainConfig, device: str = "cpu") -> TrainResult:
    torch.manual_seed(train_cfg.seed)
    model = model_factory().to(device)

    train_ds = TensorDataset(_to_tensor(X_train, device), _to_tensor(r_train, device))
    loader = DataLoader(train_ds, batch_size=min(train_cfg.batch_size, len(train_ds)), shuffle=True)
    X_val_t, r_val_t = _to_tensor(X_val, device), _to_tensor(r_val, device)

    cal_idx, eval_idx = _inner_split(len(X_val), train_cfg.seed)
    x_inner_cal, r_inner_cal = X_val_t[cal_idx], r_val_t[cal_idx]
    x_inner_eval, r_inner_eval = X_val_t[eval_idx], r_val_t[eval_idx]

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate)
    stopper = EarlyStopper(patience=train_cfg.patience)
    tracker = CheckpointTracker(alpha=loss_cfg.alpha, coverage_tolerance=train_cfg.coverage_tolerance,
                                 min_selection_epoch=train_cfg.min_selection_epoch)

    train_loss_history, val_loss_history = [], []
    for epoch in range(train_cfg.max_epochs):
        if hasattr(model, "set_qmc_epoch"):
            model.set_qmc_epoch(epoch)

        model.train()
        batch_losses = []
        for xb, rb in loader:
            optimizer.zero_grad()
            loss = prediction_set_loss(model, xb, rb, loss_cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip_norm)
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss_history.append(float(np.mean(batch_losses)))

        model.eval()
        with torch.no_grad():
            val_loss = prediction_set_loss(model, X_val_t, r_val_t, loss_cfg).item()
        val_loss_history.append(val_loss)

        if epoch % train_cfg.checkpoint_every == 0 or epoch == train_cfg.max_epochs - 1:
            tracker.record(model, x_inner_cal, r_inner_cal, x_inner_eval, r_inner_eval, epoch)

        if stopper.step(val_loss, epoch, model):
            break

    if not tracker.restore_best(model):
        stopper.restore_best(model)
        selected_epoch = stopper.best_epoch
    else:
        selected_epoch = tracker.best.epoch

    model.eval()
    return TrainResult(model=model, selected_epoch=selected_epoch,
                        train_loss_history=train_loss_history, val_loss_history=val_loss_history)
