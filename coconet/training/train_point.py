"""Training loop for the MLP point predictor F(x) (and the baselines' auxiliary MLPs).

Trains on the internal D_pred-train split with Adam on the MSE, early-stops on the internal
D_pred-val split, and restores the best epoch. The returned model is frozen by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from coconet.models.mlp import MLP
from coconet.training.early_stopping import EarlyStopper
from coconet.utils import get_logger

logger = get_logger(__name__)


@dataclass
class PointTrainConfig:
    hidden_dims: list[int]
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 300
    patience: int = 20
    min_delta: float = 1e-5
    seed: int = 0
    output_dim: int = 1


@dataclass
class PointTrainResult:
    model: MLP
    train_loss_history: list[float] = field(default_factory=list)
    val_loss_history: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")


def train_point_predictor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: PointTrainConfig,
    device: str = "cpu",
) -> PointTrainResult:
    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        raise ValueError("Point predictor requires non-empty train and val sets")

    torch.manual_seed(config.seed)
    input_dim = X_train.shape[1]
    model = MLP(input_dim=input_dim, hidden_dims=config.hidden_dims, output_dim=config.output_dim).to(device)

    train_ds = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32).view(-1, config.output_dim),
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=device).view(-1, config.output_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    stopper = EarlyStopper(patience=config.patience, min_delta=config.min_delta)

    result = PointTrainResult(model=model)

    for epoch in range(config.max_epochs):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}: {loss.item()}")
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        train_loss = float(np.mean(epoch_losses))

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = float(loss_fn(val_pred, y_val_t).item())

        result.train_loss_history.append(train_loss)
        result.val_loss_history.append(val_loss)

        should_stop = stopper.step(val_loss, epoch, model)
        if should_stop:
            logger.info("Point predictor early stopping at epoch %d (best=%d)", epoch, stopper.best_epoch)
            break

    stopper.restore_best(model)
    result.best_epoch = stopper.best_epoch
    result.best_val_loss = stopper.best_value
    model.to("cpu")
    return result
