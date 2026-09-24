"""MLP regression with Adam on the MSE and early stopping on a validation split. Used for the
point predictor F(x) and for the auxiliary nets of two baselines."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from coconet.models.mlp import MLP
from coconet.training.early_stopping import EarlyStopper


@dataclass
class PointTrainConfig:
    hidden_dims: list
    output_dim: int
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    max_epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-5
    seed: int = 0


def train_mlp(X_train, y_train, X_val, y_val, cfg, device="cpu"):
    torch.manual_seed(cfg.seed)
    model = MLP(X_train.shape[1], list(cfg.hidden_dims), cfg.output_dim).to(device)
    train_ds = TensorDataset(torch.as_tensor(X_train, dtype=torch.float32),
                             torch.as_tensor(y_train, dtype=torch.float32).view(-1, cfg.output_dim))
    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=device).view(-1, cfg.output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()
    stopper = EarlyStopper(patience=cfg.patience, min_delta=cfg.min_delta)

    for epoch in range(cfg.max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(X_val_t), y_val_t).item())
        if stopper.step(val_loss, epoch, model):
            break
    stopper.restore_best(model)
    model.to("cpu")
    model.eval()
    return model
