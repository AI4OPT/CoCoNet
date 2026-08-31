"""The frozen mean predictor mu_hat(x), fit once on D_pred before any score training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from learnscore.early_stopping import EarlyStopper


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class PointConfig:
    hidden_dims: list[int]
    output_dim: int
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 300
    patience: int = 20
    seed: int = 0


def train_point_predictor(X_train, y_train, X_val, y_val, cfg: PointConfig, device: str = "cpu") -> MLP:
    torch.manual_seed(cfg.seed)
    model = MLP(X_train.shape[1], cfg.hidden_dims, cfg.output_dim).to(device)

    train_ds = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32).view(-1, cfg.output_dim))
    loader = DataLoader(train_ds, batch_size=min(cfg.batch_size, len(train_ds)), shuffle=True)

    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_val, dtype=torch.float32, device=device).view(-1, cfg.output_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()
    stopper = EarlyStopper(patience=cfg.patience)

    for epoch in range(cfg.max_epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        if stopper.step(val_loss, epoch, model):
            break
    stopper.restore_best(model)
    model.eval()
    return model


def predict(model: MLP, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    with torch.no_grad():
        x = torch.as_tensor(X, dtype=torch.float32, device=device)
        return model(x).cpu().numpy()
