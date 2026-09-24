"""Training loop of one learned score at fixed loss weights.

Adam with gradient clipping on `score_loss`. Early stopping watches the validation loss; the
kept epoch is chosen separately from calibrated validation coverage and size
(checkpoint_selection.py). At the end the model is calibrated on one half of the validation
split and scored on the other half; the lambda search and the tuners select on those numbers.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from coconet.conformal.calibration import conformal_threshold
from coconet.data.splits import train_val_split
from coconet.training.checkpoint_selection import CheckpointTracker, calibrated_coverage_and_size
from coconet.training.early_stopping import EarlyStopper
from coconet.training.loss import score_loss


class TrainingFailed(RuntimeError):
    pass


@dataclass
class TrainSettings:
    learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    grad_clip_norm: float
    coverage_tolerance: float
    checkpoint_every: int
    min_selection_epoch: int
    tau: float
    kappa_cov: float
    seed: int


@dataclass
class TrainedScore:
    model: torch.nn.Module
    selected_epoch: int
    selection_rule: str
    val_coverage: float      # calibrated coverage on the held-out half of the validation split
    val_size: float          # mean set size there
    history: list


def train_score(build_model, X, r, X_val, r_val, alpha, settings, lambda_miss, lambda_cov, device="cpu"):
    torch.manual_seed(settings.seed)
    model = build_model().to(device)
    loader = DataLoader(TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                                      torch.as_tensor(r, dtype=torch.float32).reshape(len(r), -1)),
                        batch_size=settings.batch_size, shuffle=True)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    r_val_t = torch.as_tensor(r_val, dtype=torch.float32, device=device).reshape(len(r_val), -1)
    # The validation split is halved once: inner calibration / inner evaluation.
    cal_idx, eval_idx = train_val_split(len(X_val), 0.5, seed=settings.seed)
    X_cal_t, r_cal_t = X_val_t[cal_idx], r_val_t[cal_idx]
    X_eval_t, r_eval_t = X_val_t[eval_idx], r_val_t[eval_idx]

    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    stopper = EarlyStopper(patience=settings.patience, min_delta=1e-6)
    tracker = CheckpointTracker(alpha, settings.coverage_tolerance, settings.min_selection_epoch)
    history = []

    def loss(x, rr):
        try:
            return score_loss(model, x, rr, alpha, settings.tau, settings.kappa_cov, lambda_miss, lambda_cov)
        except ValueError as exc:
            raise TrainingFailed(str(exc)) from exc

    if hasattr(model, "set_qmc_epoch"):          # ICNN, d > 2: new Sobol directions every epoch
        model.qmc_base_seed = settings.seed * 100_003
    for epoch in range(settings.max_epochs):
        if hasattr(model, "set_qmc_epoch"):
            model.set_qmc_epoch(epoch)
        model.train()
        train_losses = []
        for xb, rb in loader:
            xb, rb = xb.to(device), rb.to(device)
            optimizer.zero_grad()
            total, _, _, _ = loss(xb, rb)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip_norm)
            optimizer.step()
            train_losses.append(total.item())

        model.eval()
        with torch.no_grad():
            val_total, val_size, val_miss, val_cov = loss(X_val_t, r_val_t)
        val_total = float(val_total.item())
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_total,
               "val_size_term": float(val_size), "val_miss_term": float(val_miss),
               "val_coverage_term": float(val_cov)}
        if epoch % settings.checkpoint_every == 0 or epoch == settings.max_epochs - 1:
            coverage, size = calibrated_coverage_and_size(model, X_cal_t, r_cal_t, X_eval_t, r_eval_t, alpha)
            tracker.record(epoch, coverage, size, model)
            row.update({"calibrated_val_coverage": coverage, "calibrated_val_size": size})
        history.append(row)
        if stopper.step(val_total, epoch, model):
            break

    tracker.restore(model)
    model.eval()
    with torch.no_grad():
        q = conformal_threshold(model.score(X_cal_t, r_cal_t).cpu().numpy().reshape(-1), alpha)
        eval_scores = model.score(X_eval_t, r_eval_t).cpu().numpy().reshape(-1)
        val_coverage = float(np.mean(eval_scores <= q))
        val_size = float("inf") if math.isinf(q) else float(model.size(X_eval_t, q).mean().item())
    model.to("cpu")
    return TrainedScore(model, tracker.best_epoch, tracker.rule, val_coverage, val_size, history)
