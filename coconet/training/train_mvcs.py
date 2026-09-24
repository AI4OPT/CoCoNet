"""Training of the convex MVCS model: Adam (separate learning rates for the matrix net and p)
with a cosine schedule and gradient clipping; p is projected onto p >= 1 after every step. The
kept epoch is the one with the smallest validation objective, which also drives early stopping."""

import math
from dataclasses import dataclass

import numpy as np
import torch

from coconet.conformal.mvcs import MvcsModel, MvcsScore


@dataclass
class MvcsConfig:
    output_dim: int
    alpha: float
    hidden_dim: int
    n_layers: int
    ridge: float
    p_init: float
    matrix_learning_rate: float
    p_learning_rate: float
    batch_size: int
    max_epochs: int
    patience: int
    grad_clip_norm: float
    nonfinite_batch_tolerance: float
    seed: int
    convex: bool = True
    p_min: float = 0.1


def train_mvcs(x, r, x_val, r_val, cfg):
    """Returns (MvcsScore, selected epoch, validation objective, per-epoch history)."""
    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    torch.manual_seed(cfg.seed)
    model = MvcsModel(x.shape[1], cfg.output_dim, cfg.hidden_dim, cfg.n_layers, cfg.ridge, cfg.p_init, cfg.convex, cfg.p_min)
    model.set_context_normalization(x)
    x_t = torch.as_tensor(x, dtype=torch.float32)
    r_t = torch.as_tensor(np.asarray(r, dtype=float), dtype=torch.float32)
    x_val_t = torch.as_tensor(np.asarray(x_val, dtype=float).reshape(len(x_val), -1), dtype=torch.float32)
    r_val_t = torch.as_tensor(np.asarray(r_val, dtype=float), dtype=torch.float32)

    groups = [{"params": list(model.matrix_net.parameters()), "lr": cfg.matrix_learning_rate},
              {"params": [model.p], "lr": cfg.p_learning_rate}]
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.max_epochs, 1))
    clipped = [t for g in groups for t in g["params"]]
    n = len(x_t)
    min_batch = max(4, math.ceil(1.0 / cfg.alpha))     # a batch needs enough rows for the order statistic
    n_batches = n_nonfinite = 0
    best = None                # (objective, epoch, state)
    best_seen, since_improvement = float("inf"), 0
    history = []

    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(n)
        seen, running = 0, 0.0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            if len(idx) < min_batch:
                continue
            opt.zero_grad(set_to_none=True)
            loss = model.loss(x_t[idx], r_t[idx], cfg.alpha)
            n_batches += 1
            if not torch.isfinite(loss):
                n_nonfinite += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clipped, cfg.grad_clip_norm)
            opt.step()
            model.project_p()
            seen += len(idx)
            running += float(loss.detach()) * len(idx)
        if seen > 0:
            sched.step()

        model.eval()
        try:
            with torch.no_grad():
                objective = float(model.loss(x_val_t, r_val_t, cfg.alpha))
        except Exception:
            objective = float("nan")
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1),
                        "val_objective": objective, "p": float(model.exponent())})
        if np.isfinite(objective):
            if best is None or objective < best[0]:
                best = (objective, epoch, {k: v.detach().clone() for k, v in model.state_dict().items()})
            if objective < best_seen - 1e-9:
                best_seen, since_improvement = objective, 0
            else:
                since_improvement += 1
        else:
            since_improvement += 1
        if since_improvement >= cfg.patience:
            break

    if best is None:
        raise RuntimeError("no epoch had a finite validation objective")
    if n_nonfinite / max(n_batches, 1) > cfg.nonfinite_batch_tolerance:
        raise RuntimeError(f"{n_nonfinite}/{n_batches} training batches had a non-finite loss")
    objective, epoch, state = best
    model.load_state_dict(state)
    model.eval()
    return MvcsScore(model), epoch, objective, history
