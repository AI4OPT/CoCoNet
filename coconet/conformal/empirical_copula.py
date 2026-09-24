"""Empirical-copula conformal prediction: an axis-aligned box with half-widths q_j sigma_j(x).

sigma_j(x) = exp(mu_j(x)) + beta, with mu an MLP fitted to log|r_j| on the training split. The
per-target thresholds q_j are read off the empirical copula of the normalized calibration scores
at one common marginal level u, the smallest level whose joint calibration coverage reaches
1 - alpha.
"""

import math

import numpy as np
import torch

from coconet.training.train_point import PointTrainConfig, train_mlp

MU_CLAMP = 30.0     # exp(30) ~ 1e13 keeps a pathological normalizer output finite


class EmpiricalCopula:
    def __init__(self, output_dim, beta, log_epsilon, normalizer):
        self.d = int(output_dim)
        self.beta = float(beta)
        self.log_epsilon = float(log_epsilon)
        self.normalizer_cfg = normalizer     # dict: hidden_dims, epochs, batch_size, learning_rate, weight_decay, patience

    def fit(self, x, r, x_val, r_val):
        nc = self.normalizer_cfg
        cfg = PointTrainConfig(hidden_dims=list(nc["hidden_dims"]), output_dim=self.d,
                               batch_size=int(nc["batch_size"]), learning_rate=float(nc["learning_rate"]),
                               weight_decay=float(nc["weight_decay"]), max_epochs=int(nc["epochs"]),
                               patience=int(nc["patience"]), seed=0)
        z = np.log(np.abs(np.asarray(r, dtype=np.float64)) + self.log_epsilon)
        z_val = np.log(np.abs(np.asarray(r_val, dtype=np.float64)) + self.log_epsilon)
        self.normalizer = train_mlp(np.asarray(x, dtype=np.float32), z.astype(np.float32),
                                    np.asarray(x_val, dtype=np.float32), z_val.astype(np.float32), cfg)
        return self

    def sigma(self, x):
        with torch.no_grad():
            mu = self.normalizer(torch.as_tensor(np.asarray(x, dtype=np.float32))).numpy().astype(np.float64)
        if not np.all(np.isfinite(mu)):
            raise RuntimeError("normalizer produced non-finite values")
        return np.exp(np.clip(mu, -MU_CLAMP, MU_CLAMP)) + self.beta

    def calibrate(self, x_cal, r_cal, alpha):
        A = np.abs(np.asarray(r_cal, dtype=np.float64)) / self.sigma(x_cal)      # (n, d)
        n = A.shape[0]
        sorted_cols = np.sort(A, axis=0)
        # pseudo-observations U_ij = F_j(A_ij) (max-rank convention for ties)
        U = np.stack([np.searchsorted(sorted_cols[:, j], A[:, j], side="right") / n
                      for j in range(self.d)], axis=1)
        # the smallest u with C(u, ..., u) = mean(max_j U_ij <= u) >= 1 - alpha
        M = np.sort(np.max(U, axis=1))
        k = max(1, min(math.ceil(n * (1.0 - alpha)), n))
        self.u = float(M[k - 1])
        # q_j = F_j^{-1}(u): the order statistic at index ceil(n u) - 1
        i = max(0, min(math.ceil(n * self.u) - 1, n - 1))
        self.q = sorted_cols[i, :].astype(float)
        return self

    def half_widths(self, x):
        return self.q[None, :] * self.sigma(x)

    def contains(self, x, r):
        return np.all(np.abs(np.asarray(r, dtype=np.float64)) <= self.half_widths(x), axis=1)

    def volume(self, x):
        return np.prod(2.0 * self.half_widths(x), axis=1)

    def n_params(self):
        return int(sum(p.numel() for p in self.normalizer.parameters()))
