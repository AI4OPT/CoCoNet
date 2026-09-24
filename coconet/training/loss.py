"""Training loss of the learned scores, at a fixed threshold tau:

    L = (mean V)^(1/d)
        + lambda_miss * mean[ (s - tau)_+ / tau ]
        + lambda_cov  * ( mean sigmoid((tau - s) / T) - (1 - alpha) )^2,     T = 1 / kappa_cov

The two weights are gamma multiples of reference scales measured once on the freshly initialised
model (`reference_scales`), so one gamma grid means the same thing on every dataset.
"""

import math

import numpy as np
import torch

MISS_EPS = 1e-8
REF_EPS = 1e-8


def size_term(model, x, tau):
    log_v = model.log_size(x, tau).reshape(-1)
    log_mean = torch.logsumexp(log_v, dim=0) - math.log(log_v.shape[0])
    return torch.exp(log_mean / model.output_dim)


def miss_term(scores, tau):
    tau_t = torch.as_tensor(float(tau), dtype=scores.dtype, device=scores.device)
    return ((scores - tau_t) / (tau_t.abs() + MISS_EPS)).clamp_min(0.0).mean()


def coverage_term(scores, tau, kappa_cov, alpha):
    tau_t = torch.as_tensor(float(tau), dtype=scores.dtype, device=scores.device)
    temperature = torch.as_tensor(1.0 / kappa_cov, dtype=scores.dtype, device=scores.device)
    smooth = torch.sigmoid((tau_t - scores) / temperature).mean()
    return (smooth - (1.0 - alpha)) ** 2


def score_loss(model, x, r, alpha, tau, kappa_cov, lambda_miss, lambda_cov):
    """Returns (total, size, miss, coverage) for one batch."""
    size = size_term(model, x, tau)
    scores = model.score(x, r).reshape(-1)
    if not torch.isfinite(scores).all():
        raise ValueError("non-finite score")
    miss = miss_term(scores, tau)
    cov = coverage_term(scores, tau, kappa_cov, alpha)
    total = size + lambda_miss * miss + lambda_cov * cov
    if not torch.isfinite(total):
        raise ValueError("non-finite loss")
    return total, size, miss, cov


def reference_scales(model, x, r, alpha, tau, kappa_cov, batch_size, seed, device, n_batches=5):
    """lambda_miss_ref = |S| / M and lambda_cov_ref = |S| / C, with S, M, C the medians of the
    size, miss and coverage terms of the initialised model over a few training batches."""
    rng = np.random.default_rng(seed)
    n = len(x)
    take = min(batch_size, n)
    dtype = next(model.parameters()).dtype
    model.eval()

    def batch():
        idx = rng.choice(n, size=take, replace=False)
        xb = torch.as_tensor(x[idx], dtype=dtype, device=device)
        rb = torch.as_tensor(r[idx], dtype=dtype, device=device).reshape(len(idx), -1)
        return xb, rb

    sizes, misses, covs = [], [], []
    with torch.no_grad():
        for _ in range(n_batches):
            xb, rb = batch()
            sizes.append(float(size_term(model, xb, tau)))
            sb = model.score(xb, rb).reshape(-1)
            misses.append(float(miss_term(sb, tau)))
            covs.append(float(coverage_term(sb, tau, kappa_cov, alpha)))
    s_ref = abs(float(np.median(sizes)))
    m_ref = float(np.median(misses))
    c_ref = float(np.median(covs))

    # At a fixed threshold the initial set may contain every point (M = 0); then the miss is
    # measured at the batch's (1 - alpha) score quantile instead, so the scale stays finite.
    if m_ref <= REF_EPS:
        fallback = []
        with torch.no_grad():
            for _ in range(n_batches):
                xb, rb = batch()
                sb = model.score(xb, rb).reshape(-1)
                fallback.append(float(miss_term(sb, float(torch.quantile(sb, 1.0 - alpha)))))
        if float(np.median(fallback)) > REF_EPS:
            m_ref = float(np.median(fallback))
    return s_ref / max(m_ref, REF_EPS), s_ref / max(c_ref, REF_EPS)
