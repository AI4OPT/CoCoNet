"""Ellipsoidal conformity scores s(x, r) = sqrt(r^T Sigma(x)^-1 r).

    global   Sigma(x) = covariance of the training residuals
    local    Sigma(x) = w Cov_local(x) + (1 - w) Sigma_global, Cov_local from the residuals of
             the K nearest training inputs

The calibrated set is {r : s(x, r) <= q} with volume q^d sqrt(det Sigma) V_d. Covariances are
symmetrized, ridged and Cholesky-factorized; if a factorization fails the ridge of that matrix
alone is grown until it succeeds.
"""

import math

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _factor_one(sigma, ridge, eye, max_retries=8, growth=10.0):
    current = float(ridge)
    for _ in range(max_retries + 1):
        try:
            chol = np.linalg.cholesky(sigma + current * eye)
        except np.linalg.LinAlgError:
            current = current * growth if current > 0 else 1e-10
            continue
        if np.all(np.isfinite(chol)):
            return chol
        current = current * growth if current > 0 else 1e-10
    raise RuntimeError("covariance could not be factorized even after growing the ridge")


def cholesky_factor(sigma, ridge):
    """Cholesky factor of one (d, d) matrix or a batch (n, d, d) after symmetrization + ridge."""
    sigma = np.asarray(sigma, dtype=float)
    sigma = 0.5 * (sigma + np.swapaxes(sigma, -1, -2))
    eye = np.eye(sigma.shape[-1])
    if sigma.ndim == 2:
        return _factor_one(sigma, ridge, eye)
    try:
        chol = np.linalg.cholesky(sigma + ridge * eye)
        if np.all(np.isfinite(chol)):
            return chol
    except np.linalg.LinAlgError:
        pass
    return np.stack([_factor_one(s, ridge, eye) for s in sigma])


def _mahalanobis(r, chol):
    """sqrt(r^T Sigma^-1 r) through the factor: solve L z = r, take ||z||."""
    r = np.asarray(r, dtype=float)
    if chol.ndim == 2:
        z = np.linalg.solve(chol, r.T).T
    else:
        z = np.linalg.solve(chol, r[:, :, None])[:, :, 0]
    return np.sqrt(np.maximum((z ** 2).sum(axis=1), 0.0))


def _volume(q, chol, d):
    log_det = 2.0 * np.log(np.diagonal(chol, axis1=-2, axis2=-1)).sum(axis=-1)
    log_unit_ball = (d / 2.0) * math.log(math.pi) - math.lgamma(d / 2.0 + 1.0)
    return np.exp(d * math.log(max(q, 1e-300)) + 0.5 * log_det + log_unit_ball)


class GlobalEllipsoid:
    def __init__(self, output_dim, ridge=1e-6):
        self.d = int(output_dim)
        self.ridge = float(ridge)

    def fit(self, x, r):
        r = np.asarray(r, dtype=float).reshape(len(r), -1)
        self.sigma = np.cov(r, rowvar=False).reshape(self.d, self.d)
        self.chol = cholesky_factor(self.sigma, self.ridge)
        return self

    def score(self, x, r):
        return _mahalanobis(np.asarray(r, dtype=float).reshape(len(r), -1), self.chol)

    def volume(self, x, q):
        return np.full(len(x), float(_volume(q, self.chol, self.d)))

    def n_params(self):
        return self.d * (self.d + 1) // 2


class LocalEllipsoid(GlobalEllipsoid):
    def __init__(self, output_dim, ridge=1e-6, local_weight=0.95, neighbor_fraction=0.05):
        super().__init__(output_dim, ridge)
        self.local_weight = float(local_weight)
        self.neighbor_fraction = float(neighbor_fraction)

    def fit(self, x, r):
        super().fit(x, r)
        x = np.asarray(x, dtype=float)
        self.x_mean = x.mean(axis=0)
        scale = x.std(axis=0)
        self.x_scale = np.where(scale > 1e-12, scale, 1.0)
        self.r_train = np.asarray(r, dtype=float).reshape(len(r), -1)
        self.k = min(max(2 * self.d + 5, math.ceil(self.neighbor_fraction * len(x))), len(x))
        self.nn = NearestNeighbors(n_neighbors=self.k).fit((x - self.x_mean) / self.x_scale)
        return self

    def _chol(self, x):
        x = (np.asarray(x, dtype=float) - self.x_mean) / self.x_scale
        idx = self.nn.kneighbors(x, return_distance=False)          # (n, K)
        neigh = self.r_train[idx]                                    # (n, K, d)
        centered = neigh - neigh.mean(axis=1, keepdims=True)
        cov_local = np.einsum("nkd,nke->nde", centered, centered) / max(self.k - 1, 1)
        sigma = self.local_weight * cov_local + (1.0 - self.local_weight) * self.sigma[None, :, :]
        return cholesky_factor(sigma, self.ridge)

    def score(self, x, r):
        return _mahalanobis(np.asarray(r, dtype=float).reshape(len(r), -1), self._chol(x))

    def volume(self, x, q):
        return _volume(q, self._chol(x), self.d)
