"""Ellipsoidal conformity scores (Messoudi et al., "Ellipsoidal conformal inference for
multi-target regression"): the "G-Ellip." and "L-Ellip." baselines.

    s(x, r) = sqrt( r^T Sigma(x)^-1 r )

global_ellipsoid     Sigma(x) = Sigma_global, the covariance of the D_score residuals.
local_ellipsoid_knn  Sigma(x) = w Cov_local(x) + (1 - w) Sigma_global, with Cov_local(x) the
                     covariance of the residuals of x's K nearest D_score contexts.

Calibrated set: { r : r^T Sigma(x)^-1 r <= q_hat^2 }, volume q_hat^d sqrt(det Sigma) V_d.
Every covariance is symmetrized, ridged and Cholesky-factorized; a failed factorization retries
with a larger ridge, resolved per matrix so one row never changes another row's score.
"""

from __future__ import annotations

import math

import numpy as np

DISTANCE_METRICS = ("euclidean", "manhattan", "chebyshev")


class EllipsoidError(ValueError):
    pass


def resolve_num_neighbors(
    n_train: int, output_dim: int, neighbor_fraction: float, num_neighbors: int | None,
) -> int:
    """K = num_neighbors if given, else max(2 d + 5, ceil(neighbor_fraction n_train)), at most n_train."""
    if num_neighbors is not None:
        k = int(num_neighbors)
        if k < 2:
            raise EllipsoidError(f"ellipsoid_num_neighbors must be >= 2, got {k}")
    else:
        if not (0.0 < neighbor_fraction <= 1.0):
            raise EllipsoidError(f"ellipsoid_neighbor_fraction must be in (0, 1], got {neighbor_fraction}")
        k = max(2 * output_dim + 5, int(math.ceil(neighbor_fraction * n_train)))
    return int(min(k, n_train))


def _resolve_ridge_and_factor(
    sigma: np.ndarray, ridge: float, max_retries: int, growth: float, eye: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    """Grow the ridge of one symmetrized (d, d) matrix until Cholesky succeeds."""
    current = float(ridge)
    for attempt in range(max_retries + 1):
        try:
            chol = np.linalg.cholesky(sigma + current * eye)
        except np.linalg.LinAlgError:
            current = current * growth if current > 0 else 1e-10
            continue
        if np.all(np.isfinite(chol)):
            return chol, current, attempt
        current = current * growth if current > 0 else 1e-10
    raise EllipsoidError(
        f"Covariance stabilization failed after {max_retries} ridge escalations "
        f"(final ridge {current:.3g}); the residual covariance may be degenerate."
    )


def _regularization_diagnostics(sigma: np.ndarray, ridge, eye: np.ndarray) -> dict:
    ridge_arr = np.asarray(ridge, dtype=float)
    broadcast = ridge_arr.reshape(ridge_arr.shape + (1, 1)) if ridge_arr.ndim else ridge_arr
    before = np.linalg.eigvalsh(sigma)
    after = np.linalg.eigvalsh(sigma + broadcast * eye)
    min_before = np.min(before, axis=-1)
    max_before = np.max(np.abs(before), axis=-1)
    min_after = np.min(after, axis=-1)
    max_after = np.max(np.abs(after), axis=-1)
    cond_before = max_before / np.maximum(np.abs(min_before), 1e-300)
    cond_after = max_after / np.maximum(np.abs(min_after), 1e-300)
    cast = float if before.ndim == 1 else (lambda v: np.asarray(v, dtype=float))
    return {
        "minimum_eigenvalue_before_regularization": cast(min_before),
        "minimum_eigenvalue_after_regularization": cast(min_after),
        "condition_number_before_regularization": cast(cond_before),
        "condition_number_after_regularization": cast(cond_after),
        "regularization_strength": cast(ridge_arr),
    }


def _stabilize_and_factor(
    sigma: np.ndarray, ridge: float, max_retries: int = 8, growth: float = 10.0,
) -> tuple[np.ndarray, float, int, dict]:
    """Symmetrize -> ridge -> Cholesky, for one (d, d) matrix or a batch (n, d, d). The batched
    call is only a fast path at the base ridge; if any matrix fails, each row resolves its own."""
    sigma = np.asarray(sigma, dtype=float)
    d = sigma.shape[-1]
    sigma = 0.5 * (sigma + np.swapaxes(sigma, -1, -2))
    eye = np.eye(d)

    if sigma.ndim == 2:
        chol, final_ridge, retries = _resolve_ridge_and_factor(sigma, ridge, max_retries, growth, eye)
        return chol, final_ridge, retries, _regularization_diagnostics(sigma, final_ridge, eye)

    n = sigma.shape[0]
    base = float(ridge)
    ridges = np.full(n, base, dtype=float)
    retries = np.zeros(n, dtype=int)
    chol = None
    try:
        candidate = np.linalg.cholesky(sigma + base * eye)
        if np.all(np.isfinite(candidate)):
            chol = candidate
    except np.linalg.LinAlgError:
        chol = None
    if chol is None:
        chol = np.empty_like(sigma)
        for i in range(n):
            chol[i], ridges[i], retries[i] = _resolve_ridge_and_factor(
                sigma[i], base, max_retries, growth, eye,
            )
    return chol, ridges, retries, _regularization_diagnostics(sigma, ridges, eye)


def _mahalanobis_sq(r: np.ndarray, chol: np.ndarray) -> np.ndarray:
    """r^T Sigma^-1 r through the Cholesky factor: solve L z = r, return ||z||^2 (clamped at 0)."""
    r = np.asarray(r, dtype=float)
    if chol.ndim == 2:
        z = np.linalg.solve(chol, r.T).T  # (n, d)
    else:
        z = np.linalg.solve(chol, r[:, :, None])[:, :, 0]  # (n, d)
    return np.maximum((z ** 2).sum(axis=1), 0.0)


def _log_det_from_chol(chol: np.ndarray) -> np.ndarray:
    diag = np.diagonal(chol, axis1=-2, axis2=-1)
    return 2.0 * np.log(diag).sum(axis=-1)


def log_ellipsoid_volume(q_hat: float, log_det: np.ndarray, output_dim: int) -> np.ndarray:
    d = int(output_dim)
    log_unit_ball = (d / 2.0) * math.log(math.pi) - math.lgamma(d / 2.0 + 1.0)
    return d * math.log(max(q_hat, 1e-300)) + 0.5 * np.asarray(log_det, dtype=float) + log_unit_ball


def ellipsoid_volume(q_hat: float, log_det: np.ndarray, output_dim: int) -> np.ndarray:
    """V = q_hat^d sqrt(det Sigma) pi^(d/2) / Gamma(d/2 + 1), computed in log space."""
    return np.exp(log_ellipsoid_volume(q_hat, log_det, output_dim))


class _BaseEllipsoidScore:
    def __init__(self, output_dim: int, covariance_ridge: float = 1e-6) -> None:
        if output_dim < 1:
            raise EllipsoidError(f"output_dim must be >= 1, got {output_dim}")
        self.output_dim = int(output_dim)
        self.covariance_ridge = float(covariance_ridge)
        self.sigma_global: np.ndarray | None = None
        self.stabilization_retries = 0
        self.final_ridge = float(covariance_ridge)
        self.regularization_diagnostics: dict = {}

    def _fit_global(self, residuals: np.ndarray) -> None:
        r = np.asarray(residuals, dtype=float).reshape(len(residuals), -1)
        if r.shape[1] != self.output_dim:
            raise EllipsoidError(f"residuals have {r.shape[1]} columns, expected output_dim={self.output_dim}")
        if len(r) < 2:
            raise EllipsoidError(f"need >= 2 training residuals to estimate a covariance, got {len(r)}")
        sigma = np.cov(r, rowvar=False).reshape(self.output_dim, self.output_dim)
        chol, ridge, retries, diagnostics = _stabilize_and_factor(sigma, self.covariance_ridge)
        self.sigma_global = sigma
        self._chol_global = chol
        self.final_ridge = ridge
        self.stabilization_retries = int(retries)
        self.regularization_diagnostics = diagnostics

    def area(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def n_params(self) -> int:
        d = self.output_dim
        return d * (d + 1) // 2


class GlobalEllipsoidScore(_BaseEllipsoidScore):
    """One covariance for every context."""

    method_name = "global_ellipsoid"

    def fit(self, x_train: np.ndarray, residuals_train: np.ndarray) -> "GlobalEllipsoidScore":
        self._fit_global(residuals_train)
        return self

    def _chol_for(self, x: np.ndarray) -> np.ndarray:
        if self.sigma_global is None:
            raise EllipsoidError("GlobalEllipsoidScore.fit must be called before scoring")
        return self._chol_global

    def score(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float).reshape(len(r), -1)
        return np.sqrt(_mahalanobis_sq(r, self._chol_for(x)))

    def covariance(self, x: np.ndarray) -> np.ndarray:
        n = len(np.asarray(x))
        return np.repeat(self.sigma_global[None, :, :], n, axis=0)

    def area(self, x: np.ndarray, q_hat: float) -> np.ndarray:
        n = len(np.asarray(x))
        log_det = _log_det_from_chol(self._chol_global)
        return np.full(n, float(ellipsoid_volume(q_hat, log_det, self.output_dim)))

    def log_volume(self, x: np.ndarray, q_hat: float) -> np.ndarray:
        n = len(np.asarray(x))
        log_det = _log_det_from_chol(self._chol_global)
        return np.full(n, float(log_ellipsoid_volume(q_hat, log_det, self.output_dim)))


class LocalKnnEllipsoidScore(_BaseEllipsoidScore):
    """Covariance of the K nearest D_score contexts, shrunk toward the global covariance."""

    method_name = "local_ellipsoid_knn"

    def __init__(
        self, output_dim: int, covariance_ridge: float = 1e-6, local_weight: float = 0.95,
        neighbor_fraction: float = 0.05, num_neighbors: int | None = None,
        distance_metric: str = "euclidean", standardize_context: bool = True,
    ) -> None:
        super().__init__(output_dim=output_dim, covariance_ridge=covariance_ridge)
        if not (0.0 <= local_weight <= 1.0):
            raise EllipsoidError(f"ellipsoid_local_weight must be in [0, 1], got {local_weight}")
        if distance_metric not in DISTANCE_METRICS:
            raise EllipsoidError(f"ellipsoid_distance_metric must be one of {DISTANCE_METRICS}, got {distance_metric!r}")
        self.local_weight = float(local_weight)
        self.neighbor_fraction = float(neighbor_fraction)
        self.num_neighbors = num_neighbors
        self.distance_metric = distance_metric
        self.standardize_context = bool(standardize_context)
        self.k: int | None = None
        self._scaler_mean: np.ndarray | None = None
        self._scaler_scale: np.ndarray | None = None
        self._nn = None
        self._r_train: np.ndarray | None = None

    def _transform_context(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if not self.standardize_context:
            return x
        return (x - self._scaler_mean) / self._scaler_scale

    def fit(self, x_train: np.ndarray, residuals_train: np.ndarray) -> "LocalKnnEllipsoidScore":
        from sklearn.neighbors import NearestNeighbors

        self._fit_global(residuals_train)
        x = np.asarray(x_train, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        self._scaler_mean = x.mean(axis=0)
        scale = x.std(axis=0)
        self._scaler_scale = np.where(scale > 1e-12, scale, 1.0)

        self._r_train = np.asarray(residuals_train, dtype=float).reshape(len(residuals_train), -1)
        self.k = resolve_num_neighbors(len(x), self.output_dim, self.neighbor_fraction, self.num_neighbors)
        self._nn = NearestNeighbors(n_neighbors=self.k, metric=self.distance_metric)
        self._nn.fit(self._transform_context(x))
        return self

    def covariance(self, x: np.ndarray) -> np.ndarray:
        """(n, d, d) shrunk local covariances."""
        if self._nn is None:
            raise EllipsoidError("LocalKnnEllipsoidScore.fit must be called before scoring")
        idx = self._nn.kneighbors(self._transform_context(x), return_distance=False)  # (n, K)
        neigh = self._r_train[idx]  # (n, K, d)
        centered = neigh - neigh.mean(axis=1, keepdims=True)
        denom = max(self.k - 1, 1)
        cov_local = np.einsum("nkd,nke->nde", centered, centered) / denom
        return self.local_weight * cov_local + (1.0 - self.local_weight) * self.sigma_global[None, :, :]

    def _chol_for(self, x: np.ndarray) -> np.ndarray:
        chol, _ridge, _retries, _diagnostics = _stabilize_and_factor(
            self.covariance(x), self.covariance_ridge,
        )
        return chol

    def score(self, x: np.ndarray, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float).reshape(len(r), -1)
        return np.sqrt(_mahalanobis_sq(r, self._chol_for(x)))

    def area(self, x: np.ndarray, q_hat: float) -> np.ndarray:
        log_det = _log_det_from_chol(self._chol_for(x))
        return ellipsoid_volume(q_hat, log_det, self.output_dim)

    def log_volume(self, x: np.ndarray, q_hat: float) -> np.ndarray:
        log_det = _log_det_from_chol(self._chol_for(x))
        return log_ellipsoid_volume(q_hat, log_det, self.output_dim)
