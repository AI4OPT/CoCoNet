"""geometry.set_boundary is generic -- it only needs `.score(x, r)` and `.residual_dim` -- so
one test exercises it against every method's model class, not just the ICNN."""

import torch

from learnscore.baseline_absolute_residual import AbsoluteResidual
from learnscore.baseline_ellipsoid import GlobalEllipsoid
from learnscore.baseline_empirical_copula import EmpiricalCopula
from learnscore.geometry import set_boundary
from learnscore.model import ICNN
from learnscore.model_asymmetric_linear import AsymmetricLinear
from learnscore.model_mvcs import MVCS
from learnscore.model_spline import SplineScore


def _check(model, residual_dim, tau=1.0):
    torch.manual_seed(0)
    x = torch.randn(3, 2)
    boundary = set_boundary(model, x, tau=tau, n_directions=8)
    assert boundary.shape[0] == 3 and boundary.shape[2] == residual_dim
    assert torch.all(torch.isfinite(boundary))


def test_boundary_for_icnn():
    _check(ICNN(input_dim=2, residual_dim=2), 2)


def test_boundary_for_asymmetric_linear():
    _check(AsymmetricLinear(input_dim=2, residual_dim=2), 2)


def test_boundary_for_spline():
    _check(SplineScore(input_dim=2, residual_dim=1), 1)


def test_boundary_for_mvcs():
    _check(MVCS(input_dim=2, residual_dim=3), 3)


def test_boundary_for_absolute_residual():
    _check(AbsoluteResidual(residual_dim=2), 2)


def test_boundary_for_global_ellipsoid():
    r_score = torch.randn(200, 2).numpy()
    _check(GlobalEllipsoid(residual_dim=2).fit(r_score), 2)


def test_boundary_for_empirical_copula():
    # score is a max of per-coordinate ECDFs, bounded in [0, 1]; tau must sit strictly below 1
    # for the sublevel set to have a finite boundary at all
    r_score = torch.randn(100, 2).numpy()
    _check(EmpiricalCopula(residual_dim=2).fit(r_score), 2, tau=0.5)
