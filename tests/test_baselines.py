import math

import torch

from learnscore.baseline_absolute_residual import AbsoluteResidual
from learnscore.baseline_ellipsoid import GlobalEllipsoid
from learnscore.baseline_empirical_copula import EmpiricalCopula


def test_absolute_residual_closed_form_volume_matches_hand_computed_box():
    torch.manual_seed(0)
    model = AbsoluteResidual(residual_dim=3)
    x = torch.randn(5, 2)
    tau = 2.0
    expected = (2.0 * tau) ** 3
    size = model.prediction_set_size(x, tau=tau)
    assert torch.allclose(size, torch.full_like(size, expected))
    log_size = model.log_prediction_set_size(x, tau=tau)
    assert torch.allclose(log_size, torch.full_like(log_size, math.log(expected)))


def test_absolute_residual_score_matches_max_abs_coordinate():
    torch.manual_seed(1)
    model = AbsoluteResidual(residual_dim=2)
    x = torch.randn(4, 2)
    r = torch.tensor([[1.0, -3.0], [0.5, 0.2], [-2.0, -2.5], [0.0, 0.0]])
    score = model.score(x, r).reshape(-1)
    expected = torch.tensor([3.0, 0.5, 2.5, 0.0])
    assert torch.allclose(score, expected)


def test_global_ellipsoid_score_is_nonnegative_and_volume_positive_finite():
    torch.manual_seed(2)
    r_score = torch.randn(200, 3).numpy()
    model = GlobalEllipsoid(residual_dim=3).fit(r_score)
    x = torch.randn(6, 2)
    r = torch.randn(6, 3)
    score = model.score(x, r)
    assert torch.all(score >= 0)

    log_v = model.log_prediction_set_size(x, tau=1.0)
    assert torch.all(torch.isfinite(log_v))
    v = model.prediction_set_size(x, tau=1.0)
    assert torch.all(v > 0)
    assert torch.all(torch.isfinite(v))


def test_empirical_copula_score_finite_and_set_size_positive_finite():
    torch.manual_seed(3)
    r_score = torch.randn(100, 2).numpy()
    model = EmpiricalCopula(residual_dim=2).fit(r_score)
    x = torch.randn(5, 2)
    r = torch.randn(5, 2)
    score = model.score(x, r)
    assert torch.all(torch.isfinite(score))
    assert torch.all(score >= 0) and torch.all(score <= 1)

    size = model.prediction_set_size(x, tau=0.9)
    assert torch.all(size > 0)
    assert torch.all(torch.isfinite(size))
