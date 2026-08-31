import math

import torch

from learnscore.model import ICNN


def test_score_is_zero_at_the_origin():
    torch.manual_seed(0)
    model = ICNN(input_dim=3, residual_dim=2)
    x = torch.randn(5, 3)
    r0 = torch.zeros(5, 2)
    score = model.score(x, r0)
    assert torch.allclose(score, torch.zeros_like(score), atol=1e-5)


def test_score_is_convex_in_r():
    torch.manual_seed(1)
    model = ICNN(input_dim=2, residual_dim=1)
    x = torch.randn(4, 2)
    a = torch.randn(4, 1)
    b = torch.randn(4, 1)
    mid = 0.5 * (a + b)
    s_a, s_b, s_mid = model.score(x, a), model.score(x, b), model.score(x, mid)
    assert torch.all(s_mid <= 0.5 * (s_a + s_b) + 1e-4)


def test_width_is_positive_and_symmetric_root_finding_agrees_with_interval():
    torch.manual_seed(2)
    model = ICNN(input_dim=2, residual_dim=1)
    x = torch.randn(6, 2)
    lower, upper = model.interval(x, tau=1.0)
    width = model.width(x, tau=1.0)
    assert torch.all(width > 0)
    assert torch.allclose(upper - lower, width, atol=1e-4)


def test_area_is_positive_for_d2():
    torch.manual_seed(3)
    model = ICNN(input_dim=2, residual_dim=2, train_area_directions=64)
    x = torch.randn(4, 2)
    area = model.area(x, tau=1.0)
    assert torch.all(area > 0)
    assert torch.all(torch.isfinite(area))


def test_log_volume_finite_for_d_greater_than_2():
    torch.manual_seed(4)
    model = ICNN(input_dim=3, residual_dim=4, train_volume_directions=64, eval_volume_directions=64)
    x = torch.randn(3, 3)
    log_v = model.log_volume(x, tau=1.0)
    assert torch.all(torch.isfinite(log_v))


def test_membership_matches_score_threshold():
    torch.manual_seed(5)
    model = ICNN(input_dim=2, residual_dim=1)
    x = torch.randn(10, 2)
    r = torch.randn(10, 1)
    tau = 1.0
    member = model.membership(x, r, tau)
    score = model.score(x, r).reshape(-1)
    assert torch.equal(member, score <= tau)
