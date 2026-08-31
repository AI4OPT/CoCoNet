import torch

from learnscore.model_mvcs import MVCS


def test_score_and_volume_are_finite_and_positive():
    torch.manual_seed(0)
    model = MVCS(input_dim=3, residual_dim=4)
    x = torch.randn(5, 3)
    r = torch.randn(5, 4)
    score = model.score(x, r)
    log_v = model.log_prediction_set_size(x, tau=1.0)
    assert torch.all(torch.isfinite(score)) and torch.all(score >= 0)
    assert torch.all(torch.isfinite(log_v))


def test_volume_matches_the_unweighted_p_ball_at_identity_init():
    # matrix_head is zero-initialized, so A(x) = I and the set is exactly a p-norm ball
    torch.manual_seed(1)
    model = MVCS(input_dim=2, residual_dim=1, p_init=2.0)
    x = torch.randn(3, 2)
    log_v = model.log_prediction_set_size(x, tau=1.0)
    # residual_dim=1: any p-ball of radius 1 has length 2
    assert torch.allclose(log_v, torch.full_like(log_v, torch.log(torch.tensor(2.0))), atol=1e-4)


def test_membership_matches_score_threshold():
    torch.manual_seed(2)
    model = MVCS(input_dim=2, residual_dim=2)
    x = torch.randn(6, 2)
    r = torch.randn(6, 2)
    tau = 1.5
    member = model.membership(x, r, tau)
    score = model.score(x, r).reshape(-1)
    assert torch.equal(member, score <= tau)
