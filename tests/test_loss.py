import torch

from learnscore.loss import LossConfig, coverage_term, miss_term, prediction_set_loss
from learnscore.model import ICNN


def test_miss_term_is_zero_when_everything_is_inside():
    scores = torch.tensor([0.1, 0.2, 0.3])
    tau = torch.tensor(1.0)
    assert miss_term(scores, tau, eps=1e-8).item() == 0.0


def test_miss_term_is_positive_when_something_is_outside():
    scores = torch.tensor([0.1, 2.0])
    tau = torch.tensor(1.0)
    assert miss_term(scores, tau, eps=1e-8).item() > 0.0


def test_coverage_term_is_zero_at_the_target_rate():
    # scores exactly at tau give smooth_coverage = sigmoid(0) = 0.5
    scores = torch.zeros(10)
    tau = torch.tensor(0.0)
    loss = coverage_term(scores, tau, alpha=0.5, kappa=10.0)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_prediction_set_loss_is_finite_and_differentiable():
    torch.manual_seed(0)
    model = ICNN(input_dim=3, residual_dim=1)
    x = torch.randn(8, 3)
    r = torch.randn(8, 1)
    cfg = LossConfig(alpha=0.1, tau_mode="fixed", tau_fixed=1.0, lambda_miss=1.0, lambda_cov=1.0)
    loss = prediction_set_loss(model, x, r, cfg)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())
