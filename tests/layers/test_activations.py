from __future__ import annotations

import numpy as np
import pytest
import torch

from matgl.layers._activations import SoftExponential, SoftPlus2, softplus_inverse


@pytest.fixture
def x():
    return torch.tensor([1.0, 2.0])


def test_softplus2(x):
    out = SoftPlus2()(x)
    np.testing.assert_allclose(out.numpy(), np.array([0.62011445, 1.4337809]))


def test_soft_exponential(x):
    # alpha == 0 is the identity, but the output now participates in autograd
    # (alpha is a learnable parameter), so detach before going to numpy.
    out = SoftExponential()(x)
    np.testing.assert_allclose(out.detach().numpy(), np.array([1.0, 2.0]))
    out = SoftExponential(1.0)(x)
    np.testing.assert_allclose(out.detach().numpy(), np.array([2.7182817, 7.389056]))

    out = SoftExponential(-1.0)(x)
    np.testing.assert_allclose(out.detach().numpy(), np.array([0.0, 0.693147]), atol=1e-5)


def test_soft_exponential_negative_x_is_finite():
    """alpha < 0 must not produce NaN/Inf for large-negative inputs.

    The alpha < 0 branch evaluates ``-log(1 - alpha*(x + alpha)) / alpha``. For
    sufficiently negative ``x`` the log argument goes <= 0, which yields NaN/Inf.
    Such inputs are easy to hit early in training when features are large.
    """
    act = SoftExponential(-1.0)
    x = torch.tensor([-10.0, -5.0, -1.0, 0.0, 1.0])
    out = act(x)
    assert torch.isfinite(out).all(), f"non-finite output: {out}"


def test_soft_exponential_alpha_grad_is_finite():
    """dL/d(alpha) must stay finite even when inputs hit the undefined region.

    A NaN in the forward pass propagates into ``alpha.grad`` and poisons the
    optimizer, silently killing training.
    """
    act = SoftExponential(-1.0)
    x = torch.tensor([-10.0, -5.0, 1.0])
    act(x).sum().backward()
    assert act.alpha.grad is not None
    assert torch.isfinite(act.alpha.grad).all(), f"non-finite alpha.grad: {act.alpha.grad}"


def test_soft_exponential_alpha_is_learnable_both_signs():
    """alpha must receive a finite, non-zero gradient in both sign regions so it can train."""
    for alpha_init in (-0.7, 0.7):
        act = SoftExponential(alpha_init)
        x = torch.tensor([0.5, 1.5, 2.0])
        act(x).sum().backward()
        assert act.alpha.grad is not None
        assert torch.isfinite(act.alpha.grad).all()
        assert act.alpha.grad.abs() > 0


def test_softplus_inverse(x):
    assert torch.allclose(softplus_inverse(torch.nn.functional.softplus(x)), x, atol=1e-5)
