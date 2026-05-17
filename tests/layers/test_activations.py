from __future__ import annotations

import numpy as np
import pytest
import torch
from matgl.layers._activations import SoftExponential, SoftPlus2, softplus_inverse


@pytest.fixture()
def x():
    return torch.tensor([1.0, 2.0])


def test_softplus2(x):
    out = SoftPlus2()(x)
    np.testing.assert_allclose(out.numpy(), np.array([0.62011445, 1.4337809]))


def test_soft_exponential(x):
    out = SoftExponential()(x)
    np.testing.assert_allclose(out.numpy(), np.array([1.0, 2.0]))
    out = SoftExponential(1.0)(x)
    np.testing.assert_allclose(out.detach().numpy(), np.array([2.7182817, 7.389056]))

    out = SoftExponential(-1.0)(x)
    np.testing.assert_allclose(out.detach().numpy(), np.array([0.0, 0.693147]), atol=1e-5)


def test_softplus_inverse(x):
    assert torch.allclose(softplus_inverse(torch.nn.functional.softplus(x)), x, atol=1e-5)
