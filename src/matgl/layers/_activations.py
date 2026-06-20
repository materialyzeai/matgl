"""Custom activation functions used across matgl architectures.

Implements two non-standard activations not shipped with stock PyTorch:

* :class:`SoftPlus2` -- ``softplus(x) - log(2)``; zero at the origin,
  smooth, non-saturating. Used by MEGNet and the original M3GNet readout.
* :class:`SoftExponential` -- a learnable activation with an adjustable
  ``alpha`` parameter (https://arxiv.org/abs/1602.01321).

Plus a small utility (:func:`softplus_inverse`) and an enum
(:class:`ActivationFunction`) that maps human-readable names to
constructors so config-driven model assembly can pick activations by
string.
"""

from __future__ import annotations

import math
from enum import Enum

import torch
from torch import nn


class SoftPlus2(nn.Module):
    """SoftPlus2 activation function.

    out = log(exp(x)+1) - log(2)
    softplus function that is 0 at x=0, the implementation aims at avoiding overflow.
    """

    def __init__(self) -> None:
        """Initializes the SoftPlus2 class."""
        super().__init__()
        self.ssp = nn.Softplus()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate activation function given the input tensor x.

        Args:
            x (torch.tensor): Input tensor

        Returns:
            out (torch.tensor): Output tensor
        """
        return self.ssp(x) - math.log(2.0)


class SoftExponential(nn.Module):
    """Soft exponential activation.

    When x < 0, SoftExponential(x,alpha) = -log(1-alpha(x+alpha))/alpha
    When x = 0, SoftExponential(x,alpha) = 0
    When x > 0, SoftExponential(x,alpha) = (exp(alpha*x)-1)/alpha + alpha.

    References: https://arxiv.org/pdf/1602.01321.pdf
    """

    # |alpha| below this is treated as the identity (alpha -> 0) region, and it
    # also floors the log argument / denominator to keep the activation and its
    # gradient finite.
    _eps = 1e-6

    def __init__(self, alpha: float | None = None):
        """Init SoftExponential with alpha value.

        Args:
            alpha (float): adjustable Torch parameter during the training.
        """
        super().__init__()

        # initialize alpha
        if alpha is None:
            self.alpha = nn.Parameter(torch.tensor(0.0))
        else:
            self.alpha = nn.Parameter(torch.tensor(alpha))

        self.alpha.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate activation function given the input tensor x.

        Branch selection uses ``torch.where`` rather than a Python ``if`` on
        ``self.alpha``: a host-side ``bool(self.alpha < 0)`` test forces a device
        sync and drops the branch from the autograd graph, so ``alpha`` could
        only ever learn within whichever sign region it was initialised in.
        ``torch.where`` keeps both formulas in the graph. Because ``where`` still
        evaluates both sides, the denominators and the ``log`` argument are
        guarded so the discarded branch cannot inject NaN/Inf into the gradient
        (the "double where" trick).

        Args:
            x (torch.tensor): Input tensor

        Returns:
            out (torch.tensor): Output tensor
        """
        alpha = self.alpha
        # Treat |alpha| < eps as the identity region (the alpha -> 0 limit)
        # instead of testing exact equality to 0.0, which a learned float never
        # hits after the first optimizer step.
        near_zero = alpha.abs() < self._eps
        # Never divide by a (near) zero alpha, even on the branch that gets
        # discarded, otherwise NaN/Inf would poison alpha.grad.
        safe_alpha = torch.where(near_zero, torch.ones_like(alpha), alpha)

        # alpha < 0 branch: -log(1 - alpha*(x + alpha)) / alpha. The log argument
        # can go <= 0 for sufficiently negative x; clamp it to a small positive
        # floor on the live branch so it never produces NaN/Inf.
        neg_log_arg = torch.where(alpha < 0.0, 1.0 - alpha * (x + alpha), torch.ones_like(x))
        neg = -torch.log(neg_log_arg.clamp_min(self._eps)) / safe_alpha

        # alpha > 0 branch: (exp(alpha*x) - 1)/alpha + alpha; expm1 is accurate near 0.
        pos = torch.expm1(safe_alpha * x) / safe_alpha + safe_alpha

        out = torch.where(alpha < 0.0, neg, pos)
        return torch.where(near_zero, x, out)


def softplus_inverse(x: torch.Tensor):
    """Inverse of the softplus function.

    Args:
        x (torch.Tensor): Input vector

    Returns:
        torch.Tensor: softplus inverse of input.
    """
    return x + (torch.log(-torch.expm1(-x)))


class ActivationFunction(Enum):
    """Enumeration of optional activation functions."""

    swish = nn.SiLU
    sigmoid = nn.Sigmoid
    tanh = nn.Tanh
    softplus = nn.Softplus
    softplus2 = SoftPlus2
    softexp = SoftExponential
