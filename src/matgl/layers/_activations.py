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

        Args:
            x (torch.tensor): Input tensor

        Returns:
            out (torch.tensor): Output tensor
        """
        if self.alpha == 0.0:
            return x
        if self.alpha < 0.0:
            return -torch.log(1.0 - self.alpha * (x + self.alpha)) / self.alpha
        return (torch.exp(self.alpha * x) - 1.0) / self.alpha + self.alpha


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
