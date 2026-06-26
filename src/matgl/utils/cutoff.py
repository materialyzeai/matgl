"""Cutoff functions for constructing M3GNet potentials."""

from __future__ import annotations

import torch


def polynomial_cutoff(r: torch.Tensor, cutoff: float, exponent: int = 3) -> torch.Tensor:
    """Envelope polynomial function that ensures a smooth cutoff.

    Ensures first and second derivative vanish at cuttoff. As described in:
        https://arxiv.org/abs/2003.03123

    Args:
        r (torch.Tensor): radius distance tensor
        cutoff (float): cutoff distance.
        exponent (int): minimum exponent of the polynomial. Default is 3.
            The polynomial includes terms of order exponent, exponent + 1, exponent + 2.

    Returns: polynomial cutoff function
    """
    coef1 = -(exponent + 1) * (exponent + 2) / 2
    coef2 = exponent * (exponent + 2)
    coef3 = -exponent * (exponent + 1) / 2
    ratio = r / cutoff
    poly_envelope = 1 + coef1 * ratio**exponent + coef2 * ratio ** (exponent + 1) + coef3 * ratio ** (exponent + 2)

    return torch.where(r <= cutoff, poly_envelope, 0.0)


def cosine_cutoff(r: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Cosine cutoff function.

    Args:
        r (torch.Tensor): radius distance tensor
        cutoff (float): cutoff distance.

    Returns:
        cosine cutoff function
    """
    return torch.where(r <= cutoff, 0.5 * (torch.cos(torch.pi * r / cutoff) + 1), 0.0)
