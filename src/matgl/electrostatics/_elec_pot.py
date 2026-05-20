"""PyG implementation of the Gaussian-smeared Coulomb electrostatic potential."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn

import matgl
from matgl.config import COULOMB_CONSTANT
from matgl.graph._compute import compute_pair_vector_and_distance
from matgl.utils.cutoff import polynomial_cutoff
from matgl.utils.maths import scatter_add


class ElectrostaticPotential(nn.Module):
    r"""Aggregate per-atom electrostatic potentials over a PyG graph.

    For an edge :math:`i \to j` the contribution to the potential at site
    :math:`i` is

    .. math::

        V_{ij} = \frac{q_j}{r_{ij}} \,
                 \mathrm{erf}\!\left(\frac{r_{ij}}{\sqrt{2}\,\gamma_{ij}}\right)
                 f_\text{cut}(r_{ij}),

    with :math:`\gamma_{ij} = \sqrt{\sigma_i^2 + \sigma_j^2}` the combined
    Gaussian width, and :math:`f_\text{cut}` a smooth polynomial cutoff. The
    edge messages are aggregated at the **source** node, matching the DGL
    implementation in legacy DGL ``ElectrostaticPotential``.
    """

    def __init__(self, element_types: tuple[str, ...], cutoff: float):
        """Initialise the electrostatic-potential module.

        Args:
            element_types: Chemical element symbols in the system. Stored for
                downstream element-specific extensions; unused in the
                computation.
            cutoff: Cutoff radius (Å) beyond which interactions are smoothly
                damped to zero by ``polynomial_cutoff``.
        """
        super().__init__()
        # Buffers are registered for parity with the DGL module; the actual
        # computation uses Python-float constants to keep mypy quiet.
        self.register_buffer("pi", torch.tensor(np.pi, dtype=matgl.float_th))
        self.register_buffer("sqrt2", torch.tensor(np.sqrt(2), dtype=matgl.float_th))
        self.element_types = element_types
        self.cutoff = cutoff
        self._inv_sqrt2 = 1.0 / math.sqrt(2.0)

    def forward(self, g: Any, charge: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Compute the per-atom electrostatic potential.

        Args:
            g: PyG ``Data`` / ``Batch``-like object. Must expose ``edge_index``
                and ``pos``; ``pbc_offshift`` is used if present.
            charge: Per-atom charges, shape ``(num_nodes,)``.
            sigma: Per-atom Gaussian widths, shape ``(num_nodes,)``.

        Returns:
            Per-atom electrostatic potential, shape ``(num_nodes,)``.
        """
        if not hasattr(g, "edge_index"):
            raise AttributeError("ElectrostaticPotential expects a PyG Data-like object with `edge_index`.")
        if not hasattr(g, "pos"):
            raise AttributeError("ElectrostaticPotential expects node positions in `pos`.")

        edge_index = g.edge_index
        src, dst = edge_index[0], edge_index[1]

        _, bond_dist = compute_pair_vector_and_distance(g.pos, edge_index, getattr(g, "pbc_offshift", None))
        gamma_ij = torch.sqrt(sigma[src] ** 2 + sigma[dst] ** 2)
        edge_msg = (
            charge[dst]
            * torch.erf(bond_dist * self._inv_sqrt2 / gamma_ij)
            * polynomial_cutoff(bond_dist, self.cutoff)
            / bond_dist
        )

        return scatter_add(edge_msg * COULOMB_CONSTANT, src, dim_size=charge.shape[0])
