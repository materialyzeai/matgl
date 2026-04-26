from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

import matgl
from matgl.config import COULOMB_CONSTANT
from matgl.utils.cutoff import polynomial_cutoff
from matgl.utils.maths import scatter_add


class ElectrostaticPotential(nn.Module):
    r"""
    Compute electrostatic potentials for atoms (backend-agnostic, operates on plain tensors).

    For each edge (src → dst), computes the pairwise Coulomb contribution:

    $$
    V_{ij} =
    \frac{q_j}{r_{ij}} \, \mathrm{erf}\!\left(
        \frac{r_{ij}}{\sqrt{2} \, \gamma_{ij}}
    \right)
    f_\text{cut}(r_{ij})
    $$

    where:
    - :math:`q_j` is the charge on the destination atom *j*,
    - :math:`r_{ij}` is the interatomic distance,
    - :math:`\gamma_{ij} = \sqrt{\sigma_i^2 + \sigma_j^2}` is the combined Gaussian width,
    - :math:`f_\text{cut}` is a smooth polynomial cutoff.

    The result is scaled by the Coulomb constant and aggregated at each destination atom.

    Parameters
    ----------
    element_types : tuple of str
        Chemical element types in the system.
    cutoff : float
        Cutoff radius (in Å).
    """

    def __init__(self, element_types: tuple[str, ...], cutoff: float):
        super().__init__()
        self.register_buffer("sqrt2", torch.tensor(np.sqrt(2), dtype=matgl.float_th))
        self.element_types = element_types
        self.cutoff = cutoff

    def forward(
        self,
        charge: torch.Tensor,
        sigma: torch.Tensor,
        bond_dist: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Aggregate edge-wise electrostatic potential contributions to destination atoms.

        Args:
            charge: Atomic charges, shape (num_nodes,).
            sigma: Gaussian widths of atomic charge distributions, shape (num_nodes,).
            bond_dist: Pairwise interatomic distances, shape (num_edges,).
            edge_index: Edge connectivity in COO format, shape (2, num_edges).
            num_nodes: Total number of atoms.

        Returns:
            elec_pot: Total electrostatic potential per atom, shape (num_nodes,).
        """
        src_idx, dst_idx = edge_index[0], edge_index[1]

        charge_dst = charge[dst_idx]
        sigma_src = sigma[src_idx]
        sigma_dst = sigma[dst_idx]

        gamma_ij = torch.sqrt(sigma_src**2 + sigma_dst**2)

        elec_pot_edge = (
            charge_dst
            * torch.erf(bond_dist / self.sqrt2 / gamma_ij)
            * polynomial_cutoff(bond_dist, self.cutoff)
            / bond_dist
        ) * COULOMB_CONSTANT

        # Aggregate messages at destination nodes
        elec_pot = scatter_add(elec_pot_edge, dst_idx, dim_size=num_nodes)

        return elec_pot
