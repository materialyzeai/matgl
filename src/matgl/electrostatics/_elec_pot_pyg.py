from __future__ import annotations

from typing import Any
import numpy as np
import torch
import torch.nn as nn

import matgl
from matgl.config import COULOMB_CONSTANT
from matgl.utils.cutoff import polynomial_cutoff
from matgl.utils.maths import scatter_add
from matgl.graph._compute_pyg import compute_pair_vector_and_distance


class ElectrostaticPotential(nn.Module):
    r"""
    Compute electrostatic potentials for atoms within a molecular graph.

    This module calculates the electrostatic potential at each atomic site
    based on pairwise Coulomb interactions between charged atoms.
    It accounts for the finite extent of atomic charge distributions using
    Gaussian smearing (parameterized by atomic widths `sigma`).

    The electrostatic potential between atoms *i* and *j* is given by:

    $$
    V_{ij} =
    \frac{q_j}{r_{ij}} \\, \\mathrm{erf}\\!\\left(
        \frac{r_{ij}}{\\sqrt{2} \\, \\gamma_{ij}}
    \right)
    f_\text{cut}(r_{ij})
    $$

    where:
    - \\( q_j \\) is the charge on atom *j*,
    - \\( r_{ij} \\) is the interatomic distance,
    - \\( \\gamma_{ij} = \\sqrt{\\sigma_i^2 + \\sigma_j^2} \\) represents the combined width of charge distributions,
    - \\( f_\text{cut}(r_{ij}) \\) is a smooth polynomial cutoff function ensuring interactions
      vanish at the cutoff radius.

    The potential is scaled by the physical Coulomb constant.

    Parameters
    ----------
    element_types : tuple of str
        Tuple specifying the chemical element types in the system. Used
        for consistency or potential element-specific parameterization.

    cutoff : float
        Cutoff radius (in Å) beyond which the electrostatic interactions
        are smoothly reduced to zero by the cutoff function.

    Notes:
    -----
    - Requires node data fields: ``charge`` and ``sigma``.
    - Requires edge data field: ``bond_dist`` (pairwise distances).
    """

    def __init__(self, element_types: tuple[str, ...], cutoff: float):
        super().__init__()
        self.register_buffer("pi", torch.tensor(np.pi, dtype=matgl.float_th))
        self.register_buffer("sqrt2", torch.tensor(np.sqrt(2), dtype=matgl.float_th))
        self.element_types = element_types
        self.cutoff = cutoff

    def forward(self, 
                g: Any,
                charge: torch.Tensor,
                sigma: torch.Tensor
                ):
        """
        Aggregate electrostatic potential contributions for all atoms in the graph.

        Parameters
        ----------
        g : A PYG molecular graph:

        Returns:
        -------
        An updated PYG molecular graph
            The same input graph with an additional feature:
              - ``elec_pot`` (torch.Tensor): The total electrostatic potential at each atom.
        """
        if not hasattr(g, "edge_index"):
            raise AttributeError("ElectrostaticPotential expects a PyG Data-like object with `edge_index`.")
        else:
            edge_index = g.edge_index

        if not hasattr(g, "pos"):
            raise AttributeError("ElectrostaticPotential expects node positions in `pos`.")
        else:
            pos = g.pos

        # if not hasattr(g, "charge"):
        #     raise AttributeError("ElectrostaticPotential expects node charges in `charge`.")
        # else:
        #     charge = g.charge.reshape(-1)

        # if not hasattr(g, "sigma"):
        #     raise AttributeError("ElectrostaticPotential expects node widths in `sigma`.")
        # else:
        #     sigma = g.sigma
        
        _, bond_dist = compute_pair_vector_and_distance(pos, edge_index, getattr(g, "pbc_offshift", None))
        src, dst = edge_index[0], edge_index[1]
        gamma_ij = torch.sqrt(sigma[src]**2 + sigma[dst]**2)
        elec_pot_edge = (
            charge[dst]
            * torch.erf(bond_dist / self.sqrt2 / gamma_ij)
            * polynomial_cutoff(bond_dist, self.cutoff)
            / bond_dist
        )

        elec_pot = scatter_add(elec_pot_edge * COULOMB_CONSTANT, dst, dim_size=charge.shape[0])

        return elec_pot
