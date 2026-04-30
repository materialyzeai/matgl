from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

import matgl
from matgl.config import COULOMB_CONSTANT
from matgl.utils.cutoff import polynomial_cutoff
from matgl.utils.maths import scatter_add

if TYPE_CHECKING:
    import dgl


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

    def message_func(self, edges: dgl.udf.EdgeBatch):
        """
        Compute the pairwise electrostatic potential contribution along each edge.

        Parameters
        ----------
        edges : dgl.udf.EdgeBatch
            Batch of edges containing:
              - ``edges.src["sigma"]`` (torch.Tensor): Width of source atom Gaussian charge distribution.
              - ``edges.dst["sigma"]`` (torch.Tensor): Width of destination atom Gaussian charge distribution.
              - ``edges.dst["charge"]`` (torch.Tensor): Atomic charge of the destination atom.
              - ``edges.data["bond_dist"]`` (torch.Tensor): Interatomic distance between source and destination atoms.

        Returns:
        -------
        dict
            Dictionary with:
              - ``"elec_pot"`` (torch.Tensor): Edge-wise electrostatic potential contribution
                (scaled by Coulomb constant).
        """
        charge = edges.dst["charge"]
        gamma_ij = torch.sqrt(edges.src["sigma"] ** 2 + edges.dst["sigma"] ** 2)
        bond_dist = edges.data["bond_dist"]
        elec_pot = (
            charge
            * torch.erf(bond_dist / self.sqrt2 / gamma_ij)
            * polynomial_cutoff(bond_dist, self.cutoff)
            / bond_dist
        )
        return {"m_elec_pot": elec_pot * COULOMB_CONSTANT}

    def forward(self, g: dgl.DGLGraph) -> dgl.DGLGraph:
        """
        Aggregate electrostatic potential contributions for all atoms in the graph.

        This function performs a message-passing operation over the molecular graph:
        each atom receives the summed electrostatic potential contributions
        from its neighboring atoms.

        Parameters
        ----------
        g : dgl.DGLGraph
            A molecular graph containing:
              - Node features:
                  - ``charge`` (torch.Tensor): Atomic charge.
                  - ``sigma`` (torch.Tensor): Gaussian width of charge distribution.
              - Edge features:
                  - ``bond_dist`` (torch.Tensor): Pairwise atomic distance.

        Returns:
        -------
        dgl.DGLGraph
            The same input graph with an additional node feature:
              - ``elec_pot`` (torch.Tensor): The total electrostatic potential at each atom.
        """
        g.apply_edges(self.message_func)
        src, _ = g.edges()
        g.ndata["elec_pot"] = scatter_add(
            g.edata["m_elec_pot"],
            src,
            dim=0,
            dim_size=g.num_nodes(),
        )
        return g
