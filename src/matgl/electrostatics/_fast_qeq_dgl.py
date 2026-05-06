from __future__ import annotations

import dgl
import torch
from torch import nn


class LinearQeq(nn.Module):
    """Charge equilibrium within batches of structures. adapted from https://github.com/choderalab/espaloma-charge/blob/main/espaloma_charge/models.py."""

    def __init__(self):
        super().__init__()

    def get_charges(self, node: dgl.udf.NodeBatch):
        r"""Compute atomic charges in a structure using the charge equilibration (QEq) model.

        This function analytically solves for the atomic charges `q_i` given
        the electronegativity (`chi`), hardness (`hardness`), and total molecular charge (`sum_q`).
        The solution is derived using the method of Lagrange multipliers to enforce charge conservation.

        The potential energy function is defined as:
        $$
        U({\bf q}) =
        \\sum_{i=1}^N \\left[ \\chi_i q_i + \frac{1}{2} \\, \text{hardness}_i \\, q_i^2 \right]
        - \\lambda \\left( \\sum_{j=1}^N q_j - Q \right)
        $$

        Solving for equilibrium gives:
        $$
        q_i^* =
        - \\chi_i \\, \text{hardness}_i^{-1}
        + \text{hardness}_i^{-1} \\,
        \frac{Q + \\sum_{i=1}^N \\chi_i \\, \text{hardness}_i^{-1}}
             {\\sum_{j=1}^N \text{hardness}_j^{-1}}
        $$

        Parameters
        ----------
        node : dgl.udf.NodeBatch
            Node batch containing:
              - ``chi`` (torch.Tensor): Electronegativity of each atom.
              - ``hardness`` (torch.Tensor): Hardness of each atom.
              - ``sum_chi_hardness_inv`` (torch.Tensor): Sum of `chi * hardness^{-1}` for the molecule.
              - ``sum_hardness_inv`` (torch.Tensor): Sum of `hardness^{-1}` for the molecule.
              - ``sum_q`` (torch.Tensor): Total molecular charge, broadcast to all atoms.

        Returns:
        -------
        dict
            A dictionary with key:
              - ``charge`` (torch.Tensor): Computed atomic charge for each node.
        """
        chi = node.data["chi"]
        hardness = node.data["hardness"]
        sum_chi_hardness_inv = node.data["sum_chi_hardness_inv"]
        sum_hardness_inv = node.data["sum_hardness_inv"]
        sum_q = node.data["sum_q"]
        return {
            "charge": -chi * hardness**-1 + (hardness**-1) * torch.div(sum_q + sum_chi_hardness_inv, sum_hardness_inv)
        }

    def forward(self, g: dgl.DGLGraph, total_charge: torch.Tensor):
        r"""Compute atomic charges in a molecule using the charge equilibration (QEq) model.

        This function analytically solves for the atomic charges `q_i` given
        the electronegativity (`chi`), hardness (`hardness`), and total molecular charge (`sum_q`).
        The solution is derived using the method of Lagrange multipliers to enforce charge conservation.

        The potential energy function is defined as:
        $$
        U({\bf q}) =
        \\sum_{i=1}^N \\left[ \\chi_i q_i + \frac{1}{2} \\, \text{hardness}_i \\, q_i^2 \right]
        - \\lambda \\left( \\sum_{j=1}^N q_j - Q \right)
        $$

        Solving for equilibrium gives:
        $$
        q_i^* =
        - \\chi_i \\, \text{hardness}_i^{-1}
        + \text{hardness}_i^{-1} \\,
        \frac{Q + \\sum_{i=1}^N \\chi_i \\, \text{hardness}_i^{-1}}
             {\\sum_{j=1}^N \text{hardness}_j^{-1}}
        $$

        Parameters
        ----------
        node : dgl.udf.NodeBatch
            Node batch containing:
              - ``chi`` (torch.Tensor): Electronegativity of each atom.
              - ``hardness`` (torch.Tensor): Hardness of each atom.
              - ``sum_chi_hardness_inv`` (torch.Tensor): Sum of `chi * hardness^{-1}` for the molecule.
              - ``sum_hardness_inv`` (torch.Tensor): Sum of `hardness^{-1}` for the molecule.
              - ``sum_q`` (torch.Tensor): Total molecular charge, broadcast to all atoms.

        Returns:
        -------
        dict
            A dictionary with key:
              - ``charge`` (torch.Tensor): Computed atomic charge for each node.
        """
        g.apply_nodes(
            lambda node: {"hardness_inv": node.data["hardness"] ** -1},
        )
        g.apply_nodes(
            lambda node: {"chi_hardness_inv": node.data["chi"] * node.data["hardness"] ** -1},
        )

        if "q_ref" in g.ndata:
            total_charge = dgl.sum_nodes(g, "q_ref")
        else:
            total_charge = torch.ones(g.batch_size, 1, device=g.device) * total_charge

        g.ndata["sum_q"] = torch.squeeze(dgl.broadcast_nodes(g, total_charge))

        sum_hardness_inv = dgl.sum_nodes(g, "hardness_inv")
        sum_chi_hardness_inv = dgl.sum_nodes(g, "chi_hardness_inv")
        g.ndata["sum_hardness_inv"] = dgl.broadcast_nodes(g, sum_hardness_inv)
        g.ndata["sum_chi_hardness_inv"] = dgl.broadcast_nodes(g, sum_chi_hardness_inv)

        g.apply_nodes(self.get_charges)

        return g
