from __future__ import annotations

from typing import Any
import torch
from torch import nn
from matgl.utils.maths import scatter_add

class LinearQeq(nn.Module):
    """Charge equilibrium within batches of structures. adapted from https://github.com/choderalab/espaloma-charge/blob/main/espaloma_charge/models.py."""

    def __init__(self):
        super().__init__()

    def forward(self, 
                g: Any, 
                total_charge: torch.Tensor,
                chi: torch.Tensor,
                hardness: torch.Tensor):
        r"""
        Compute atomic charges in a molecule using the charge equilibration (QEq) model.

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
        """
        # if not hasattr(g, "chi") or not hasattr(g, "hardness"):
        #     raise AttributeError("LinearQeq expects node attributes `chi` and `hardness`.")
        
        chi = chi.reshape(-1)
        hardness = hardness.reshape(-1)

        batch = getattr(g, "batch", None)
        if batch is None:
            batch = torch.zeros(chi.shape[0], dtype=torch.long, device=chi.device)
            num_graphs = 1
        else:
            batch = batch.to(torch.long)
            num_graphs = int(getattr(g, "num_graphs", int(batch.max().item()) + 1))

        hardness_inv = hardness.reciprocal()
        chi_hardness_inv = chi * hardness_inv

        if hasattr(g, "q_ref"):
            q_ref = g.q_ref.reshape(-1)
            total_charge_graph = scatter_add(q_ref, batch, dim_size=num_graphs)
        else:
            if total_charge is None:
                total_charge_graph = torch.zeros(num_graphs, device=chi.device, dtype=chi.dtype)
            else:
                total_charge_graph = total_charge.to(device=chi.device, dtype=chi.dtype).reshape(-1)
                if total_charge_graph.numel() == 1:
                    total_charge_graph = total_charge_graph.expand(num_graphs)
                elif total_charge_graph.numel() != num_graphs:
                    raise ValueError("total_charge must be a scalar or have one value per graph in the batch.")

        sum_hardness_inv = scatter_add(hardness_inv, batch, dim_size=num_graphs)
        sum_chi_hardness_inv = scatter_add(chi_hardness_inv, batch, dim_size=num_graphs)

        sum_q = total_charge_graph[batch]
        sum_hardness_inv = sum_hardness_inv[batch]
        sum_chi_hardness_inv = sum_chi_hardness_inv[batch]
        
        charge = -chi * hardness_inv + hardness_inv * (sum_q + sum_chi_hardness_inv) / sum_hardness_inv
        return charge