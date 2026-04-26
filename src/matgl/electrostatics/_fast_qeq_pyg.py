from __future__ import annotations

import torch
from torch import nn

from matgl.utils.maths import scatter_add


class LinearQeq(nn.Module):
    """Charge equilibration (backend-agnostic, operates on plain tensors + batch indices).

    Adapted from https://github.com/choderalab/espaloma-charge/blob/main/espaloma_charge/models.py.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        chi: torch.Tensor,
        hardness: torch.Tensor,
        batch: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
        q_ref: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Compute atomic charges using the charge equilibration (QEq) model.

        Analytically solves for atomic charges given electronegativity (chi), hardness,
        and total molecular charge (sum_q) via Lagrange multipliers:

        $$
        q_i^* =
        - \chi_i \, \text{hardness}_i^{-1}
        + \text{hardness}_i^{-1} \,
        \frac{Q + \sum_{i=1}^N \chi_i \, \text{hardness}_i^{-1}}
             {\sum_{j=1}^N \text{hardness}_j^{-1}}
        $$

        Args:
            chi: Electronegativity per atom, shape (num_nodes,).
            hardness: Hardness per atom, shape (num_nodes,).
            batch: Graph batch indices, shape (num_nodes,). If None, treated as single graph.
            total_charge: Total charge per graph, shape (num_graphs,) or scalar. Ignored if q_ref is provided.
            q_ref: Reference per-atom charges. If provided, graph total charges are derived from this.

        Returns:
            charge: Computed atomic charges, shape (num_nodes,).
        """
        num_nodes = chi.shape[0]

        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=chi.device)

        num_graphs = int(batch.max().item()) + 1

        hardness_inv = hardness**-1
        chi_hardness_inv = chi * hardness_inv

        if q_ref is not None:
            sum_q = scatter_add(q_ref, batch, dim_size=num_graphs)
        elif total_charge is not None:
            if total_charge.dim() == 0 or (total_charge.dim() == 1 and total_charge.shape[0] == 1):
                sum_q = total_charge.expand(num_graphs)
            else:
                sum_q = total_charge.view(num_graphs)
        else:
            sum_q = torch.zeros(num_graphs, device=chi.device, dtype=chi.dtype)

        sum_hardness_inv = scatter_add(hardness_inv, batch, dim_size=num_graphs)
        sum_chi_hardness_inv = scatter_add(chi_hardness_inv, batch, dim_size=num_graphs)

        # Broadcast per-graph aggregates back to per-atom
        sum_q_per_atom = sum_q[batch]
        sum_hardness_inv_per_atom = sum_hardness_inv[batch]
        sum_chi_hardness_inv_per_atom = sum_chi_hardness_inv[batch]

        charge = (
            -chi * hardness_inv
            + hardness_inv * (sum_q_per_atom + sum_chi_hardness_inv_per_atom) / sum_hardness_inv_per_atom
        )

        return charge
