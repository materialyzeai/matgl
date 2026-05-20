"""PyG implementation of the closed-form charge-equilibration solver."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from matgl.utils.maths import scatter_add

# Lower bound applied to ``sum_j eta_j^{-1}`` before dividing. Guards against
# inf/nan when an entire graph in the batch has near-zero inverse hardness
# (degenerate atom types or an unhealthy training step). Chosen well below any
# physical inverse-hardness scale but large enough to avoid float32 underflow.
_QEQ_DENOM_EPS = 1e-12


class LinearQeq(nn.Module):
    r"""Charge equilibrium within batches of structures (PyG backend).

    Adapted from
    https://github.com/choderalab/espaloma-charge/blob/main/espaloma_charge/models.py.

    The atomic charges :math:`q_i` are obtained by analytically minimising

    .. math::

        U(\mathbf{q}) = \sum_{i=1}^N \left[ \chi_i q_i + \tfrac12 \, \eta_i q_i^2 \right]
        - \lambda \left( \sum_{j=1}^N q_j - Q \right)

    using the method of Lagrange multipliers, which yields

    .. math::

        q_i^* = -\chi_i \eta_i^{-1}
        + \eta_i^{-1} \frac{Q + \sum_j \chi_j \eta_j^{-1}}{\sum_j \eta_j^{-1}}.
    """

    def forward(
        self,
        g: Any,
        total_charge: torch.Tensor | None,
        chi: torch.Tensor,
        hardness: torch.Tensor,
    ) -> torch.Tensor:
        """Solve QEq analytically for every graph in a batch.

        Args:
            g: PyG ``Data`` / ``Batch``-like object. Optional attributes consulted:
                ``batch`` (per-node graph index), ``num_graphs`` (batch size), and
                ``q_ref`` (per-node reference charges; if present the per-graph
                total charge is taken as their sum, overriding ``total_charge``).
            total_charge: Per-graph total charge. ``None`` is treated as zero;
                a scalar tensor is broadcast to every graph; otherwise must have
                one entry per graph in the batch.
            chi: Per-node electronegativity, shape ``(num_nodes,)``.
            hardness: Per-node chemical hardness, shape ``(num_nodes,)``.

        Returns:
            Per-node equilibrated charges, shape ``(num_nodes,)``.
        """
        chi = chi.reshape(-1)
        hardness = hardness.reshape(-1)

        batch = getattr(g, "batch", None)
        if batch is None:
            batch = torch.zeros(chi.shape[0], dtype=torch.long, device=chi.device)
            num_graphs = 1
        else:
            batch = batch.to(torch.long)
            num_graphs = int(getattr(g, "num_graphs", int(batch.max()) + 1))

        hardness_inv = hardness.reciprocal()
        chi_hardness_inv = chi * hardness_inv

        if hasattr(g, "q_ref"):
            q_ref = g.q_ref.reshape(-1)
            total_charge_graph = scatter_add(q_ref, batch, dim_size=num_graphs)
        elif total_charge is None:
            total_charge_graph = torch.zeros(num_graphs, device=chi.device, dtype=chi.dtype)
        else:
            total_charge_graph = total_charge.to(device=chi.device, dtype=chi.dtype).reshape(-1)
            if total_charge_graph.numel() == 1:
                total_charge_graph = total_charge_graph.expand(num_graphs)
            elif total_charge_graph.numel() != num_graphs:
                raise ValueError("total_charge must be a scalar or have one value per graph in the batch.")

        sum_hardness_inv = scatter_add(hardness_inv, batch, dim_size=num_graphs)[batch]
        sum_chi_hardness_inv = scatter_add(chi_hardness_inv, batch, dim_size=num_graphs)[batch]
        sum_q = total_charge_graph[batch]

        # Clamp the denominator to avoid silent inf/nan for degenerate batches.
        sum_hardness_inv = sum_hardness_inv.clamp_min(_QEQ_DENOM_EPS)
        return -chi * hardness_inv + hardness_inv * (sum_q + sum_chi_hardness_inv) / sum_hardness_inv
