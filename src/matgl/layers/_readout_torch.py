"""Backend-agnostic readout primitives.

Houses readout-style modules that operate on plain tensors and a
``batch`` index vector (PyG-style), making them usable from either
backend code path. Used internally by both
:mod:`matgl.layers._readout_pyg` and a handful of DGL readouts that
expose tensor-level helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from matgl.utils.maths import scatter_add

from ._core import MLP, GatedMLP

if TYPE_CHECKING:
    from collections.abc import Sequence


class WeightedAtomReadOut(nn.Module):
    """Weighted atom readout for graph properties using pure PyTorch tensors.

    This follows the TensorFlow WeightedReadout implementation:

        updated_field = mlp(field)
        weights = weight_mlp(field)
        factor = weights / sum(weights)
        readout = sum(factor * updated_field)

    where the normalization is performed independently for each graph.
    """

    def __init__(self, in_feats: int, dims: Sequence[int], activation: nn.Module):
        """Initialize the readout module.

        Args:
            in_feats: Input node feature dimension.
            dims: NN architecture for the MLP. The final entry is the output dimension.
            activation: Activation function for multi-layer perceptrons.
        """
        super().__init__()

        self.dims = [in_feats, *dims]
        self.activation = activation

        self.mlp = MLP(
            dims=self.dims,
            activation=self.activation,
            activate_last=True,
        )

        self.weight = MLP(
            dims=[*self.dims[:-1], 1],
            activation=self.activation,
            activate_last=False,
        )
        self.weight_activation = nn.Sigmoid()

    def forward(
        self,
        node_feat: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate weighted node features into graph-level representations.

        Args:
            node_feat: Node features with shape ``[num_nodes, in_feats]``.
            batch: Optional graph index tensor with shape ``[num_nodes]``.
                If ``None``, all nodes are treated as belonging to one graph.

        Returns:
            Graph-level tensor with shape ``[num_graphs, output_dim]``.
        """
        if batch is None:
            batch = torch.zeros(
                node_feat.size(0),
                dtype=torch.long,
                device=node_feat.device,
            )
        else:
            batch = batch.to(device=node_feat.device, dtype=torch.long)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        # updated_field = self.mlp(field)
        updated_field = self.mlp(node_feat)  # [num_nodes, output_dim]

        # weights = self.weight(field)
        weights = self.weight_activation(self.weight(node_feat))  # [num_nodes, 1]

        # sum_j w_j for each graph
        weight_sum = scatter_add(
            weights,
            batch,
            dim_size=num_graphs,
            dim=0,
        )  # [num_graphs, 1]

        # factor_i = w_i / sum_j w_j
        factor = weights / weight_sum[batch].clamp(min=1e-8)  # [num_nodes, 1]

        # sum_i factor_i * updated_field_i
        return scatter_add(
            factor * updated_field,
            batch,
            dim_size=num_graphs,
            dim=0,
        )  # [num_graphs, output_dim]


class ReduceReadOut(nn.Module):
    """Reduce node features into graph-level representations."""

    def __init__(self, op: str = "mean", field: str = "node_feat"):
        super().__init__()
        self.op = op
        self.field = field
        if op not in ["mean", "sum", "max"]:
            raise ValueError("op must be 'mean', 'sum', or 'max'")

    def forward(self, node_feat: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        if batch is not None:
            num_graphs = int(batch.max().item()) + 1
            if self.op == "sum":
                out = torch.zeros(num_graphs, node_feat.size(1), device=node_feat.device, dtype=node_feat.dtype)
                out.index_add_(0, batch.to(torch.long), node_feat)
            elif self.op == "mean":
                out = torch.zeros(num_graphs, node_feat.size(1), device=node_feat.device, dtype=node_feat.dtype)
                out.index_add_(0, batch.to(torch.long), node_feat)
                counts = torch.zeros(num_graphs, device=node_feat.device, dtype=torch.long)
                counts.index_add_(0, batch.to(torch.long), torch.ones_like(batch, dtype=torch.long))
                out = out / counts.unsqueeze(1).clamp(min=1)
            else:  # max
                out = torch.full(
                    (num_graphs, node_feat.size(1)),
                    float("-inf"),
                    device=node_feat.device,
                    dtype=node_feat.dtype,
                )
                out.index_reduce_(0, batch.to(torch.long), node_feat, "amax", include_self=False)
        elif self.op == "sum":
            out = node_feat.sum(dim=0, keepdim=True)
        elif self.op == "mean":
            out = node_feat.mean(dim=0, keepdim=True)
        else:  # max
            out = node_feat.max(dim=0, keepdim=True)[0]

        return out


class WeightedReadOut(nn.Module):
    """Per-node gated readout for atomic properties."""

    def __init__(self, in_feats: int, dims: Sequence[int], num_targets: int):
        super().__init__()
        self.in_feats = in_feats
        self.dims = [in_feats, *dims, num_targets]
        self.gated = GatedMLP(in_feats=in_feats, dims=self.dims, activate_last=False)

    def forward(self, node_feat: torch.Tensor) -> torch.Tensor:
        return self.gated(node_feat)
