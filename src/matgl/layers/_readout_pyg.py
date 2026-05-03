"""Readout layer in PYG."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from matgl.layers import MLP, GatedMLP

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch_geometric.data import Data


class ReduceReadOut(nn.Module):
    """Reduce node or edge attributes into lower dimensional tensors as readout in PyTorch Geometric.

    This could be summing up the nodes or edges, or taking the mean, etc.
    """

    def __init__(self, op: str = "mean", field: str = "node_feat"):
        """Initialize the ReduceReadOut.

        Args:
            op (str): Operation for the reduction ('mean', 'sum', or 'max').
            field (str): Field to perform the reduction ('node_feat' or 'edge_feat').
        """
        super().__init__()
        self.op = op
        self.field = field
        if op not in ["mean", "sum", "max"]:
            raise ValueError("op must be 'mean', 'sum', or 'max'")
        if field not in ["node_feat"]:
            raise ValueError("field must be 'node_feat'")

        # Map operation to PyG pooling function
        self.pool_fn = {"mean": global_mean_pool, "sum": global_add_pool, "max": global_max_pool}[op]

    def forward(self, graph: Data) -> torch.Tensor:
        """Forward pass.

        Args:
            graph (Data): PyG Data object containing x, edge_attr, edge_index, and batch.

        Returns:
            torch.Tensor: Pooled features, shape (num_graphs, feature_dim).
        """
        if not hasattr(graph, "node_feat") or graph.node_feat is None:
            raise ValueError("Data object must contain node features (graph.node_feat)")
        return self.pool_fn(graph.node_feat, graph.batch)


class WeightedReadOut(nn.Module):
    """Feed node features into Gated MLP as readout for atomic properties."""

    def __init__(self, in_feats: int, dims: Sequence[int], num_targets: int):
        """Initialize the WeightedReadOut.

        Args:
            in_feats: input features (nodes).
            dims: NN architecture for Gated MLP.
            num_targets: number of target properties.
        """
        super().__init__()
        self.in_feats = in_feats
        self.dims = [in_feats, *dims, num_targets]
        self.gated = GatedMLP(in_feats=in_feats, dims=self.dims, activate_last=False)

    def forward(self, node_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            node_feat (torch.Tensor): Per-node input features.
                Shape: (num_nodes, num_node_features).

        Returns:
            atomic_properties (torch.Tensor): Per-node atomic properties.
                Shape: (num_nodes, num_targets).
        """
        atomic_properties = self.gated(node_feat)
        return atomic_properties


class WeightedAtomReadOut(nn.Module):
    """Weighted atom readout for graph properties in PyTorch Geometric."""

    def __init__(self, in_feats: int, dims: Sequence[int], activation: nn.Module):
        """Initialize the WeightedAtomReadOut.

        Args:
            in_feats: Input features (nodes).
            dims: NN architecture for Gated MLP.
            activation: Activation function for multi-layer perceptrons.
        """
        super().__init__()
        self.dims = [in_feats, *dims]
        self.activation = activation
        self.mlp = MLP(dims=self.dims, activation=self.activation, activate_last=True)
        self.weight = nn.Sequential(nn.Linear(in_feats, 1), nn.Sigmoid())

    def forward(self, graph: Data) -> torch.Tensor:
        """Run the weighted atom readout.

        Args:
            graph: PyG graph Data object.

        Returns:
            atomic_properties: Tensor of shape (num_graphs, output_dim).
        """
        # Apply MLP to node features
        h = self.mlp(graph.node_feat)  # Shape: (num_nodes, output_dim)

        # Compute weights for each node
        w = self.weight(graph.node_feat)  # Shape: (num_nodes, 1)

        # Weighted node features
        weighted_h = h * w  # Element-wise multiplication, shape: (num_nodes, output_dim)

        # Aggregate weighted node features per graph using global_add_pool
        h_g_sum = global_add_pool(weighted_h, graph.batch)  # Shape: (num_graphs, output_dim)

        return h_g_sum
