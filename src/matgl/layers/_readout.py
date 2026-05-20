"""Readout layers for graph-level predictions.

Public exports re-exported from :mod:`matgl.layers`:

* :class:`Set2SetReadOut` and :class:`EdgeSet2Set` (node / edge variants
  of Set2Set);
* :class:`ReduceReadOut` -- ``sum``/``mean``/``max`` pooling;
* :class:`WeightedReadOut` and :class:`WeightedAtomReadOut` -- learned
  per-atom weighting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.nn import LSTM
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.nn.aggr import Set2Set as PyGSet2Set
from torch_geometric.utils import softmax as pyg_softmax

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
        return self.gated(node_feat)


class WeightedAtomReadOut(nn.Module):
    """Weighted atom readout for graph properties in PyTorch Geometric.

    This follows the TensorFlow WeightedReadout implementation:

        updated_field = mlp(field)
        weights = weight_mlp(field)
        factor = weights / sum(weights)
        readout = sum(factor * updated_field)

    where the normalization is performed independently for each graph.
    """

    def __init__(self, in_feats: int, dims: Sequence[int], activation: nn.Module):
        """Initialize the WeightedAtomReadOut.

        Args:
            in_feats: Input features (nodes).
            dims: NN architecture for the MLP. The final entry is the output dimension.
            activation: Activation function for multi-layer perceptrons.
        """
        super().__init__()

        self.dims = [in_feats, *dims]
        self.activation = activation

        # Equivalent to the TensorFlow readout MLP:
        # self.mlp = MLP(neurons=neurons, activations=[activation] * n_layer)
        self.mlp = MLP(
            dims=self.dims,
            activation=self.activation,
            activate_last=True,
        )

        # Equivalent to:
        # weight_neurons = neurons[:-1] + [1]
        # weight_activations = [activation] * (n_layer - 1) + ["sigmoid"]
        self.weight = MLP(
            dims=[*self.dims[:-1], 1],
            activation=self.activation,
            activate_last=False,
        )
        self.weight_activation = nn.Sigmoid()

    def forward(self, graph: Data) -> torch.Tensor:
        """Run the weighted atom readout.

        Args:
            graph: PyG graph Data object with ``graph.node_feat`` and ``graph.batch``.

        Returns:
            atomic_properties: Tensor of shape ``[num_graphs, output_dim]``.
        """
        node_feat = graph.node_feat

        if hasattr(graph, "batch") and graph.batch is not None:
            batch = graph.batch
        else:
            batch = torch.zeros(
                node_feat.shape[0],
                dtype=torch.long,
                device=node_feat.device,
            )

        # updated_field = self.mlp(field)
        updated_field = self.mlp(node_feat)  # [num_nodes, output_dim]

        # weights = self.weight(field)
        weights = self.weight_activation(self.weight(node_feat))  # [num_nodes, 1]

        # sum_j w_j for each graph
        weight_sum = global_add_pool(weights, batch)  # [num_graphs, 1]

        # factor_i = w_i / sum_j w_j
        factor = weights / weight_sum[batch].clamp(min=1e-8)  # [num_nodes, 1]

        # sum_i factor_i * updated_field_i
        return global_add_pool(factor * updated_field, batch)


class Set2SetReadOut(PyGSet2Set):
    """Set2Set readout for nodes (PyG).

    Subclasses :class:`torch_geometric.nn.aggr.Set2Set` so the underlying
    ``lstm`` parameter path matches the DGL implementation
    (``node_s2s.lstm.weight_ih_l0`` etc.), enabling cross-backend ``state_dict``
    transfer for parity testing. Output dimension is ``2 * input_dim``.
    """

    def __init__(self, in_feats: int, n_iters: int, n_layers: int, field: str = "node_feat") -> None:
        """Initialize the Set2Set readout.

        Args:
            in_feats: Per-node feature dimension.
            n_iters: Number of iterative refinement steps in Set2Set.
            n_layers: Number of LSTM layers inside Set2Set.
            field: Kept for signature parity with the DGL implementation; only
                ``"node_feat"`` is supported in the PyG variant. Use
                :class:`EdgeSet2Set` for edge-based readouts.
        """
        if field != "node_feat":
            raise NotImplementedError(
                "Set2SetReadOut (PyG) only supports field='node_feat'. Use EdgeSet2Set for edges."
            )
        super().__init__(in_channels=in_feats, processing_steps=n_iters, num_layers=n_layers)
        self.in_feats = in_feats
        self.out_feats = 2 * in_feats
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.field = field

    def forward(  # type: ignore[override]
        self,
        x: torch.Tensor,
        index: torch.Tensor | None = None,
        ptr: torch.Tensor | None = None,
        dim_size: int | None = None,
        dim: int = -2,
    ) -> torch.Tensor:
        """Run Set2Set on per-node features.

        Args:
            x: Per-node features, shape ``(num_nodes, in_feats)``.
            index: Per-node batch index. ``None`` is treated as a single graph.
            ptr: Optional CSR-style boundary pointer (see PyG).
            dim_size: Output dimension; computed automatically if ``None``.
            dim: Aggregation dimension (passed through to PyG).

        Returns:
            Per-graph readout vector, shape ``(num_graphs, 2 * in_feats)``.
        """
        if index is None and ptr is None:
            index = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if index is not None:
            index = index.to(torch.long)
        return super().forward(x, index=index, ptr=ptr, dim_size=dim_size, dim=dim)


class EdgeSet2Set(nn.Module):
    """Set2Set-style readout for edges.

    Uses ``torch_geometric.utils.softmax`` keyed on the per-edge batch index.
    Output dimension is ``2 * input_dim``.
    """

    def __init__(self, input_dim: int, n_iters: int, n_layers: int) -> None:
        """Initialize the EdgeSet2Set readout.

        Args:
            input_dim: Per-edge feature dimension.
            n_iters: Number of iterative refinement steps.
            n_layers: Number of LSTM layers inside Set2Set.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = LSTM(self.output_dim, self.input_dim, n_layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialize learnable parameters."""
        self.lstm.reset_parameters()

    def forward(
        self,
        edge_feat: torch.Tensor,
        edge_batch: torch.Tensor | None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Run Set2Set on per-edge features.

        Args:
            edge_feat: Per-edge features, shape ``(num_edges, input_dim)``.
            edge_batch: Per-edge batch index. ``None`` is treated as a single graph.
            num_graphs: Number of graphs in the batch. Inferred from ``edge_batch``
                if ``None``.

        Returns:
            Per-graph readout vector, shape ``(num_graphs, 2 * input_dim)``.
        """
        if edge_batch is None:
            edge_batch = torch.zeros(edge_feat.size(0), dtype=torch.long, device=edge_feat.device)
        edge_batch = edge_batch.to(torch.long)
        if num_graphs is None:
            num_graphs = int(edge_batch.max().item()) + 1 if edge_batch.numel() > 0 else 1

        # Edge-less graphs: Set2Set output is zeros (matches DGL ``sum_edges`` on
        # an empty edge set). Skip the LSTM iterations to avoid scatter on empty.
        if edge_feat.size(0) == 0:
            return edge_feat.new_zeros(num_graphs, self.output_dim)

        h = (
            edge_feat.new_zeros((self.n_layers, num_graphs, self.input_dim)),
            edge_feat.new_zeros((self.n_layers, num_graphs, self.input_dim)),
        )
        q_star = edge_feat.new_zeros(num_graphs, self.output_dim)

        for _ in range(self.n_iters):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(num_graphs, self.input_dim)
            # Per-edge attention logits: dot(edge_feat, q[edge_batch])
            e = (edge_feat * q[edge_batch]).sum(dim=-1, keepdim=True)
            alpha = pyg_softmax(e, edge_batch, num_nodes=num_graphs)
            weighted = edge_feat * alpha
            readout = torch.zeros(
                num_graphs, weighted.size(-1), dtype=weighted.dtype, device=weighted.device
            ).index_add(0, edge_batch, weighted)
            q_star = torch.cat([q, readout], dim=-1)

        return q_star
