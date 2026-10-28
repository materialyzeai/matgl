"""Readout layers for graph-level predictions.

Public exports re-exported from :mod:`matgl.layers`:

* :class:`Set2SetReadOut` and :class:`EdgeSet2Set` (node / edge variants
  of Set2Set);
* :class:`ReduceReadOut` -- ``sum``/``mean``/``max`` pooling;
* :class:`WeightedReadOut` and :class:`WeightedAtomReadOut` -- learned
  per-atom weighting.

The graph-level readouts take per-node features plus a ``batch`` index and an
optional ``num_graphs``. Pass ``num_graphs`` whenever it is known: inferring it
from ``batch.max() + 1`` drops rows for trailing graphs that contribute no
nodes. For backwards compatibility, ``ReduceReadOut`` and
``WeightedAtomReadOut`` also accept a PyG ``Data``/``Batch`` carrying
``node_feat`` (and ``batch``) in place of the feature tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import LSTM
from torch_geometric.nn.aggr import Set2Set as PyGSet2Set
from torch_geometric.utils import softmax as pyg_softmax

from matgl.utils.maths import scatter_add

from ._core import MLP, GatedMLP

if TYPE_CHECKING:
    from collections.abc import Sequence


def _resolve_inputs(
    node_feat: Any, batch: torch.Tensor | None, num_graphs: int | None
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Normalise readout inputs to ``(node_feat, batch, num_graphs)``.

    Accepts either a feature tensor or a PyG graph carrying ``node_feat`` (legacy call style). A missing ``batch``
    means all nodes belong to a single graph.
    """
    if not isinstance(node_feat, torch.Tensor):
        graph = node_feat
        node_feat = graph.node_feat
        if node_feat is None:
            raise ValueError("Data object must contain node features (graph.node_feat)")
        if batch is None:
            batch = getattr(graph, "batch", None)
        if num_graphs is None and batch is not None:
            num_graphs = getattr(graph, "num_graphs", None)
    if batch is None:
        batch = torch.zeros(node_feat.size(0), dtype=torch.long, device=node_feat.device)
        num_graphs = 1 if num_graphs is None else num_graphs
    else:
        batch = batch.to(device=node_feat.device, dtype=torch.long)
    if num_graphs is None:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    return node_feat, batch, int(num_graphs)


class ReduceReadOut(nn.Module):
    """Reduce node (or edge) features into graph-level representations by ``sum``/``mean``/``max`` pooling."""

    def __init__(self, op: str = "mean", field: str = "node_feat"):
        """Initialize the ReduceReadOut.

        Args:
            op (str): Operation for the reduction ('mean', 'sum', or 'max').
            field (str): Name of the field being reduced. Informational; the caller passes the tensor to reduce.
        """
        super().__init__()
        self.op = op
        self.field = field
        if op not in ["mean", "sum", "max"]:
            raise ValueError("op must be 'mean', 'sum', or 'max'")

    def forward(
        self,
        node_feat: torch.Tensor,
        batch: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Pool per-node features into per-graph features.

        Args:
            node_feat: Per-node features, shape ``(num_nodes, feature_dim)``, or a PyG graph with ``node_feat``.
            batch: Per-node graph index, shape ``(num_nodes,)``. ``None`` treats all nodes as one graph.
            num_graphs: Number of graphs in the batch. Inferred from ``batch`` if ``None``.

        Returns:
            Pooled features, shape ``(num_graphs, feature_dim)``. Graphs without nodes give zeros.
        """
        node_feat, batch, num_graphs = _resolve_inputs(node_feat, batch, num_graphs)
        shape = (num_graphs, node_feat.size(1))
        if self.op == "max":
            out = torch.full(shape, float("-inf"), device=node_feat.device, dtype=node_feat.dtype)
            out = out.index_reduce(0, batch, node_feat, "amax", include_self=False)
            counts = torch.bincount(batch, minlength=num_graphs)
            return torch.where(counts.unsqueeze(1) > 0, out, torch.zeros_like(out))
        out = torch.zeros(shape, device=node_feat.device, dtype=node_feat.dtype).index_add(0, batch, node_feat)
        if self.op == "mean":
            counts = torch.bincount(batch, minlength=num_graphs).clamp(min=1)
            out = out / counts.unsqueeze(1).to(out.dtype)
        return out


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
    """Weighted atom readout for graph properties.

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

    def forward(
        self,
        node_feat: torch.Tensor,
        batch: torch.Tensor | None = None,
        num_graphs: int | None = None,
    ) -> torch.Tensor:
        """Aggregate weighted node features into graph-level representations.

        Args:
            node_feat: Node features, shape ``(num_nodes, in_feats)``, or a PyG graph with ``node_feat``.
            batch: Per-node graph index, shape ``(num_nodes,)``. ``None`` treats all nodes as one graph.
            num_graphs: Number of graphs in the batch. Inferred from ``batch`` if ``None``.

        Returns:
            Graph-level tensor with shape ``(num_graphs, output_dim)``. Graphs without nodes give zeros.
        """
        node_feat, batch, num_graphs = _resolve_inputs(node_feat, batch, num_graphs)

        # updated_field = self.mlp(field)
        updated_field = self.mlp(node_feat)  # [num_nodes, output_dim]

        # weights = self.weight(field)
        weights = self.weight_activation(self.weight(node_feat))  # [num_nodes, 1]

        # sum_j w_j for each graph
        weight_sum = scatter_add(weights, batch, dim_size=num_graphs, dim=0)  # [num_graphs, 1]

        # factor_i = w_i / sum_j w_j
        factor = weights / weight_sum[batch].clamp(min=1e-8)  # [num_nodes, 1]

        # sum_i factor_i * updated_field_i
        return scatter_add(factor * updated_field, batch, dim_size=num_graphs, dim=0)  # [num_graphs, output_dim]


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
