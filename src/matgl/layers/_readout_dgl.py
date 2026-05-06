"""Readout layer for M3GNet."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dgl
import torch
import torch.nn.functional as F
from dgl.nn import Set2Set
from torch import nn

from matgl.layers._activations import SoftPlus2

from ._core import MLP, GatedMLP
from ._core_dgl import EdgeSet2Set

if TYPE_CHECKING:
    from collections.abc import Sequence


class Set2SetReadOut(nn.Module):
    """The Set2Set readout function."""

    def __init__(
        self,
        in_feats: int,
        n_iters: int,
        n_layers: int,
        field: Literal["node_feat", "edge_feat"],
    ):
        """Initialize the Set2SetReadOut.

        Args:
            in_feats (int): length of input feature vector
            n_iters (int): Number of LSTM steps
            n_layers (int): Number of layers.
            field (str): Field of graph to perform the readout.
        """
        super().__init__()
        self.field = field
        self.n_iters = n_iters
        self.n_layers = n_layers
        if field == "node_feat":
            self.set2set = Set2Set(in_feats, n_iters, n_layers)
        elif field == "edge_feat":
            self.set2set = EdgeSet2Set(in_feats, n_iters, n_layers)
        else:
            raise ValueError("Field must be node_feat or edge_feat")

    def forward(self, g: dgl.DGLGraph):
        if self.field == "node_feat":
            return self.set2set(g, g.ndata["node_feat"])
        return self.set2set(g, g.edata["edge_feat"])


class ReduceReadOut(nn.Module):
    """Reduce atom or bond attributes into lower dimensional tensors as readout.

    This could be summing up the atoms or bonds, or taking the mean, etc.
    """

    def __init__(self, op: str = "mean", field: Literal["node_feat", "edge_feat"] = "node_feat"):
        """Initialize the ReduceReadOut.

        Args:
            op (str): op for the reduction
            field (str): Field of graph to perform the reduction.
        """
        super().__init__()
        self.op = op
        self.field = field

    def forward(self, g: dgl.DGLGraph):
        """Reduce attributes over a graph.

        Args:
            g: DGL graph.

        Returns:
            torch.tensor.
        """
        if self.field == "node_feat":
            return dgl.readout_nodes(g, feat="node_feat", op=self.op)
        return dgl.readout_edges(g, feat="edge_feat", op=self.op)


class WeightedReadOut(nn.Module):
    """Feed node features into Gated MLP as readout for atomic properties."""

    def __init__(self, in_feats: int, dims: Sequence[int], num_targets: int):
        """Initialize the WeightedReadOut.

        Args:
            in_feats: input features (nodes)
            dims: NN architecture for Gated MLP
            num_targets: number of target properties.
        """
        super().__init__()
        self.in_feats = in_feats
        self.dims = [in_feats, *dims, num_targets]
        self.gated = GatedMLP(in_feats=in_feats, dims=self.dims, activate_last=False)

    def forward(self, g: dgl.DGLGraph):
        """Run the weighted readout.

        Args:
            g: DGL graph.

        Returns:
            atomic_properties: torch.Tensor.
        """
        atomic_properties = self.gated(g.ndata["node_feat"])
        return atomic_properties


class WeightedAtomReadOut(nn.Module):
    """Weighted atom readout for graph properties.

    This follows the TensorFlow WeightedReadout implementation:

        updated_field = mlp(field)
        weights = weight_mlp(field)
        factor = weights / sum(weights)
        readout = sum(factor * updated_field)

    where the normalization is done independently for each graph.
    """

    def __init__(self, in_feats: int, dims: Sequence[int], activation: nn.Module):
        """
        Args:
            in_feats: Input node feature dimension.
            dims: MLP architecture. The final entry is the output dimension.
            activation: Activation function for multi-layer perceptrons.
        """
        super().__init__()

        self.dims = [in_feats, *dims]
        self.activation = activation

        # Equivalent to:
        # self.mlp = MLP(neurons=neurons, activations=[activation] * n_layer)
        self.mlp = MLP(
            dims=self.dims,
            activation=self.activation,
            activate_last=True,
        )

        # Equivalent to:
        # neurons = neurons[:-1] + [1]
        # acts = [activation] * (n_layer - 1) + ["sigmoid"]
        self.weight = MLP(
            dims=[*self.dims[:-1], 1],
            activation=self.activation,
            activate_last=False,
        )
        self.weight_activation = nn.Sigmoid()

    def forward(self, g: dgl.DGLGraph):
        """Run weighted atom readout.

        Args:
            g: DGL graph with ``g.ndata["node_feat"]``.

        Returns:
            atomic_properties: Tensor of shape ``[num_graphs, dims[-1]]``.
        """
        with g.local_scope():
            field = g.ndata["node_feat"]

            # updated_field = self.mlp(field)
            g.ndata["updated_field"] = self.mlp(field)

            # weights = self.weight(field)
            g.ndata["weights"] = self.weight_activation(self.weight(field))

            # sum_j w_j for each graph
            weight_sum = dgl.sum_nodes(g, "weights")

            # Broadcast graph-level sums back to nodes
            g.ndata["weight_sum"] = dgl.broadcast_nodes(g, weight_sum)

            # factor_i = w_i / sum_j w_j
            g.ndata["factor"] = g.ndata["weights"] / g.ndata["weight_sum"].clamp(min=1e-8)

            # sum_i factor_i * updated_field_i
            readout = dgl.sum_nodes(g, "updated_field", "factor")

        return readout


class WeightedReadOutPair(nn.Module):
    """Feed the average of atomic features i and j into weighted readout layer."""

    def __init__(self, in_feats, dims, num_targets, activation=None):
        super().__init__()
        self.in_feats = in_feats
        self.activation = activation
        self.num_targets = num_targets
        self.dims = [*dims, num_targets]
        self.gated = GatedMLP(in_feats=in_feats, dims=self.dims, activate_last=False)

    def forward(self, g: dgl.DGLGraph):
        num_nodes = g.ndata["node_feat"].size(dim=0)
        pair_properties = torch.zeros(num_nodes, num_nodes, self.num_targets)
        for i in range(num_nodes):
            for j in range(i, num_nodes):
                in_features = (g.ndata["node_feat"][i][:] + g.ndata["node_feat"][j][:]) / 2.0
                pair_properties[i][j][:] = self.gated(in_features)[:]
                pair_properties[j][i][:] = pair_properties[i][j][:]
        return pair_properties


class GlobalPool(nn.Module):
    """One-step readout in AttentiveFP. Token from dgllife.model.readout.attentivefp_readout.

    Parameters
    ----------
    feat_size : int
        Size for the input node features, graph features and output graph
        representations.
    dropout : float
        The probability for performing dropout.
    """

    def __init__(self, feat_size: int, dropout: float = 0.0):
        super().__init__()

        self.compute_logits = nn.Sequential(nn.Linear(2 * feat_size, 1), SoftPlus2())
        self.project_nodes = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_size, feat_size))
        self.gru = nn.GRUCell(feat_size, feat_size)

    def forward(self, g, node_feats, g_feats, get_node_weight=False):
        """Perform one-step readout.

        Args:
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs.
        node_feats : float32 tensor of shape (V, node_feat_size)
            Input node features. V for the number of nodes.
        g_feats : float32 tensor of shape (G, graph_feat_size)
            Input graph features. G for the number of graphs.
        get_node_weight : bool
            Whether to get the weights of atoms during readout.

        Returns:
        -------
        float32 tensor of shape (G, graph_feat_size)
            Updated graph features.
        float32 tensor of shape (V, 1)
            The weights of nodes in readout.
        """
        with g.local_scope():
            g.ndata["z"] = self.compute_logits(torch.cat([dgl.broadcast_nodes(g, F.relu(g_feats)), node_feats], dim=1))
            g.ndata["a"] = dgl.softmax_nodes(g, "z")
            g.ndata["hv"] = self.project_nodes(node_feats)

            g_repr = dgl.sum_nodes(g, "hv", "a")
            context = F.elu(g_repr)

            if get_node_weight:
                return self.gru(context, g_feats), g.ndata["a"]

            return self.gru(context, g_feats)


class AttentiveFPReadout(nn.Module):
    """Readout in AttentiveFP. Token from dgllife.model.readout.attentivefp_readout.

    AttentiveFP is introduced in `Pushing the Boundaries of Molecular Representation for
    Drug Discovery with the Graph Attention Mechanism
    <https://www.ncbi.nlm.nih.gov/pubmed/31408336>`__

    This class computes graph representations out of node features.

    Parameters
    ----------
    feat_size : int
        Size for the input node features, graph features and output graph
        representations.
    num_timesteps : int
        Times of updating the graph representations with GRU. Default to 2.
    dropout : float
        The probability for performing dropout. Default to 0.
    """

    def __init__(self, feat_size, num_timesteps=2, dropout=0.0):
        super().__init__()

        self.readouts = nn.ModuleList()
        for _ in range(num_timesteps):
            self.readouts.append(GlobalPool(feat_size, dropout))

    def forward(self, g: dgl.DGLGraph, node_feats: torch.Tensor, get_node_weight=False):
        """Computes graph representations out of node features.

        Args:
            g (dgl.DGLGraph): DGLGraph for a batch of graphs.
            node_feats (torch.Tenor): Input node features. V for the number of nodes.
            get_node_weight (bool):  Whether to get the weights of nodes in readout. Default to False.

        Returns:
            g_feats (torch.Tensor): float32 tensor of shape (G, graph_feat_size)
                Graph representations computed. G for the number of graphs.
            node_weights (torch.Tensor): list of float32 tensor of shape (V, 1), optional
                This is returned when ``get_node_weight`` is ``True``.

        """
        with g.local_scope():
            g.ndata["hv"] = node_feats
            g_feats = dgl.sum_nodes(g, "hv")
        node_weights = None
        if get_node_weight:
            node_weights = []

        for readout in self.readouts:
            if get_node_weight:
                g_feats, node_weights_t = readout(g, node_feats, g_feats, get_node_weight)
                node_weights.append(node_weights_t)
            else:
                g_feats = readout(g, node_feats, g_feats)

        if get_node_weight:
            return g_feats, torch.hstack(node_weights)

        return g_feats
