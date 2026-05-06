from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Dropout, Identity, Module

from matgl.layers._core import MLP, GatedMLP
from matgl.utils.cutoff import cosine_cutoff
from matgl.utils.maths import (
    decompose_tensor,
    new_radial_tensor,
    scatter_add,
    scatter_mean,
    tensor_norm,
)


class TensorNetInteraction(nn.Module):
    """A Graph Convolution block for TensorNet adapted for PyTorch Geometric."""

    def __init__(
        self,
        num_rbf: int,
        units: int,
        activation: nn.Module,
        cutoff: float,
        equivariance_invariance_group: str,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize the TensorNetInteraction.

        Args:
            num_rbf: Number of radial basis functions.
            units: Number of hidden neurons.
            activation: Activation function.
            cutoff: Cutoff radius for graph construction.
            equivariance_invariance_group: Group action on geometric tensor representations, either O(3) or SO(3).
            dtype: Data type for all variables.
        """
        super().__init__()
        self.num_rbf = num_rbf
        self.units = units
        self.cutoff = cutoff
        self.equivariance_invariance_group = equivariance_invariance_group

        # Scalar linear layers
        self.linears_scalar = nn.ModuleList(
            [
                nn.Linear(num_rbf, units, bias=True, dtype=dtype),
                nn.Linear(units, 2 * units, bias=True, dtype=dtype),
                nn.Linear(2 * units, 3 * units, bias=True, dtype=dtype),
            ]
        )

        # Tensor linear layers (6 layers for scalar, skew, and traceless components)
        self.linears_tensor = nn.ModuleList([nn.Linear(units, units, bias=False, dtype=dtype) for _ in range(6)])

        self.act = activation
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize the parameters."""
        for linear in self.linears_scalar:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        for linear in self.linears_tensor:
            nn.init.xavier_uniform_(linear.weight)

    def forward(
        self, edge_index: torch.Tensor, edge_weight: torch.Tensor, edge_attr: torch.Tensor, X: torch.Tensor
    ) -> torch.Tensor:
        """Run the TensorNet interaction.

        Args:
            edge_index (torch.Tensor): Graph connectivity in COO format specifying source and target nodes.
                Shape: (2, num_edges).
            edge_weight (torch.Tensor): Edge distance between source and target nodes.
                Shape: (num_edges,) or (num_edges, 1).
            edge_attr (torch.Tensor): Edge-wise attributes encoding geometric or chemical information.
                Shape: (num_edges, num_edge_features).
            X (torch.Tensor): Node feature representations.
                Shape: (num_nodes, hidden_channels).

        Returns:
            X (torch.Tensor): Updated node feature representations after message passing.
                Shape: (num_nodes, hidden_channels).
        """
        # Process edge attributes
        C = cosine_cutoff(edge_weight, self.cutoff)
        for linear_scalar in self.linears_scalar:
            edge_attr = self.act(linear_scalar(edge_attr))
        edge_attr_processed = (edge_attr * C.view(-1, 1)).reshape(edge_attr.shape[0], self.units, 3)

        # Normalize input tensor
        X = X / (tensor_norm(X) + 1)[..., None, None]

        # Decompose input tensor
        scalars, skew_metrices, traceless_tensors = decompose_tensor(X)

        # Apply tensor linear transformations
        scalars = self.linears_tensor[0](scalars.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        skew_metrices = self.linears_tensor[1](skew_metrices.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        traceless_tensors = self.linears_tensor[2](traceless_tensors.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        Y = scalars + skew_metrices + traceless_tensors

        # Message passing
        x_I = scalars
        x_A = skew_metrices
        x_S = traceless_tensors

        messages = self.message(edge_index, x_I, x_A, x_S, edge_attr_processed)
        Im, Am, Sm = self.aggregate(messages, edge_index[0], X.size(0))
        # Combine messages
        msg = Im + Am + Sm

        # Apply group action
        if self.equivariance_invariance_group == "O(3)":
            A = torch.matmul(msg, Y)
            B = torch.matmul(Y, msg)
            scalars, skew_metrices, traceless_tensors = decompose_tensor(A + B)
        elif self.equivariance_invariance_group == "SO(3)":
            B = torch.matmul(Y, msg)
            scalars, skew_metrices, traceless_tensors = decompose_tensor(2 * B)
        else:
            raise ValueError("equivariance_invariance_group must be 'O(3)' or 'SO(3)'")

        # Normalize and apply final tensor transformations
        normp1 = (tensor_norm(scalars + skew_metrices + traceless_tensors) + 1)[..., None, None]
        scalars = scalars / normp1
        skew_metrices = skew_metrices / normp1
        traceless_tensors = traceless_tensors / normp1

        scalars = self.linears_tensor[3](scalars.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        skew_metrices = self.linears_tensor[4](skew_metrices.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        traceless_tensors = self.linears_tensor[5](traceless_tensors.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Compute update
        dX = scalars + skew_metrices + traceless_tensors
        X = X + dX + torch.matmul(dX, dX)

        return X

    def message(self, edge_index, x_I: torch.Tensor, x_A: torch.Tensor, x_S: torch.Tensor, edge_attr: torch.Tensor):
        """Compute messages for each edge."""
        _, dst = edge_index
        x_I_j = x_I[dst]
        x_A_j = x_A[dst]
        x_S_j = x_S[dst]
        scalars, skew_metrices, traceless_tensors = new_radial_tensor(
            x_I_j, x_A_j, x_S_j, edge_attr[..., 0], edge_attr[..., 1], edge_attr[..., 2]
        )
        return scalars, skew_metrices, traceless_tensors

    def aggregate(self, inputs, index, dim_size):
        """Aggregate messages for node updates."""
        scalars, skew_matrices, traceless_tensors = inputs
        scalars_agg = scatter_add(scalars, index, dim_size=dim_size)
        skew_matrices_agg = scatter_add(skew_matrices, index, dim_size=dim_size)
        traceless_tensors_agg = scatter_add(traceless_tensors, index, dim_size=dim_size)
        return scalars_agg, skew_matrices_agg, traceless_tensors_agg


def _broadcast_to_nodes(state_feat: Tensor, batch: Tensor | None, target_size: int) -> Tensor:
    """Replicate per-graph state features across the nodes/edges of each graph.

    Mirrors ``dgl.broadcast_nodes`` / ``dgl.broadcast_edges``. When ``batch`` is
    provided, ``state_feat`` is indexed by ``batch``. When ``batch`` is ``None``
    (single-graph case), ``state_feat`` is replicated to ``target_size`` rows.
    """
    if state_feat.dim() == 1:
        state_feat = state_feat.unsqueeze(0)
    if batch is None:
        return state_feat.expand(target_size, -1)
    return state_feat[batch.to(torch.long)]


def _per_graph_mean(feat: Tensor, batch: Tensor | None, num_graphs: int) -> Tensor:
    """Per-graph mean of node/edge features. Mirrors ``dgl.readout_*`` with op=mean."""
    if batch is None:
        return feat.mean(dim=0, keepdim=True)
    return scatter_mean(feat, batch.to(torch.long), dim_size=num_graphs, dim=0)


class MEGNetGraphConv(Module):
    """A MEGNet graph convolution layer in PyG.

    Direct port of :class:`matgl.layers._graph_convolution_dgl.MEGNetGraphConv`
    using ``edge_index`` + scatter primitives. Aggregation convention matches
    the DGL implementation post-#761: edge messages are aggregated into the
    *source* node of each edge with ``scatter_mean``.
    """

    def __init__(self, edge_func: Module, node_func: Module, state_func: Module) -> None:
        """Initialize a MEGNet graph convolution layer.

        Args:
            edge_func: Edge update function.
            node_func: Node update function.
            state_func: Global state update function.
        """
        super().__init__()
        self.edge_func = edge_func
        self.node_func = node_func
        self.state_func = state_func

    @staticmethod
    def from_dims(
        edge_dims: list[int],
        node_dims: list[int],
        state_dims: list[int],
        activation: Module,
    ) -> MEGNetGraphConv:
        """Create a MEGNetGraphConv from layer dimensions."""
        edge_update = MLP(edge_dims, activation, activate_last=True)
        node_update = MLP(node_dims, activation, activate_last=True)
        state_update = MLP(state_dims, activation, activate_last=True)
        return MEGNetGraphConv(edge_update, node_update, state_update)

    def edge_update_(
        self,
        edge_index: Tensor,
        edge_feat: Tensor,
        node_feat: Tensor,
        u_per_edge: Tensor,
    ) -> Tensor:
        """Edge update: concat(vi, vj, eij, u) -> MLP."""
        src, dst = edge_index[0], edge_index[1]
        vi = node_feat[src]
        vj = node_feat[dst]
        inputs = torch.hstack([vi, vj, edge_feat, u_per_edge])
        return self.edge_func(inputs)

    def node_update_(
        self,
        edge_index: Tensor,
        edge_feat: Tensor,
        node_feat: Tensor,
        u_per_node: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """Node update: aggregate edge messages into source node, then MLP."""
        src = edge_index[0]
        ve = scatter_mean(edge_feat, src, dim_size=num_nodes, dim=0)
        inputs = torch.hstack([node_feat, ve, u_per_node])
        return self.node_func(inputs)

    def state_update_(
        self,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_feat: Tensor,
        edge_batch: Tensor | None,
        node_batch: Tensor | None,
        num_graphs: int,
    ) -> Tensor:
        """Global update: per-graph mean of edges + nodes concatenated with u, then MLP."""
        u_edge = _per_graph_mean(edge_feat, edge_batch, num_graphs)
        u_vertex = _per_graph_mean(node_feat, node_batch, num_graphs)
        u_edge = torch.squeeze(u_edge)
        u_vertex = torch.squeeze(u_vertex)
        inputs = torch.hstack([state_feat.squeeze(), u_edge, u_vertex])
        return self.state_func(inputs)

    def forward(
        self,
        edge_index: Tensor,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_feat: Tensor,
        node_batch: Tensor | None,
        edge_batch: Tensor | None,
        num_nodes: int,
        num_graphs: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run a full edge -> node -> state update.

        Args:
            edge_index: COO connectivity, shape ``(2, num_edges)``.
            edge_feat: Per-edge features, shape ``(num_edges, edge_dim)``.
            node_feat: Per-node features, shape ``(num_nodes, node_dim)``.
            state_feat: Per-graph state features, shape ``(num_graphs, state_dim)``.
            node_batch: Per-node batch index. ``None`` is treated as a single-graph batch.
            edge_batch: Per-edge batch index (typically ``node_batch[edge_index[0]]``).
            num_nodes: Total number of nodes across all graphs.
            num_graphs: Number of graphs in the batch.
        """
        num_edges = edge_index.size(1)
        u_per_node = _broadcast_to_nodes(state_feat, node_batch, num_nodes)
        u_per_edge = _broadcast_to_nodes(state_feat, edge_batch, num_edges)
        edge_feat_new = self.edge_update_(edge_index, edge_feat, node_feat, u_per_edge)
        node_feat_new = self.node_update_(edge_index, edge_feat_new, node_feat, u_per_node, num_nodes)
        state_feat_new = self.state_update_(
            edge_feat_new, node_feat_new, state_feat, edge_batch, node_batch, num_graphs
        )
        return edge_feat_new, node_feat_new, state_feat_new


class MEGNetBlock(Module):
    """A MEGNet block (PyG): pre-MLPs, conv, optional dropout, optional skip."""

    def __init__(
        self,
        dims: list[int],
        conv_hiddens: list[int],
        act: Module,
        dropout: float | None = None,
        skip: bool = True,
    ) -> None:
        """Initialize a MEGNet block.

        Args:
            dims: Dimensions of the dense layers applied before the convolution.
            conv_hiddens: Hidden-layer architecture of the inner ``MEGNetGraphConv``.
            act: Activation module.
            dropout: Dropout probability (``None`` disables dropout).
            skip: Whether to add a residual connection around the block.
        """
        super().__init__()
        self.has_dense = len(dims) > 1
        self.activation = act
        conv_dim = dims[-1]
        out_dim = conv_hiddens[-1]

        mlp_kwargs = {
            "dims": dims,
            "activation": self.activation,
            "activate_last": True,
            "bias_last": True,
        }
        self.edge_func = MLP(**mlp_kwargs) if self.has_dense else Identity()  # type: ignore
        self.node_func = MLP(**mlp_kwargs) if self.has_dense else Identity()  # type: ignore
        self.state_func = MLP(**mlp_kwargs) if self.has_dense else Identity()  # type: ignore

        edge_in = 2 * conv_dim + conv_dim + conv_dim  # 2*NDIM+EDIM+GDIM
        node_in = out_dim + conv_dim + conv_dim
        state_in = out_dim + out_dim + conv_dim
        self.conv = MEGNetGraphConv.from_dims(
            edge_dims=[edge_in, *conv_hiddens],
            node_dims=[node_in, *conv_hiddens],
            state_dims=[state_in, *conv_hiddens],
            activation=self.activation,
        )

        self.dropout = Dropout(dropout) if dropout else None
        self.skip = skip

    def forward(
        self,
        edge_index: Tensor,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_feat: Tensor,
        node_batch: Tensor | None,
        edge_batch: Tensor | None,
        num_nodes: int,
        num_graphs: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the MEGNet block."""
        inputs = (edge_feat, node_feat, state_feat)
        edge_feat = self.edge_func(edge_feat)
        node_feat = self.node_func(node_feat)
        state_feat = self.state_func(state_feat)

        edge_feat, node_feat, state_feat = self.conv(
            edge_index, edge_feat, node_feat, state_feat, node_batch, edge_batch, num_nodes, num_graphs
        )

        if self.dropout:
            edge_feat = self.dropout(edge_feat)
            node_feat = self.dropout(node_feat)
            state_feat = self.dropout(state_feat)

        if self.skip:
            edge_feat = edge_feat + inputs[0]
            node_feat = node_feat + inputs[1]
            state_feat = state_feat + inputs[2]

        return edge_feat, node_feat, state_feat


class M3GNetGraphConv(Module):
    """A M3GNet graph convolution layer in PyG.

    Port of :class:`matgl.layers._graph_convolution_dgl.M3GNetGraphConv` using
    explicit gather + scatter ops. Aggregation convention matches the DGL
    implementation post-#761: edge messages are scattered into the *source*
    node of each edge.
    """

    def __init__(
        self,
        include_state: bool,
        edge_update_func: Module,
        edge_weight_func: Module,
        node_update_func: Module,
        node_weight_func: Module,
        state_update_func: Module | None,
    ):
        """Initialize the M3GNetGraphConv.

        Args:
            include_state: Whether to include state features in updates.
            edge_update_func: Gated MLP for edge updates (Eq. 4).
            edge_weight_func: Linear projection of the radial basis for edges.
            node_update_func: Gated MLP for node updates (Eq. 5).
            node_weight_func: Linear projection of the radial basis for nodes.
            state_update_func: MLP for state updates (Eq. 6); ignored when
                ``include_state`` is ``False``.
        """
        super().__init__()
        self.include_state = include_state
        self.edge_update_func = edge_update_func
        self.edge_weight_func = edge_weight_func
        self.node_update_func = node_update_func
        self.node_weight_func = node_weight_func
        self.state_update_func = state_update_func

    @staticmethod
    def from_dims(
        degree: int,
        include_state: bool,
        edge_dims: list[int],
        node_dims: list[int],
        state_dims: list[int] | None,
        activation: Module,
    ) -> M3GNetGraphConv:
        """Build an ``M3GNetGraphConv`` from layer dimensions."""
        edge_update_func = GatedMLP(in_feats=edge_dims[0], dims=edge_dims[1:])
        edge_weight_func = nn.Linear(in_features=degree, out_features=edge_dims[-1], bias=False)

        node_update_func = GatedMLP(in_feats=node_dims[0], dims=node_dims[1:])
        node_weight_func = nn.Linear(in_features=degree, out_features=node_dims[-1], bias=False)
        state_update_func = MLP(state_dims, activation, activate_last=True) if include_state else None  # type: ignore[arg-type]
        return M3GNetGraphConv(
            include_state, edge_update_func, edge_weight_func, node_update_func, node_weight_func, state_update_func
        )

    def edge_update_(
        self,
        edge_index: Tensor,
        edge_feat: Tensor,
        node_feat: Tensor,
        rbf: Tensor,
        u_per_edge: Tensor | None,
    ) -> Tensor:
        """Compute the edge-update message (Eq. 4)."""
        src, dst = edge_index[0], edge_index[1]
        vi = node_feat[src]
        vj = node_feat[dst]
        if self.include_state:
            assert u_per_edge is not None
            inputs = torch.hstack([vi, vj, edge_feat, u_per_edge])
        else:
            inputs = torch.hstack([vi, vj, edge_feat])
        return self.edge_update_func(inputs) * self.edge_weight_func(rbf)

    def node_update_(
        self,
        edge_index: Tensor,
        edge_feat: Tensor,
        node_feat: Tensor,
        rbf: Tensor,
        u_per_edge: Tensor | None,
        num_nodes: int,
    ) -> Tensor:
        """Compute the node-update message (Eq. 5) and scatter into source nodes."""
        src, dst = edge_index[0], edge_index[1]
        vi = node_feat[src]
        vj = node_feat[dst]
        if self.include_state:
            assert u_per_edge is not None
            inputs = torch.hstack([vi, vj, edge_feat, u_per_edge])
        else:
            inputs = torch.hstack([vi, vj, edge_feat])
        mess = self.node_update_func(inputs) * self.node_weight_func(rbf)
        return scatter_add(mess, src, dim_size=num_nodes, dim=0)

    def state_update_(
        self,
        node_feat: Tensor,
        state_feat: Tensor,
        node_batch: Tensor | None,
        num_graphs: int,
    ) -> Tensor:
        """Compute the state update (Eq. 6) using per-graph mean of node features."""
        uv = _per_graph_mean(node_feat, node_batch, num_graphs)
        inputs = torch.hstack([state_feat, uv])
        return self.state_update_func(inputs)  # type: ignore[misc]

    def forward(
        self,
        edge_index: Tensor,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_feat: Tensor | None,
        rbf: Tensor,
        node_batch: Tensor | None,
        edge_batch: Tensor | None,
        num_nodes: int,
        num_graphs: int,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Run a full edge -> node -> (optional) state update."""
        num_edges = edge_index.size(1)
        u_per_edge = (
            _broadcast_to_nodes(state_feat, edge_batch, num_edges)
            if self.include_state and state_feat is not None
            else None
        )
        edge_update = self.edge_update_(edge_index, edge_feat, node_feat, rbf, u_per_edge)
        edge_feat_new = edge_feat + edge_update
        node_update = self.node_update_(edge_index, edge_feat_new, node_feat, rbf, u_per_edge, num_nodes)
        node_feat_new = node_feat + node_update
        state_feat_new: Tensor | None = state_feat
        if self.include_state and state_feat is not None:
            state_feat_new = self.state_update_(node_feat_new, state_feat, node_batch, num_graphs)
        return edge_feat_new, node_feat_new, state_feat_new


class M3GNetBlock(Module):
    """A M3GNet block (PyG): wrapper around ``M3GNetGraphConv`` with optional dropout."""

    def __init__(
        self,
        degree: int,
        activation: Module,
        conv_hiddens: list[int],
        dim_node_feats: int,
        dim_edge_feats: int,
        dim_state_feats: int = 0,
        include_state: bool = False,
        dropout: float | None = None,
    ) -> None:
        """Initialize the M3GNet block.

        Args:
            degree: Number of radial basis functions feeding the weight branches.
            activation: Activation module.
            conv_hiddens: Hidden-layer dimensions for the inner gated MLPs.
            dim_node_feats: Per-node feature dimension.
            dim_edge_feats: Per-edge feature dimension.
            dim_state_feats: Global-state feature dimension (only used when
                ``include_state`` is ``True``).
            include_state: Whether the global state participates in updates.
            dropout: Dropout probability (``None`` disables dropout).
        """
        super().__init__()
        self.activation = activation
        self.include_state = include_state

        edge_in = 2 * dim_node_feats + dim_edge_feats + (dim_state_feats if include_state else 0)
        node_in = 2 * dim_node_feats + dim_edge_feats + (dim_state_feats if include_state else 0)
        state_in = dim_state_feats + dim_node_feats if include_state else 0

        self.conv = M3GNetGraphConv.from_dims(
            degree=degree,
            include_state=include_state,
            edge_dims=[edge_in, *conv_hiddens, dim_edge_feats],
            node_dims=[node_in, *conv_hiddens, dim_node_feats],
            state_dims=[state_in, *conv_hiddens, dim_state_feats] if include_state else None,
            activation=activation,
        )

        self.dropout = Dropout(dropout) if dropout else None

    def forward(
        self,
        edge_index: Tensor,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_feat: Tensor | None,
        rbf: Tensor,
        node_batch: Tensor | None,
        edge_batch: Tensor | None,
        num_nodes: int,
        num_graphs: int,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Run the M3GNet block."""
        edge_feat, node_feat, state_feat = self.conv(
            edge_index, edge_feat, node_feat, state_feat, rbf, node_batch, edge_batch, num_nodes, num_graphs
        )
        if self.dropout:
            edge_feat = self.dropout(edge_feat)
            node_feat = self.dropout(node_feat)
            if state_feat is not None:
                state_feat = self.dropout(state_feat)
        return edge_feat, node_feat, state_feat
