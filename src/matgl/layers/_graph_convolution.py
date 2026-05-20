"""Graph-convolution / interaction blocks for M3GNet, MEGNet, TensorNet, and CHGNet.

Each block consumes a ``torch_geometric`` ``Data`` / ``Batch`` graph (plus,
for M3GNet, an optional line graph) and updates node, edge, and optional
state features:

* :class:`MEGNetGraphConv` / :class:`MEGNetBlock` -- the classic
  edge-node-state update of MEGNet.
* :class:`M3GNetGraphConv` / :class:`M3GNetBlock` -- M3GNet's two-body
  edge update, three-body angular term, and optional state coupling.
* :class:`TensorNetInteraction` -- the equivariant Cartesian-tensor
  message passing of TensorNet (works under either ``O(3)`` or ``SO(3)``).
* CHGNet atom-/bond-/line-graph blocks.
"""

from __future__ import annotations

import itertools

import torch
from torch import Tensor, nn
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
        return X + dX + torch.matmul(dX, dX)

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

    Direct port of :class:`MEGNetGraphConv`
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
    """A M3GNet graph convolution layer.

    Uses explicit gather + scatter ops. Edge messages are scattered into the
    *source* node of each edge (post-#761 aggregation convention).
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


# ---------------------------------------------------------------------------
# CHGNet PyG convolution layers
# ---------------------------------------------------------------------------


class _MLPNorm(nn.Module):
    """MLP with optional LayerNorm — mirrors DGL MLPNorm key/index structure exactly.

    norm index layout (matches DGL norm_layers):
      hidden layers: norms[0..n_hidden-1] only if normalize_hidden=True
      last layer:    norms[-1] always (when normalization is set)
    """

    def __init__(
        self,
        dims: list[int],
        activation: nn.Module,
        activate_last: bool = True,
        bias_last: bool = True,
        normalize_hidden: bool = False,
        normalization: str | None = None,
    ) -> None:
        super().__init__()
        self._depth = len(dims) - 1
        self.layers = nn.ModuleList()
        self.norms: nn.ModuleList | None = nn.ModuleList() if normalization == "layer" else None
        self.activation = activation
        self.activate_last = activate_last
        self.normalize_hidden = normalize_hidden

        for i, (in_d, out_d) in enumerate(itertools.pairwise(dims)):
            is_last = i == self._depth - 1
            self.layers.append(nn.Linear(in_d, out_d, bias=True if not is_last else bias_last))
            if self.norms is not None:
                if not is_last:
                    if normalize_hidden:
                        self.norms.append(nn.LayerNorm(out_d))
                else:
                    self.norms.append(nn.LayerNorm(out_d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self._depth - 1):
            x = self.layers[i](x)
            if self.norms is not None and self.normalize_hidden:
                x = self.norms[i](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        if self.norms is not None:
            x = self.norms[-1](x)
        if self.activate_last:
            x = self.activation(x)
        return x


class _GatedMLPNorm(nn.Module):
    """Gated MLP with optional LayerNorm — graph-backend-agnostic."""

    def __init__(
        self,
        in_feats: int,
        dims: list[int],
        activation: nn.Module,
        activate_last: bool = True,
        normalize_hidden: bool = False,
        normalization: str | None = None,
        bias_last: bool = True,
    ) -> None:
        super().__init__()
        all_dims = [in_feats, *dims]
        self.value = _MLPNorm(
            all_dims,
            activation,
            activate_last=activate_last,
            bias_last=bias_last,
            normalize_hidden=normalize_hidden,
            normalization=normalization,
        )
        self.gate = _MLPNorm(
            all_dims,
            activation,
            activate_last=False,
            bias_last=bias_last,
            normalize_hidden=normalize_hidden,
            normalization=normalization,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(x) * self.sigmoid(self.gate(x))


class CHGNetGraphConv(nn.Module):
    """CHGNet atom-graph convolution layer (PyG backend).

    Mirrors the DGL implementation: edges are directed center→neighbor, so messages
    are accumulated at the destination (neighbor) node, matching DGL ``fn.sum`` semantics.
    """

    def __init__(
        self,
        node_update_func: nn.Module,
        node_out_func: nn.Module,
        edge_update_func: nn.Module | None,
        node_weight_func: nn.Module | None,
        edge_weight_func: nn.Module | None,
        state_update_func: nn.Module | None,
    ) -> None:
        super().__init__()
        self.include_state = state_update_func is not None
        self.node_update_func = node_update_func
        self.node_out_func = node_out_func
        self.node_weight_func = node_weight_func
        self.edge_update_func = edge_update_func
        self.edge_weight_func = edge_weight_func
        self.state_update_func = state_update_func

    @classmethod
    def from_dims(
        cls,
        activation: nn.Module,
        node_dims: list[int],
        edge_dims: list[int] | None = None,
        state_dims: list[int] | None = None,
        normalization: str | None = None,
        normalize_hidden: bool = False,
        rbf_order: int = 0,
    ) -> CHGNetGraphConv:
        """Build a ``CHGNetGraphConv`` from layer dimension lists."""
        node_update_func = _GatedMLPNorm(
            node_dims[0], node_dims[1:], activation, normalization=normalization, normalize_hidden=normalize_hidden
        )
        node_out_func = nn.Linear(node_dims[-1], node_dims[-1], bias=False)
        node_weight_func = nn.Linear(rbf_order, node_dims[-1], bias=False) if rbf_order > 0 else None

        edge_update_func = (
            _GatedMLPNorm(
                edge_dims[0], edge_dims[1:], activation, normalization=normalization, normalize_hidden=normalize_hidden
            )
            if edge_dims is not None
            else None
        )
        edge_weight_func = (
            nn.Linear(rbf_order, edge_dims[-1], bias=False) if rbf_order > 0 and edge_dims is not None else None
        )
        from matgl.layers._core import MLP

        state_update_func = MLP(state_dims, activation, activate_last=True) if state_dims is not None else None

        return cls(
            node_update_func, node_out_func, edge_update_func, node_weight_func, edge_weight_func, state_update_func
        )

    # ------------------------------------------------------------------
    # Edge update (per-edge, no aggregation direction issue)
    # ------------------------------------------------------------------
    def edge_update_(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        bond_expansion: torch.Tensor,
        state_feat_per_edge: torch.Tensor | None,
        shared_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        atom_i = node_features[src]
        atom_j = node_features[dst]
        if self.include_state and state_feat_per_edge is not None:
            inputs = torch.cat([atom_i, edge_features, atom_j, state_feat_per_edge], dim=-1)
        else:
            inputs = torch.cat([atom_i, edge_features, atom_j], dim=-1)
        assert self.edge_update_func is not None
        edge_update = self.edge_update_func(inputs)
        if self.edge_weight_func is not None:
            edge_update = edge_update * self.edge_weight_func(bond_expansion.float())
        if shared_weights is not None:
            edge_update = edge_update * shared_weights
        return edge_update

    # ------------------------------------------------------------------
    # Node update -- scatter onto DST (neighbor), matching DGL fn.sum semantics.
    # Edges go center(src)->neighbor(dst); fn.sum accumulates at dst.
    # ------------------------------------------------------------------
    def node_update_(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        bond_expansion: torch.Tensor,
        state_feat_per_edge: torch.Tensor | None,
        shared_weights: torch.Tensor | None,
        num_nodes: int,
    ) -> torch.Tensor:
        atom_i = node_features[src]  # central atom
        atom_j = node_features[dst]  # neighbor
        if self.include_state and state_feat_per_edge is not None:
            inputs = torch.cat([atom_i, edge_features, atom_j, state_feat_per_edge], dim=-1)
        else:
            inputs = torch.cat([atom_i, edge_features, atom_j], dim=-1)
        messages = self.node_update_func(inputs)
        if self.node_weight_func is not None:
            messages = messages * self.node_weight_func(bond_expansion.float())
        if shared_weights is not None:
            messages = messages * shared_weights
        # Scatter onto DST (neighbor) -- matches DGL fn.sum aggregation direction
        feat_update = scatter_add(messages, dst, dim=0, dim_size=num_nodes)
        return self.node_out_func(feat_update)

    def state_update_(
        self,
        node_features: torch.Tensor,
        state_attr: torch.Tensor,
        batch: torch.Tensor | None,
        num_graphs: int,
    ) -> torch.Tensor:
        if batch is None:
            node_avg = node_features.mean(dim=0, keepdim=True)
        else:
            node_avg = scatter_add(node_features, batch.long(), dim=0, dim_size=num_graphs)
            counts = torch.bincount(batch.long(), minlength=num_graphs).float().clamp(min=1)
            node_avg = node_avg / counts.unsqueeze(1)
        inputs = torch.cat([state_attr, node_avg], dim=-1)
        assert self.state_update_func is not None
        return self.state_update_func(inputs)

    def forward(
        self,
        edge_index: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        bond_expansion: torch.Tensor,
        state_attr: torch.Tensor | None,
        batch: torch.Tensor | None,
        shared_node_weights: torch.Tensor | None,
        shared_edge_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        src, dst = edge_index[0], edge_index[1]
        num_nodes = node_features.size(0)
        num_graphs = (int(batch.max().item()) + 1) if batch is not None else 1

        state_per_edge: torch.Tensor | None = None
        if self.include_state and state_attr is not None:
            if batch is None:
                state_per_edge = state_attr.expand(edge_index.size(1), -1)
            else:
                edge_batch = batch[src]
                state_per_edge = state_attr[edge_batch]

        # Edge update (optional)
        if self.edge_update_func is not None:
            edge_update = self.edge_update_(
                src, dst, node_features, edge_features, bond_expansion, state_per_edge, shared_edge_weights
            )
            new_edge_features = edge_features + edge_update
        else:
            new_edge_features = edge_features

        # Node update -- scatter onto dst (neighbor atoms)
        node_update = self.node_update_(
            src, dst, node_features, new_edge_features, bond_expansion, state_per_edge, shared_node_weights, num_nodes
        )
        new_node_features = node_features + node_update

        # State update (optional)
        if self.include_state and state_attr is not None:
            state_attr = self.state_update_(new_node_features, state_attr, batch, num_graphs)

        return new_node_features, new_edge_features, state_attr


class CHGNetAtomGraphBlock(nn.Module):
    """CHGNet atom-graph block wrapping ``CHGNetGraphConv``."""

    def __init__(
        self,
        num_atom_feats: int,
        num_bond_feats: int,
        activation: nn.Module,
        atom_hidden_dims: list[int],
        bond_hidden_dims: list[int] | None = None,
        normalization: str | None = None,
        normalize_hidden: bool = False,
        num_state_feats: int | None = None,
        rbf_order: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        node_input_dim = 2 * num_atom_feats + num_bond_feats
        if num_state_feats is not None:
            node_input_dim += num_state_feats
            state_dims = [num_atom_feats + num_state_feats, *atom_hidden_dims, num_state_feats]
        else:
            state_dims = None
        node_dims = [node_input_dim, *atom_hidden_dims, num_atom_feats]
        edge_dims = [node_input_dim, *bond_hidden_dims, num_bond_feats] if bond_hidden_dims is not None else None

        self.conv = CHGNetGraphConv.from_dims(
            activation=activation,
            node_dims=node_dims,
            edge_dims=edge_dims,
            state_dims=state_dims,
            normalization=normalization,
            normalize_hidden=normalize_hidden,
            rbf_order=rbf_order,
        )
        self.atom_norm = nn.LayerNorm(num_atom_feats) if normalization == "layer" else None
        self.bond_norm = nn.LayerNorm(num_bond_feats) if normalization == "layer" else None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        edge_index: torch.Tensor,
        atom_features: torch.Tensor,
        bond_features: torch.Tensor,
        bond_expansion: torch.Tensor,
        state_attr: torch.Tensor | None,
        batch: torch.Tensor | None,
        shared_node_weights: torch.Tensor | None,
        shared_edge_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        atom_features, bond_features, state_attr = self.conv(
            edge_index,
            atom_features,
            bond_features,
            bond_expansion,
            state_attr,
            batch,
            shared_node_weights,
            shared_edge_weights,
        )
        atom_features = self.dropout(atom_features)
        bond_features = self.dropout(bond_features)
        if self.atom_norm is not None:
            atom_features = self.atom_norm(atom_features)
        if self.bond_norm is not None:
            bond_features = self.bond_norm(bond_features)
        if state_attr is not None:
            state_attr = self.dropout(state_attr)
        return atom_features, bond_features, state_attr


class CHGNetLineGraphConv(nn.Module):
    """CHGNet bond-graph (line-graph) convolution layer (PyG backend).

    Updates bond features via message passing over the directed line graph, and
    optionally updates angle features.  In the directed line graph ``dst`` is the
    bond being updated (convention set at construction time), so accumulating
    onto ``dst`` is *correct* here — no fix needed.
    """

    def __init__(
        self,
        node_update_func: nn.Module,
        node_out_func: nn.Module,
        edge_update_func: nn.Module | None,
        node_weight_func: nn.Module | None,
    ) -> None:
        super().__init__()
        self.node_update_func = node_update_func
        self.node_out_func = node_out_func
        self.node_weight_func = node_weight_func
        self.edge_update_func = edge_update_func

    @classmethod
    def from_dims(
        cls,
        node_dims: list[int],
        edge_dims: list[int] | None = None,
        activation: nn.Module | None = None,
        normalization: str | None = None,
        normalize_hidden: bool = False,
        node_weight_input_dims: int = 0,
    ) -> CHGNetLineGraphConv:
        act = activation or nn.SiLU()
        node_update_func = _GatedMLPNorm(
            node_dims[0], node_dims[1:], act, normalization=normalization, normalize_hidden=normalize_hidden
        )
        node_out_func = nn.Linear(node_dims[-1], node_dims[-1], bias=False)
        node_weight_func = nn.Linear(node_weight_input_dims, node_dims[-1]) if node_weight_input_dims > 0 else None
        edge_update_func = (
            _GatedMLPNorm(
                edge_dims[0], edge_dims[1:], act, normalization=normalization, normalize_hidden=normalize_hidden
            )
            if edge_dims is not None
            else None
        )
        return cls(node_update_func, node_out_func, edge_update_func, node_weight_func)

    def node_update_(
        self,
        lg_edge_index: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        aux_edge_features: torch.Tensor,
        bond_expansion: torch.Tensor | None,
        shared_weights: torch.Tensor | None,
        num_lg_nodes: int,
    ) -> torch.Tensor:
        lg_src, lg_dst = lg_edge_index[0], lg_edge_index[1]
        bonds_i = node_features[lg_src]
        bonds_j = node_features[lg_dst]
        inputs = torch.cat([bonds_i, edge_features, aux_edge_features, bonds_j], dim=-1)
        messages = self.node_update_func(inputs)

        if self.node_weight_func is not None and bond_expansion is not None:
            weights_i = self.node_weight_func(bond_expansion[lg_src])
            weights_j = self.node_weight_func(bond_expansion[lg_dst])
            messages = messages * weights_i * weights_j
        if shared_weights is not None:
            messages = messages * shared_weights[lg_src] * shared_weights[lg_dst]

        # Accumulate onto dst (bond being updated) -- correct for line graph
        feat_update = scatter_add(messages, lg_dst.long(), dim=0, dim_size=num_lg_nodes)
        return self.node_out_func(feat_update)

    def edge_update_(
        self,
        lg_edge_index: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        aux_edge_features: torch.Tensor,
    ) -> torch.Tensor:
        lg_src, lg_dst = lg_edge_index[0], lg_edge_index[1]
        bonds_i = node_features[lg_src]
        bonds_j = node_features[lg_dst]
        inputs = torch.cat([bonds_i, edge_features, aux_edge_features, bonds_j], dim=-1)
        assert self.edge_update_func is not None
        return self.edge_update_func(inputs)

    def forward(
        self,
        lg_edge_index: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        aux_edge_features: torch.Tensor,
        bond_expansion: torch.Tensor | None,
        shared_node_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_lg_nodes = node_features.size(0)

        node_update = self.node_update_(
            lg_edge_index,
            node_features,
            edge_features,
            aux_edge_features,
            bond_expansion,
            shared_node_weights,
            num_lg_nodes,
        )
        new_node_features = node_features + node_update

        if self.edge_update_func is not None:
            edge_update = self.edge_update_(lg_edge_index, new_node_features, edge_features, aux_edge_features)
            new_edge_features = edge_features + edge_update
        else:
            new_edge_features = edge_features

        return new_node_features, new_edge_features


class CHGNetBondGraphBlock(nn.Module):
    """CHGNet bond-graph block wrapping ``CHGNetLineGraphConv``."""

    def __init__(
        self,
        num_atom_feats: int,
        num_bond_feats: int,
        num_angle_feats: int,
        activation: nn.Module,
        bond_hidden_dims: list[int],
        angle_hidden_dims: list[int] | None,
        normalization: str | None = None,
        normalize_hidden: bool = False,
        rbf_order: int = 0,
        bond_dropout: float = 0.0,
        angle_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        node_input_dim = 2 * num_bond_feats + num_angle_feats + num_atom_feats
        node_dims = [node_input_dim, *bond_hidden_dims, num_bond_feats]
        edge_dims = [node_input_dim, *angle_hidden_dims, num_angle_feats] if angle_hidden_dims is not None else None

        self.conv = CHGNetLineGraphConv.from_dims(
            node_dims=node_dims,
            edge_dims=edge_dims,
            activation=activation,
            normalization=normalization,
            normalize_hidden=normalize_hidden,
            node_weight_input_dims=rbf_order,
        )
        self.bond_dropout = nn.Dropout(bond_dropout) if bond_dropout > 0.0 else nn.Identity()
        self.angle_dropout = nn.Dropout(angle_dropout) if angle_dropout > 0.0 else nn.Identity()

    def forward(
        self,
        lg_edge_index: torch.Tensor,
        bond_features: torch.Tensor,
        angle_features: torch.Tensor,
        atom_features: torch.Tensor,
        bond_index: torch.Tensor,
        center_atom_index: torch.Tensor,
        bond_expansion: torch.Tensor | None,
        shared_node_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Line-graph nodes carry features for bonds within threebody cutoff
        node_features = bond_features[bond_index]
        aux_edge_features = atom_features[center_atom_index]

        new_node_features, new_angle_features = self.conv(
            lg_edge_index,
            node_features,
            angle_features,
            aux_edge_features,
            bond_expansion,
            shared_node_weights,
        )

        new_node_features = self.bond_dropout(new_node_features)
        new_angle_features = self.angle_dropout(new_angle_features)

        # Write updated bond features back to the full bond feature tensor
        new_bond_features = bond_features.clone()
        new_bond_features[bond_index] = new_node_features

        return new_bond_features, new_angle_features
