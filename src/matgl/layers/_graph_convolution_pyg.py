from __future__ import annotations

import torch
import torch.nn as nn

from matgl.utils.cutoff import cosine_cutoff
from matgl.utils.maths import (
    decompose_tensor,
    new_radial_tensor,
    scatter_add,
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
        """
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
        """
        Args:
        edge_index (torch.Tensor):
            Graph connectivity in COO format specifying source and target nodes.
            Shape: (2, num_edges).

        edge_weight (torch.Tensor):
            Edge distance between source and target nodes.
            Shape: (num_edges,) or (num_edges, 1).

        edge_attr (torch.Tensor):
            Edge-wise attributes encoding geometric or chemical information.
            Shape: (num_edges, num_edge_features).

        X (torch.Tensor):
            Node feature representations.
            Shape: (num_nodes, hidden_channels).

        Returns:
        X (torch.Tensor):
            Updated node feature representations after message passing.
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


# ---------------------------------------------------------------------------
# CHGNet PyG convolution layers
# ---------------------------------------------------------------------------


class _MLPNorm(nn.Module):
    """MLP with optional LayerNorm — graph-backend-agnostic."""

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
        self.layers = nn.ModuleList()
        self.norms: nn.ModuleList | None = nn.ModuleList() if normalization == "layer" else None
        self.activation = activation
        self.activate_last = activate_last
        self.normalize_hidden = normalize_hidden

        for i, (in_d, out_d) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
            is_last = i == len(dims) - 2
            self.layers.append(nn.Linear(in_d, out_d, bias=True if not is_last else bias_last))
            if self.norms is not None:
                if normalize_hidden or is_last:
                    self.norms.append(nn.LayerNorm(out_d))
                else:
                    self.norms.append(nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.norms is not None:
                x = self.norms[i](x)
            if i < n - 1 or self.activate_last:
                x = self.activation(x)
        return x


class _GatedMLPNorm(nn.Module):
    """Gated MLP with optional LayerNorm — graph-backend-agnostic."""

    def __init__(
        self,
        in_feats: int,
        dims: list[int],
        activation: nn.Module,
        normalize_hidden: bool = False,
        normalization: str | None = None,
        bias_last: bool = True,
    ) -> None:
        super().__init__()
        all_dims = [in_feats, *dims]
        self.value = _MLPNorm(
            all_dims, activation, activate_last=True, bias_last=bias_last,
            normalize_hidden=normalize_hidden, normalization=normalization,
        )
        self.gate = _MLPNorm(
            all_dims, activation, activate_last=False, bias_last=bias_last,
            normalize_hidden=normalize_hidden, normalization=normalization,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(x) * self.sigmoid(self.gate(x))


class CHGNetGraphConv(nn.Module):
    """CHGNet atom-graph convolution layer (PyG backend).

    Fixes the DGL bug: atom messages are scattered onto the **source** (central)
    atom, not the destination (neighbor) atom.
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
        gmlp_kw = dict(activation=activation, normalization=normalization, normalize_hidden=normalize_hidden)

        node_update_func = _GatedMLPNorm(node_dims[0], node_dims[1:], **gmlp_kw)
        node_out_func = nn.Linear(node_dims[-1], node_dims[-1], bias=False)
        node_weight_func = nn.Linear(rbf_order, node_dims[-1], bias=False) if rbf_order > 0 else None

        edge_update_func = _GatedMLPNorm(edge_dims[0], edge_dims[1:], **gmlp_kw) if edge_dims is not None else None
        edge_weight_func = (
            nn.Linear(rbf_order, edge_dims[-1], bias=False)
            if rbf_order > 0 and edge_dims is not None else None
        )
        from matgl.layers._core import MLP
        state_update_func = MLP(state_dims, activation, activate_last=True) if state_dims is not None else None

        return cls(node_update_func, node_out_func, edge_update_func, node_weight_func, edge_weight_func, state_update_func)

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
        edge_update = self.edge_update_func(inputs)
        if self.edge_weight_func is not None:
            edge_update = edge_update * self.edge_weight_func(bond_expansion.float())
        if shared_weights is not None:
            edge_update = edge_update * shared_weights
        return edge_update

    # ------------------------------------------------------------------
    # Node update — scatter onto SRC (central atom) — THE BUG FIX
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
        atom_i = node_features[src]   # central atom
        atom_j = node_features[dst]   # neighbor
        if self.include_state and state_feat_per_edge is not None:
            inputs = torch.cat([atom_i, edge_features, atom_j, state_feat_per_edge], dim=-1)
        else:
            inputs = torch.cat([atom_i, edge_features, atom_j], dim=-1)
        messages = self.node_update_func(inputs)
        if self.node_weight_func is not None:
            messages = messages * self.node_weight_func(bond_expansion.float())
        if shared_weights is not None:
            messages = messages * shared_weights
        # Scatter onto SRC (central atom) — fixes the DGL aggregation bug
        feat_update = scatter_add(messages, src, dim=0, dim_size=num_nodes)
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
            edge_update = self.edge_update_(src, dst, node_features, edge_features, bond_expansion, state_per_edge, shared_edge_weights)
            new_edge_features = edge_features + edge_update
        else:
            new_edge_features = edge_features

        # Node update — scatter onto src (central atoms)
        node_update = self.node_update_(src, dst, node_features, new_edge_features, bond_expansion, state_per_edge, shared_node_weights, num_nodes)
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
            edge_index, atom_features, bond_features, bond_expansion,
            state_attr, batch, shared_node_weights, shared_edge_weights,
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
        gmlp_kw = dict(activation=activation or nn.SiLU(), normalization=normalization, normalize_hidden=normalize_hidden)
        node_update_func = _GatedMLPNorm(node_dims[0], node_dims[1:], **gmlp_kw)
        node_out_func = nn.Linear(node_dims[-1], node_dims[-1], bias=False)
        node_weight_func = nn.Linear(node_weight_input_dims, node_dims[-1]) if node_weight_input_dims > 0 else None
        edge_update_func = _GatedMLPNorm(edge_dims[0], edge_dims[1:], **gmlp_kw) if edge_dims is not None else None
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

        # Accumulate onto dst (bond being updated) — correct for line graph
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
            lg_edge_index, node_features, edge_features, aux_edge_features,
            bond_expansion, shared_node_weights, num_lg_nodes,
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
            lg_edge_index, node_features, angle_features, aux_edge_features,
            bond_expansion, shared_node_weights,
        )

        new_node_features = self.bond_dropout(new_node_features)
        new_angle_features = self.angle_dropout(new_angle_features)

        # Write updated bond features back to the full bond feature tensor
        new_bond_features = bond_features.clone()
        new_bond_features[bond_index] = new_node_features

        return new_bond_features, new_angle_features
