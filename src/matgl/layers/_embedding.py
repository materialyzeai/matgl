"""Embedding blocks for matgl graph models.

Holds the backend-agnostic :class:`EmbeddingBlock` plus the TensorNet-style
:class:`TensorEmbedding` that maps atomic numbers and edge vectors into a
Cartesian rank-2 tensor representation ``(N, units, 3, 3)``.
"""

from __future__ import annotations

import torch
from torch import nn

import matgl
from matgl.layers._core import MLP
from matgl.utils.cutoff import cosine_cutoff
from matgl.utils.maths import (
    new_radial_tensor,
    scatter_add,
    tensor_norm,
    vector_to_skewtensor,
    vector_to_symtensor,
)


class EmbeddingBlock(nn.Module):
    """Embedding block for generating node, bond and state features.

    Operates purely on tensors.
    """

    def __init__(
        self,
        degree_rbf: int,
        activation: nn.Module,
        dim_node_embedding: int,
        dim_edge_embedding: int | None = None,
        dim_state_feats: int | None = None,
        ntypes_node: int | None = None,
        include_state: bool = False,
        ntypes_state: int | None = None,
        dim_state_embedding: int | None = None,
    ):
        """Initialize the embedding block.

        Args:
            degree_rbf (int): number of rbf
            activation (nn.Module): activation type
            dim_node_embedding (int): dimensionality of node features
            dim_edge_embedding (int): dimensionality of edge features
            dim_state_feats: dimensionality of state features
            ntypes_node: number of node labels
            include_state: Whether to include state embedding
            ntypes_state: number of state labels
            dim_state_embedding: dimensionality of state embedding.
        """
        super().__init__()
        self.include_state = include_state
        self.ntypes_state = ntypes_state
        self.dim_node_embedding = dim_node_embedding
        self.dim_edge_embedding = dim_edge_embedding
        self.dim_state_feats = dim_state_feats
        self.ntypes_node = ntypes_node
        self.dim_state_embedding = dim_state_embedding
        self.activation = activation
        if ntypes_state and dim_state_embedding is not None:
            self.layer_state_embedding = nn.Embedding(ntypes_state, dim_state_embedding)  # type: ignore
        elif dim_state_feats is not None:
            self.layer_state_embedding = nn.Sequential(  # type:ignore[assignment]
                nn.LazyLinear(dim_state_feats, bias=False, dtype=matgl.float_th),
                activation,
            )
        if ntypes_node is not None:
            self.layer_node_embedding = nn.Embedding(ntypes_node, dim_node_embedding)
        else:
            self.layer_node_embedding = nn.Sequential(  # type:ignore[assignment]
                nn.LazyLinear(dim_node_embedding, bias=False, dtype=matgl.float_th),
                activation,
            )
        if dim_edge_embedding is not None:
            dim_edges = [degree_rbf, dim_edge_embedding]
            self.layer_edge_embedding = MLP(dim_edges, activation=activation, activate_last=True)

    def forward(self, node_attr, edge_attr, state_attr):
        """Output embedded features.

        Args:
            node_attr: node attribute
            edge_attr: edge attribute
            state_attr: state attribute

        Returns:
            node_feat: embedded node features
            edge_feat: embedded edge features
            state_feat: embedded state features
        """
        if self.ntypes_node is not None:
            node_feat = self.layer_node_embedding(node_attr)
        else:
            node_feat = self.layer_node_embedding(node_attr.to(matgl.float_th))
        if self.dim_edge_embedding is not None:
            edge_feat = self.layer_edge_embedding(edge_attr.to(matgl.float_th))
        else:
            edge_feat = edge_attr
        if self.include_state is True:
            if self.ntypes_state and self.dim_state_embedding is not None:
                state_feat = self.layer_state_embedding(state_attr)
            elif self.dim_state_feats is not None:
                state_attr = torch.unsqueeze(state_attr, 0)
                state_feat = self.layer_state_embedding(state_attr.to(matgl.float_th))
            else:
                state_feat = state_attr
        else:
            state_feat = None
        return node_feat, edge_feat, state_feat


class TensorEmbedding(nn.Module):
    """Embedding block for TensorNet to generate node, edge, and optional state features.

    Adapted from https://github.com/torchmd/torchmd-net.
    """

    def __init__(
        self,
        units: int,
        degree_rbf: int,
        activation: nn.Module,
        ntypes_node: int,
        cutoff: float,
        dtype: torch.dtype = matgl.float_th,
    ):
        """Initialize the TensorEmbedding.

        Args:
            units (int): Number of hidden neurons.
            degree_rbf (int): Number of radial basis functions.
            activation (nn.Module): Activation function.
            ntypes_node (int): Number of node labels.
            cutoff (float): Cutoff radius for graph construction.
            dtype (torch.dtype): Data type for all variables.
        """
        super().__init__()
        self.units = units
        self.cutoff = cutoff

        # Initialize layers
        self.distance_proj1 = nn.Linear(degree_rbf, units, dtype=dtype)
        self.distance_proj2 = nn.Linear(degree_rbf, units, dtype=dtype)
        self.distance_proj3 = nn.Linear(degree_rbf, units, dtype=dtype)
        self.emb = nn.Embedding(ntypes_node, units, dtype=dtype)
        self.emb2 = nn.Linear(2 * units, units, dtype=dtype)
        self.act = activation
        self.linears_tensor = nn.ModuleList([nn.Linear(units, units, bias=False, dtype=dtype) for _ in range(3)])
        self.linears_scalar = nn.ModuleList(
            [
                nn.Linear(units, 2 * units, bias=True, dtype=dtype),
                nn.Linear(2 * units, 3 * units, bias=True, dtype=dtype),
            ]
        )
        self.init_norm = nn.LayerNorm(units, dtype=dtype)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize the parameters."""
        self.distance_proj1.reset_parameters()
        self.distance_proj2.reset_parameters()
        self.distance_proj3.reset_parameters()
        self.emb.reset_parameters()
        self.emb2.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()
        for linear in self.linears_scalar:
            linear.reset_parameters()
        self.init_norm.reset_parameters()

    def message(self, x_i, x_j, edge_attr, edge_weight, Iij, Aij, Sij):
        """Message function for edge updates."""
        vi = x_i  # Source node features
        vj = x_j  # Destination node features
        # Concatenate node and state features
        zij = torch.cat([vi, vj], dim=-1)
        Zij = self.emb2(zij)
        scalars = Zij[..., None, None] * Iij
        skew_matrices = Zij[..., None, None] * Aij
        traceless_tensors = Zij[..., None, None] * Sij
        return {"I": scalars, "A": skew_matrices, "S": traceless_tensors}

    def aggregate(self, msg, index, dim_size=None):
        """Aggregate messages for node updates."""
        scalars = scatter_add(msg["I"], index, dim_size=dim_size)
        skew_matrices = scatter_add(msg["A"], index, dim_size=dim_size)
        traceless_tensors = scatter_add(msg["S"], index, dim_size=dim_size)
        return scalars, skew_matrices, traceless_tensors

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_vec: torch.Tensor,
        state_attr=None,
    ):
        """Compute embedded node tensors and (optional) state features.

        Args:
            z (torch.Tensor): Node-wise features, e.g. atomic embeddings or species encodings.
                Shape: (num_nodes, hidden_channels) or (num_nodes,).
            edge_index (torch.Tensor): Graph connectivity in COO format specifying source and target nodes.
                Shape: (2, num_edges).
            edge_attr (torch.Tensor): Edge-wise attributes encoding geometric or chemical information
                (e.g. radial basis expansion or bond features). Shape: (num_edges, num_edge_features).
            edge_weight (torch.Tensor): Edge distance between source and target nodes.
                Shape: (num_edges,) or (num_edges, 1).
            edge_vec (torch.Tensor): Edge direction vectors from source to target nodes.
                Shape: (num_edges, 3).
            state_attr (torch.Tensor, optional): Global (graph-level) state attributes shared by all nodes
                (e.g. external conditions or control parameters). Shape: (num_graphs, state_dim).

        Returns:
            x (torch.Tensor): Updated node representations after message passing.
                Shape: (num_nodes, hidden_channels).
            state_feat (torch.Tensor): Updated global state features.
                Shape: (num_graphs, state_hidden_channels).
        """
        # Node embedding
        x = self.emb(z)  # Assuming node_type is integer for embedding

        # set dummy state features for now and nothing to do for state_attr
        state_feat = None

        # Edge processing
        C = cosine_cutoff(edge_weight, self.cutoff)
        W1 = self.distance_proj1(edge_attr) * C.view(-1, 1)
        W2 = self.distance_proj2(edge_attr) * C.view(-1, 1)
        W3 = self.distance_proj3(edge_attr) * C.view(-1, 1)
        edge_vec = edge_vec / torch.norm(edge_vec, dim=1, keepdim=True).clamp(min=1e-6)

        # Radial tensor components
        Iij, Aij, Sij = new_radial_tensor(
            torch.eye(3, 3, device=edge_vec.device, dtype=edge_vec.dtype).unsqueeze(0).unsqueeze(0),
            vector_to_skewtensor(edge_vec).unsqueeze(-3),
            vector_to_symtensor(edge_vec).unsqueeze(-3),
            W1,
            W2,
            W3,
        )

        # Perform message passing
        msg = self.message(
            x_i=x[edge_index[0]],
            x_j=x[edge_index[1]],
            edge_attr=edge_attr,
            edge_weight=edge_weight,
            Iij=Iij,
            Aij=Aij,
            Sij=Sij,
        )

        scalars, skew_matrices, traceless_tensors = self.aggregate(msg, edge_index[0], dim_size=x.size(0))

        # Node update
        norm = tensor_norm(scalars + skew_matrices + traceless_tensors)
        norm = self.init_norm(norm)
        scalars = self.linears_tensor[0](scalars.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        skew_matrices = self.linears_tensor[1](skew_matrices.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        traceless_tensors = self.linears_tensor[2](traceless_tensors.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for linear_scalar in self.linears_scalar:
            norm = self.act(linear_scalar(norm))
        norm = norm.reshape(norm.shape[0], self.units, 3)
        scalars, skew_matrices, traceless_tensors = new_radial_tensor(
            scalars, skew_matrices, traceless_tensors, norm[..., 0], norm[..., 1], norm[..., 2]
        )
        X = scalars + skew_matrices + traceless_tensors

        return X, state_feat
