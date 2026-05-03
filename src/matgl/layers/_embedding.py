"""Backend-agnostic embedding blocks.

Holds embedding modules whose ``forward`` operates on plain tensors and is
therefore reusable from both DGL and PyG model implementations. Backend-
specific embeddings (``TensorEmbedding`` for TensorNet, ``NeighborEmbedding``,
etc.) live in ``_embedding_dgl.py`` / ``_embedding_pyg.py``.
"""

from __future__ import annotations

import torch
from torch import nn

import matgl
from matgl.layers._core import MLP


class EmbeddingBlock(nn.Module):
    """Embedding block for generating node, bond and state features.

    Operates purely on tensors; works with both DGL and PyG model code.
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
