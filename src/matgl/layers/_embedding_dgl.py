"""Embedding node, edge and optional state attributes."""

from __future__ import annotations

import dgl
import torch
from torch import nn

import matgl
from matgl.layers._core import MLP
from matgl.utils.cutoff import cosine_cutoff
from matgl.utils.maths import new_radial_tensor, scatter_add, tensor_norm, vector_to_skewtensor, vector_to_symtensor


class EmbeddingBlock(nn.Module):
    """Embedding block for generating node, bond and state features."""

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
        """
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
    """Embedding block for TensorNet to generate node, edge and optional state features.
    The official implementation can be found in https://github.com/torchmd/torchmd-net.
    """

    def __init__(
        self,
        units: int,
        degree_rbf: int,
        activation: nn.Module,
        ntypes_node: int,
        cutoff: float,
        dtype: torch.dtype = matgl.float_th,
        include_state: bool = False,
        ntypes_state: int | None = None,
        dim_state_feats: int | None = None,
        dim_state_embedding: int = 0,
    ):
        """
        Args:
            units (int): number of hidden neurons
            degree_rbf (int): number of rbf
            activation (nn.Module): activation type
            ntypes_node: number of node labels
            cutoff (float): cutoff radius for graph construction
            dtype (torch.dtype): data type for all variables
            include_state: Whether to include state embedding
            ntypes_state (int): number of state labels
            dim_state_feats (int): number of global state features
            dim_state_embedding: dimensionality of state embedding.
        """
        super().__init__()

        self.units = units
        self.distance_proj1 = nn.Linear(degree_rbf, units, dtype=dtype)
        self.distance_proj2 = nn.Linear(degree_rbf, units, dtype=dtype)
        self.distance_proj3 = nn.Linear(degree_rbf, units, dtype=dtype)
        self.emb = torch.nn.Embedding(ntypes_node, units, dtype=dtype)
        #        self.act = activation()
        self.act = activation
        self.linears_tensor = nn.ModuleList()
        self.linears_tensor = nn.ModuleList([nn.Linear(units, units, bias=False) for _ in range(3)])
        self.linears_scalar = nn.ModuleList()
        self.linears_scalar.append(nn.Linear(units, 2 * units, bias=True, dtype=dtype))
        self.linears_scalar.append(nn.Linear(2 * units, 3 * units, bias=True, dtype=dtype))
        self.init_norm = nn.LayerNorm(units, dtype=dtype)
        self.cutoff = cutoff
        if ntypes_state is not None and dim_state_embedding > 0:
            self.layer_state_embedding = nn.Embedding(ntypes_state, dim_state_embedding)  # type: ignore
            self.emb2 = nn.Linear(2 * units + dim_state_embedding, units, dtype=dtype)  # type: ignore
        elif dim_state_feats is not None:
            self.layer_state_mlp = nn.Sequential(nn.LazyLinear(dim_state_feats, bias=False, dtype=dtype), activation)
            self.emb2 = nn.Linear(2 * units + dim_state_feats, units, dtype=dtype)
        else:
            self.emb2 = nn.Linear(2 * units, units, dtype=dtype)
        self.emb3 = nn.Linear(degree_rbf, units)
        self.dim_state_feats = dim_state_feats
        self.include_state = include_state
        self.ntypes_state = ntypes_state
        self.dim_state_embedding = dim_state_embedding
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize the parameters."""
        self.distance_proj1.reset_parameters()
        self.distance_proj2.reset_parameters()
        self.distance_proj3.reset_parameters()
        if self.dim_state_embedding > 0:
            self.layer_state_embedding.reset_parameters()
        if self.dim_state_feats is not None:
            for layer in self.layer_state_mlp:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        self.emb.reset_parameters()
        self.emb2.reset_parameters()
        self.emb3.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()
        for linear in self.linears_scalar:
            linear.reset_parameters()
        self.init_norm.reset_parameters()

    def _edge_udf(self, edges: dgl.udf.EdgeBatch):
        """Edge message function.

        Args:
            edges: input edge features.

        Returns:
            mij: message function.

        """
        vi = edges.src["v"]
        vj = edges.dst["v"]
        u = None
        if self.include_state:
            u = edges.src["u"]
        zij = torch.hstack([vi, vj, u]) if self.include_state else torch.hstack([vi, vj])  # type:ignore[list-item]
        Zij = self.emb2(zij)
        scalars = Zij[..., None, None] * edges.data.pop("Iij")
        skew_matrices = Zij[..., None, None] * edges.data.pop("Aij")
        traceless_tensors = Zij[..., None, None] * edges.data.pop("Sij")
        mij = {"I": scalars, "A": skew_matrices, "S": traceless_tensors}
        return mij

    def edge_update_(self, graph: dgl.DGLGraph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform edge update.

        :param graph: Input graph
        :return: Output tensor for edges.
        """
        graph.apply_edges(self._edge_udf)
        scalars = graph.edata.pop("I")
        skew_metrices = graph.edata.pop("A")
        traceless_tensors = graph.edata.pop("S")
        return scalars, skew_metrices, traceless_tensors

    def node_update_(self, graph: dgl.DGLGraph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate edge messages onto source nodes.

        Since edges are center -> neighbor, summing onto src makes the
        center atom collect information from all of its neighboring dst nodes.
        """
        src, _ = graph.edges()

        scalars = scatter_add(graph.edata["I"], src, dim=0, dim_size=graph.num_nodes())
        skew_matrices = scatter_add(graph.edata["A"], src, dim=0, dim_size=graph.num_nodes())
        traceless_tensors = scatter_add(graph.edata["S"], src, dim=0, dim_size=graph.num_nodes())

        return scalars, skew_matrices, traceless_tensors

    def forward(self, g: dgl.DGLGraph, state_attr: torch.Tensor | None = None):
        """

        Args:
            g: dgl graph.
            state_attr: global state attributes.

        Returns:
            X: embedded node tensor representation.
            edge_feat: embedded edge features.
            state_feat: embedded state features.
        """
        edge_attr = g.edata["edge_attr"]
        edge_weight = g.edata["bond_dist"]
        Z = self.emb(g.ndata["node_type"])
        C = cosine_cutoff(edge_weight, self.cutoff)
        edge_vec = g.edata["bond_vec"]
        W1 = self.distance_proj1(edge_attr) * C.view(-1, 1)
        W2 = self.distance_proj2(edge_attr) * C.view(-1, 1)
        W3 = self.distance_proj3(edge_attr) * C.view(-1, 1)
        edge_vec = edge_vec / torch.norm(edge_vec, dim=1).unsqueeze(1)
        Iij, Aij, Sij = new_radial_tensor(
            torch.eye(3, 3, device=edge_vec.device, dtype=edge_vec.dtype)[None, None, :, :],
            vector_to_skewtensor(edge_vec)[..., None, :, :],
            vector_to_symtensor(edge_vec)[..., None, :, :],
            W1,
            W2,
            W3,
        )
        state_feat = None
        if self.include_state is True:
            if self.ntypes_state and self.dim_state_embedding is not None:
                state_feat = self.layer_state_embedding(state_attr)
            elif self.dim_state_feats is not None:
                state_attr = torch.unsqueeze(state_attr, 0)  # type:ignore[arg-type]
                state_feat = self.layer_state_mlp(state_attr.to(matgl.float_th))

        edge_feat = self.emb3(edge_attr)
        with g.local_scope():
            g.edata["Iij"] = Iij
            g.edata["Aij"] = Aij
            g.edata["Sij"] = Sij
            g.ndata["v"] = Z
            if self.include_state:
                g.ndata["u"] = dgl.broadcast_nodes(g, state_feat)
            scalars, skew_metrices, traceless_tensors = self.edge_update_(g)
            g.edata["I"] = scalars
            g.edata["A"] = skew_metrices
            g.edata["S"] = traceless_tensors
            scalars, skew_metrices, traceless_tensors = self.node_update_(g)

            norm = tensor_norm(scalars + skew_metrices + traceless_tensors)
            norm = self.init_norm(norm)
            scalars = self.linears_tensor[0](scalars.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            skew_metrices = self.linears_tensor[1](skew_metrices.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            traceless_tensors = self.linears_tensor[2](traceless_tensors.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            for linear_scalar in self.linears_scalar:
                norm = self.act(linear_scalar(norm))
            norm = norm.reshape(norm.shape[0], self.units, 3)
            scalars, skew_metrices, traceless_tensors = new_radial_tensor(
                scalars,
                skew_metrices,
                traceless_tensors,
                norm[..., 0],
                norm[..., 1],
                norm[..., 2],
            )
            X = scalars + skew_metrices + traceless_tensors

        return X, edge_feat, state_feat


class NeighborEmbedding(nn.Module):
    def __init__(
        self,
        ntypes_node: int,
        hidden_channels: int,
        num_rbf: int,
        cutoff: float,
        dtype: torch.dtype = matgl.float_th,
    ):
        """
        The ET architecture assigns two  learned vectors to each atom type
        zi. One  is used to  encode information  specific to an  atom, the
        other (this  class) takes  the role  of a  neighborhood embedding.
        The neighborhood embedding, which is  an embedding of the types of
        neighboring atoms, is multiplied by a distance filter.


        This embedding allows  the network to store  information about the
        interaction of atom pairs.

        See eq. 3 in https://arxiv.org/pdf/2202.02541.pdf for more details.
        """
        super().__init__()
        self.embedding = nn.Embedding(ntypes_node, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels, dtype=dtype)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels, dtype=dtype)
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(
        self,
        z: torch.Tensor,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z (Tensor): Atomic numbers of shape [num_nodes].
            node_feat (Tensor): graph-convoluted node features [num_nodes, hidden_channels].
            edge_index (Tensor): Graph connectivity (list of neighbor pairs) with shape [2, num_edges].
            edge_weight (Tensor): Edge weight vector of shape [num_edges].
            edge_attr (Tensor): Edge attribute matrix of shape [num_edges, num_rbf].

        Returns:
            x_neighbors (Tensor): The embedding of the neighbors of each atom of shape [num_nodes, hidden_channels].
        """
        C = cosine_cutoff(edge_weight, self.cutoff)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        msg = W * x_neighbors.index_select(0, edge_index[1])
        x_neighbors = torch.zeros(
            node_feat.shape[0],
            node_feat.shape[1],
            dtype=node_feat.dtype,
            device=node_feat.device,
        ).index_add(0, edge_index[0], msg)
        x_neighbors = self.combine(torch.cat([node_feat, x_neighbors], dim=1))
        return x_neighbors
