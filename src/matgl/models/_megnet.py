"""PyTorch Geometric implementation of MEGNet.

Uses ``edge_index`` / scatter-based message passing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch import nn

from matgl.config import DEFAULT_ELEMENTS
from matgl.graph._compute import compute_pair_vector_and_distance
from matgl.layers import (
    MLP,
    ActivationFunction,
    BondExpansion,
    EdgeSet2Set,
    EmbeddingBlock,
    MEGNetBlock,
    Set2SetReadOut,
)

from ._core import MatGLModel

if TYPE_CHECKING:
    from typing import Any

    from matgl.graph.converters import GraphConverter

logger = logging.getLogger(__name__)


class MEGNet(MatGLModel):
    """PyG implementation of MEGNet."""

    __version__ = 1

    def __init__(
        self,
        dim_node_embedding: int = 16,
        dim_edge_embedding: int = 100,
        dim_state_embedding: int = 2,
        ntypes_state: int | None = None,
        nblocks: int = 3,
        hidden_layer_sizes_input: tuple[int, ...] = (64, 32),
        hidden_layer_sizes_conv: tuple[int, ...] = (64, 64, 32),
        hidden_layer_sizes_output: tuple[int, ...] = (32, 16),
        nlayers_set2set: int = 1,
        niters_set2set: int = 2,
        activation_type: str = "softplus2",
        is_classification: bool = False,
        include_state: bool = True,
        dropout: float = 0.0,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        bond_expansion: BondExpansion | None = None,
        cutoff: float = 4.0,
        gauss_width: float = 0.5,
        **kwargs,
    ):
        """Initialize MEGNet."""
        super().__init__()

        self.save_args(locals(), kwargs)

        self.element_types = element_types or DEFAULT_ELEMENTS
        self.cutoff = cutoff
        self.bond_expansion = bond_expansion or BondExpansion(
            rbf_type="Gaussian", initial=0.0, final=cutoff + 1.0, num_centers=dim_edge_embedding, width=gauss_width
        )

        node_dims = [dim_node_embedding, *hidden_layer_sizes_input]
        edge_dims = [dim_edge_embedding, *hidden_layer_sizes_input]
        state_dims = [dim_state_embedding, *hidden_layer_sizes_input]

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        self.embedding = EmbeddingBlock(
            degree_rbf=dim_edge_embedding,
            dim_node_embedding=dim_node_embedding,
            ntypes_node=len(self.element_types),
            ntypes_state=ntypes_state,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding,
            activation=activation,
        )

        self.edge_encoder = MLP(edge_dims, activation, activate_last=True)
        self.node_encoder = MLP(node_dims, activation, activate_last=True)
        self.state_encoder = MLP(state_dims, activation, activate_last=True)

        dim_blocks_in = hidden_layer_sizes_input[-1]
        dim_blocks_out = hidden_layer_sizes_conv[-1]
        block_args = {
            "conv_hiddens": list(hidden_layer_sizes_conv),
            "dropout": dropout,
            "act": activation,
            "skip": True,
        }
        blocks = [MEGNetBlock(dims=[dim_blocks_in], **block_args)] + [  # type: ignore[arg-type]
            MEGNetBlock(dims=[dim_blocks_out, *hidden_layer_sizes_input], **block_args)  # type: ignore[arg-type]
            for _ in range(nblocks - 1)
        ]
        self.blocks = nn.ModuleList(blocks)

        s2s_kwargs = {"n_iters": niters_set2set, "n_layers": nlayers_set2set}
        self.edge_s2s = EdgeSet2Set(dim_blocks_out, **s2s_kwargs)
        self.node_s2s = Set2SetReadOut(dim_blocks_out, **s2s_kwargs)  # type: ignore[arg-type]

        self.output_proj = MLP(
            # 2*S2S(out=2*dim) + state -> output
            dims=[2 * 2 * dim_blocks_out + dim_blocks_out, *hidden_layer_sizes_output, 1],
            activation=activation,
            activate_last=False,
        )

        self.dropout = nn.Dropout(dropout) if dropout else None

        self.is_classification = is_classification
        self.include_state_embedding = include_state

    def forward(self, g: Any, state_attr: torch.Tensor | None = None, **kwargs):
        """Forward pass of MEGNet (PyG).

        Intermediate layer features are stored on ``self.feature_dict`` (keys
        ``edge_attr``, ``embedding``, ``gc_<i>``, ``readout``, ``final``) and
        overwritten on every call.

        Args:
            g: PyG ``Data`` (or ``Data``-like) object with attributes ``node_type``
                (or ``z``), ``pos``, ``edge_index``, and optionally ``pbc_offshift``,
                ``batch`` and ``num_graphs``.
            state_attr: Per-graph state attributes, shape ``(num_graphs, ...)``
                (or ``(...,)`` for a single graph).
            **kwargs: Reserved for future extensions.
        """
        node_attr = getattr(g, "node_type", getattr(g, "z", None))
        pos = g.pos
        edge_index = g.edge_index
        pbc_offshift = getattr(g, "pbc_offshift", None)
        batch = getattr(g, "batch", None)
        num_graphs = getattr(g, "num_graphs", None)
        num_nodes = pos.size(0)
        if num_graphs is None:
            num_graphs = 1 if batch is None else int(batch.max().item()) + 1

        edge_batch = None if batch is None else batch[edge_index[0]].to(torch.long)

        _, bond_dist = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)
        edge_attr = self.bond_expansion(bond_dist)

        node_feat, edge_feat, state_feat = self.embedding(node_attr, edge_attr, state_attr)
        edge_feat = self.edge_encoder(edge_feat)
        node_feat = self.node_encoder(node_feat)
        state_feat = self.state_encoder(state_feat)

        fea_dict: dict = {
            "edge_attr": edge_attr,
            "embedding": {"node_feat": node_feat, "edge_feat": edge_feat, "state_feat": state_feat},
        }

        # Ensure state_feat has a leading num_graphs dimension for the conv layers.
        if state_feat.dim() == 1:
            state_feat = state_feat.unsqueeze(0)

        for i, block in enumerate(self.blocks):
            edge_feat, node_feat, state_feat = block(
                edge_index,
                edge_feat,
                node_feat,
                state_feat,
                batch,
                edge_batch,
                num_nodes,
                num_graphs,
            )
            if state_feat.dim() == 1:
                state_feat = state_feat.unsqueeze(0)
            fea_dict[f"gc_{i + 1}"] = {
                "node_feat": node_feat,
                "edge_feat": edge_feat,
                "state_feat": state_feat,
            }

        node_vec = self.node_s2s(node_feat, batch)
        edge_vec = self.edge_s2s(edge_feat, edge_batch, num_graphs=num_graphs)

        node_vec = node_vec.view(num_graphs, -1)
        edge_vec = edge_vec.view(num_graphs, -1)
        state_vec = state_feat.view(num_graphs, -1)

        vec = torch.cat([node_vec, edge_vec, state_vec], dim=-1)
        fea_dict["readout"] = vec

        if self.dropout:
            vec = self.dropout(vec)

        output = self.output_proj(vec)
        if self.is_classification:
            output = torch.sigmoid(output)

        output = torch.squeeze(output)
        fea_dict["final"] = output
        self.feature_dict = fea_dict
        return output

    def predict_structure(  # type: ignore[override]
        self,
        structure,
        state_attr: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
    ):
        """Convenience method to directly predict a property from a structure.

        Args:
            structure: Input crystal/molecule (pymatgen ``Structure``/``Molecule`` or ASE ``Atoms``).
            state_attr: Graph attributes (optional).
            graph_converter: Custom converter; defaults to one matching the input type.

        Returns:
            Output property tensor.
        """
        return super().predict_structure(structure, state_feats=state_attr, graph_converter=graph_converter)
