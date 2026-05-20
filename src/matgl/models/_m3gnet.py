"""PyTorch Geometric implementation of M3GNet.

Uses ``edge_index`` / scatter-based message passing and a tensor-bundle
line graph from ``matgl.graph._compute.create_line_graph``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn

from matgl.config import DEFAULT_ELEMENTS
from matgl.graph._compute import (
    compute_pair_vector_and_distance,
    compute_theta_and_phi,
    create_line_graph,
    ensure_line_graph_compatibility,
)
from matgl.layers import (
    MLP,
    ActivationFunction,
    BondExpansion,
    EmbeddingBlock,
    GatedMLP,
    M3GNetBlock,
    Set2SetReadOut,
    SphericalBesselWithHarmonics,
    ThreeBodyInteractions,
)
from matgl.layers._readout_torch import ReduceReadOut, WeightedAtomReadOut, WeightedReadOut
from matgl.utils.cutoff import polynomial_cutoff

from ._core import MatGLModel

if TYPE_CHECKING:
    from matgl.graph._converters import GraphConverter

logger = logging.getLogger(__name__)


class M3GNet(MatGLModel):
    """PyG implementation of the M3GNet model."""

    __version__ = 2

    def __init__(
        self,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        dim_node_embedding: int = 64,
        dim_edge_embedding: int = 64,
        dim_state_embedding: int = 0,
        ntypes_state: int | None = None,
        dim_state_feats: int | None = None,
        max_n: int = 3,
        max_l: int = 3,
        nblocks: int = 3,
        rbf_type: Literal["Gaussian", "SphericalBessel"] = "SphericalBessel",
        is_intensive: bool = True,
        readout_type: Literal["set2set", "weighted_atom", "reduce_atom"] = "weighted_atom",
        task_type: Literal["classification", "regression"] = "regression",
        cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
        units: int = 64,
        ntargets: int = 1,
        use_smooth: bool = False,
        use_phi: bool = False,
        niters_set2set: int = 3,
        nlayers_set2set: int = 3,
        field: Literal["node_feat", "edge_feat"] = "node_feat",
        include_state: bool = False,
        activation_type: Literal["swish", "tanh", "sigmoid", "softplus2", "softexp"] = "swish",
        dropout: float | None = None,
        **kwargs,
    ):
        """Initialize the M3GNet model."""
        super().__init__()

        self.save_args(locals(), kwargs)

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        self.element_types = element_types or DEFAULT_ELEMENTS

        self.bond_expansion = BondExpansion(max_l, max_n, cutoff, rbf_type=rbf_type, smooth=use_smooth)

        degree = max_n * max_l * max_l if use_phi else max_n * max_l
        degree_rbf = max_n if use_smooth else max_n * max_l

        self.embedding = EmbeddingBlock(
            degree_rbf=degree_rbf,
            dim_node_embedding=dim_node_embedding,
            dim_edge_embedding=dim_edge_embedding,
            ntypes_node=len(element_types),
            ntypes_state=ntypes_state,
            dim_state_feats=dim_state_feats,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding,
            activation=activation,
        )

        self.basis_expansion = SphericalBesselWithHarmonics(
            max_n=max_n,
            max_l=max_l,
            cutoff=cutoff,
            use_phi=use_phi,
            use_smooth=use_smooth,
        )
        self.three_body_interactions = nn.ModuleList(
            [
                ThreeBodyInteractions(
                    update_network_atom=MLP(
                        dims=[dim_node_embedding, degree],
                        activation=nn.Sigmoid(),
                        activate_last=True,
                    ),
                    update_network_bond=GatedMLP(in_feats=degree, dims=[dim_edge_embedding], use_bias=False),
                )
                for _ in range(nblocks)
            ]
        )

        dim_state_feats_used = dim_state_embedding

        self.graph_layers = nn.ModuleList(
            [
                M3GNetBlock(
                    degree=degree_rbf,
                    activation=activation,
                    conv_hiddens=[units, units],
                    dim_node_feats=dim_node_embedding,
                    dim_edge_feats=dim_edge_embedding,
                    dim_state_feats=dim_state_feats_used,
                    include_state=include_state,
                    dropout=dropout,
                )
                for _ in range(nblocks)
            ]
        )

        if is_intensive:
            input_feats = dim_node_embedding if field == "node_feat" else dim_edge_embedding
            if readout_type == "set2set":
                if field != "node_feat":
                    raise NotImplementedError("Set2Set readout on edge features is not implemented for PyG yet.")
                self.readout = Set2SetReadOut(  # type: ignore[call-arg]
                    in_feats=input_feats, n_iters=niters_set2set, n_layers=nlayers_set2set
                )
                readout_feats = 2 * input_feats + dim_state_feats_used if include_state else 2 * input_feats
            elif readout_type == "weighted_atom":
                self.readout = WeightedAtomReadOut(  # type: ignore[assignment]
                    in_feats=input_feats, dims=[units, units], activation=activation
                )
                readout_feats = units + dim_state_feats_used if include_state else units
            else:
                self.readout = ReduceReadOut("mean", field=field)  # type: ignore[assignment]
                readout_feats = input_feats + dim_state_feats_used if include_state else input_feats

            dims_final_layer = [readout_feats, units, units, ntargets]
            self.final_layer = MLP(dims_final_layer, activation, activate_last=False)
            if task_type == "classification":
                self.sigmoid = nn.Sigmoid()
        else:
            if task_type == "classification":
                raise ValueError("Classification task cannot be extensive.")
            self.final_layer = WeightedReadOut(  # type: ignore[assignment]
                in_feats=dim_node_embedding,
                dims=[units, units],
                num_targets=ntargets,
            )

        self.max_n = max_n
        self.max_l = max_l
        self.n_blocks = nblocks
        self.units = units
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.include_state = include_state
        self.task_type = task_type
        self.is_intensive = is_intensive
        self.field = field
        self.readout_type = readout_type

    def _readout(self, node_feat: torch.Tensor, edge_feat: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:
        """Dispatch the configured readout on the right field tensor."""
        x = node_feat if self.field == "node_feat" else edge_feat
        if isinstance(self.readout, ReduceReadOut):
            return self.readout(x, batch)
        return self.readout(x, batch)

    def forward(
        self,
        g: Any,
        state_attr: torch.Tensor | None = None,
        l_g: dict[str, torch.Tensor] | None = None,
        return_all_layer_output: bool = False,
    ):
        """Forward pass of M3GNet (PyG).

        Args:
            g: PyG ``Data`` (or ``Data``-like) object with attributes
                ``node_type`` (or ``z``), ``pos``, ``edge_index``, optionally
                ``pbc_offshift`` and ``batch`` / ``num_graphs``.
            state_attr: Per-graph state features (optional).
            l_g: Cached line-graph bundle from
                :func:`matgl.graph._compute.create_line_graph`. If ``None``,
                a fresh one is built from ``g``.
            return_all_layer_output: Whether to return all intermediate
                features as a dict (useful for featurization).
        """
        node_types = getattr(g, "node_type", getattr(g, "z", None))
        pos = g.pos
        edge_index = g.edge_index
        pbc_offshift = getattr(g, "pbc_offshift", None)
        batch = getattr(g, "batch", None)
        num_graphs = getattr(g, "num_graphs", None)
        num_nodes = pos.size(0)
        num_bonds = edge_index.size(1)
        if num_graphs is None:
            num_graphs = 1 if batch is None else int(batch.max().item()) + 1
        edge_batch = None if batch is None else batch[edge_index[0]].to(torch.long)

        bond_vec, bond_dist = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)
        expanded_dists = self.bond_expansion(bond_dist)

        if l_g is None:
            l_g = create_line_graph(edge_index, bond_dist, bond_vec, pbc_offshift, num_nodes, self.threebody_cutoff)
        else:
            l_g = ensure_line_graph_compatibility(l_g, bond_dist, bond_vec, pbc_offshift, self.threebody_cutoff)

        angles = compute_theta_and_phi(l_g["bond_vec"], l_g["bond_dist"], l_g["line_edge_index"])
        three_body_basis = self.basis_expansion(angles["triple_bond_lengths"], angles["cos_theta"], angles["phi"])
        three_body_cutoff = polynomial_cutoff(bond_dist, self.threebody_cutoff)

        node_feat, edge_feat, state_feat = self.embedding(node_types, expanded_dists, state_attr)
        if self.include_state and state_feat is not None and state_feat.dim() == 1:
            state_feat = state_feat.unsqueeze(0)

        fea_dict: dict[str, Any] = {
            "bond_expansion": expanded_dists,
            "three_body_basis": three_body_basis,
            "embedding": {"node_feat": node_feat, "edge_feat": edge_feat, "state_feat": state_feat},
        }

        edge_dst_atom = edge_index[1]
        line_edge_index = l_g["line_edge_index"]
        n_triple_ij = l_g["n_triple_ij"]

        for i in range(self.n_blocks):
            edge_feat = self.three_body_interactions[i](
                edge_dst_atom,
                line_edge_index,
                n_triple_ij,
                num_bonds,
                three_body_basis,
                three_body_cutoff,
                node_feat,
                edge_feat,
            )
            edge_feat, node_feat, state_feat = self.graph_layers[i](
                edge_index,
                edge_feat,
                node_feat,
                state_feat,
                expanded_dists,
                batch,
                edge_batch,
                num_nodes,
                num_graphs,
            )
            fea_dict[f"gc_{i + 1}"] = {
                "node_feat": node_feat,
                "edge_feat": edge_feat,
                "state_feat": state_feat,
            }

        if self.is_intensive:
            field_vec = self._readout(node_feat, edge_feat, batch)
            if self.include_state and state_feat is not None:
                state_view = state_feat.view(num_graphs, -1)
                readout_vec = torch.hstack([field_vec, state_view])
            else:
                readout_vec = field_vec
            fea_dict["readout"] = readout_vec
            output = self.final_layer(readout_vec)
            if self.task_type == "classification":
                output = self.sigmoid(output)
        else:
            atomic = self.final_layer(node_feat)  # (num_nodes, ntargets)
            fea_dict["readout"] = atomic
            atomic = atomic.view(-1)
            if batch is None:
                output = atomic.sum().view(1)
            else:
                output = torch.zeros(num_graphs, dtype=atomic.dtype, device=atomic.device)
                output = output.index_add(0, batch.to(torch.long), atomic)

        fea_dict["final"] = output
        if return_all_layer_output:
            return fea_dict
        return torch.squeeze(output)

    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
        output_layers: list | None = None,
        return_features: bool = False,
    ):
        """Convenience method to predict a property from a structure (PyG)."""
        import matgl

        if graph_converter is None:
            from matgl.ext._pymatgen import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)
        g, lat, state_attr_default = graph_converter.get_graph(structure)
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        g.pos = g.frac_coords @ lat[0]
        if state_feats is None:
            state_feats = torch.tensor(state_attr_default, dtype=matgl.float_th)
        if return_features:
            return self(g=g, state_attr=state_feats, return_all_layer_output=True)
        return self(g=g, state_attr=state_feats).detach()
