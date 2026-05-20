"""Implementation of the Crystal Hamiltonian Graph neural Network (CHGNet) model.

CHGNet is a graph neural network model that includes 3-body interactions through a
directed line graph, and also includes charge information by including training and
prediction of site-wise magnetic moments.

Reference paper: https://doi.org/10.1038/s42256-023-00716-3

Line-graph construction is wrapped in ``torch.no_grad()`` so that three-body bond
vectors and distances are detached from the position gradient graph.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn

from matgl.config import DEFAULT_ELEMENTS
from matgl.graph._compute import (
    compute_pair_vector_and_distance,
    compute_theta,
    create_directed_line_graph,
)
from matgl.layers._activations import ActivationFunction
from matgl.layers._basis import FourierExpansion, RadialBesselFunction
from matgl.layers._graph_convolution import (
    CHGNetAtomGraphBlock,
    CHGNetBondGraphBlock,
    _GatedMLPNorm,
)
from matgl.layers._graph_convolution import (
    _MLPNorm as _EmbedMLP,
)
from matgl.utils.cutoff import polynomial_cutoff
from matgl.utils.maths import scatter_add

from ._core import MatGLModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch_geometric.data import Data

    from matgl.graph._converters import GraphConverter

logger = logging.getLogger(__name__)

# Extend default elements with Po, At, Rn, Fr, Ra.
DEFAULT_ELEMENTS = (*list(DEFAULT_ELEMENTS[:83]), "Po", "At", "Rn", "Fr", "Ra", *list(DEFAULT_ELEMENTS[83:]))


class CHGNet(MatGLModel):
    """CHGNet model.

    Uses PyTorch Geometric ``Data`` objects with center→neighbor edge direction
    and destination-node aggregation. Line-graph construction is wrapped in
    ``torch.no_grad()`` so three-body bond data is detached from the position gradient.
    """

    __version__ = 1

    def __init__(
        self,
        element_types: tuple[str, ...] | None = None,
        dim_atom_embedding: int = 64,
        dim_bond_embedding: int = 64,
        dim_angle_embedding: int = 64,
        dim_state_embedding: int | None = None,
        dim_state_feats: int | None = None,
        non_linear_bond_embedding: bool = False,
        non_linear_angle_embedding: bool = False,
        cutoff: float = 6.0,
        threebody_cutoff: float = 3.0,
        cutoff_exponent: int = 5,
        max_n: int = 9,
        max_f: int = 4,
        learn_basis: bool = True,
        num_blocks: int = 4,
        shared_bond_weights: Literal["bond", "three_body_bond", "both"] | None = "both",
        layer_bond_weights: Literal["bond", "three_body_bond", "both"] | None = None,
        atom_conv_hidden_dims: Sequence[int] = (64,),
        bond_update_hidden_dims: Sequence[int] | None = None,
        bond_conv_hidden_dims: Sequence[int] = (64,),
        angle_update_hidden_dims: Sequence[int] | None = (),
        conv_dropout: float = 0.0,
        final_mlp_type: Literal["gated", "mlp"] = "mlp",
        final_hidden_dims: Sequence[int] = (64, 64),
        final_dropout: float = 0.0,
        pooling_operation: Literal["sum", "mean"] = "sum",
        readout_field: Literal["atom_feat", "bond_feat", "angle_feat"] = "atom_feat",
        activation_type: str = "swish",
        normalization: Literal["layer"] | None = None,
        normalize_hidden: bool = False,
        is_intensive: bool = False,
        num_targets: int = 1,
        num_site_targets: int = 1,
        task_type: Literal["regression", "classification"] = "regression",
        **kwargs,
    ) -> None:
        """Initialize CHGNet PyG model.

        Args:
            element_types: List of element types. Defaults to DEFAULT_ELEMENTS.
            dim_atom_embedding: Atom embedding dimension. Default = 64
            dim_bond_embedding: Bond embedding dimension. Default = 64
            dim_angle_embedding: Angle embedding dimension. Default = 64
            dim_state_embedding: State embedding dimension. Default = None
            dim_state_feats: Number of state features. Default = None
            non_linear_bond_embedding: Non-linear bond embedding. Default = False
            non_linear_angle_embedding: Non-linear angle embedding. Default = False
            cutoff: Atom graph cutoff radius. Default = 6.0
            threebody_cutoff: Three-body cutoff radius. Default = 3.0
            cutoff_exponent: Polynomial cutoff exponent. Default = 5
            max_n: Radial basis functions count. Default = 9
            max_f: Fourier expansion terms. Default = 4
            learn_basis: Learnable basis frequencies. Default = True
            num_blocks: Number of graph convolution blocks. Default = 4
            shared_bond_weights: Shared bond distance weights. Default = "both"
            layer_bond_weights: Per-layer bond weights. Default = None
            atom_conv_hidden_dims: Atom conv hidden dims. Default = (64,)
            bond_update_hidden_dims: Bond update hidden dims. Default = None
            bond_conv_hidden_dims: Bond conv hidden dims. Default = (64,)
            angle_update_hidden_dims: Angle update hidden dims. Default = ()
            conv_dropout: Convolution dropout. Default = 0.0
            final_mlp_type: Readout MLP type ("gated" or "mlp"). Default = "mlp"
            final_hidden_dims: Readout MLP hidden dims. Default = (64, 64)
            final_dropout: Readout dropout. Default = 0.0
            pooling_operation: Graph pooling ("sum" or "mean"). Default = "sum"
            readout_field: Feature field to read out from. Default = "atom_feat"
            activation_type: Activation function name. Default = "swish"
            normalization: Normalization type. Only "layer" supported. Default = None
            normalize_hidden: Normalize hidden layers. Default = False
            is_intensive: Intensive target. Default = False
            num_targets: Number of targets. Default = 1
            num_site_targets: Number of site-wise targets. Default = 1
            task_type: "regression" or "classification". Default = "regression"
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.save_args(locals(), kwargs)

        if task_type == "classification":
            raise NotImplementedError("Classification with CHGNet is not yet implemented.")
        if is_intensive:
            raise NotImplementedError("Intensive targets with CHGNet are not yet implemented.")
        if normalization == "graph":
            raise ValueError("GraphNorm is not supported in the PyG CHGNet backend. Use normalization='layer' or None.")

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        element_types = element_types or DEFAULT_ELEMENTS
        self.use_bond_graph = threebody_cutoff > 0
        if not self.use_bond_graph and readout_field == "angle_feat":
            raise ValueError(
                f"Angle readout requires threebody_cutoff > 0, but got threebody_cutoff={threebody_cutoff}"
            )

        # --- basis expansions ---
        self.bond_expansion = RadialBesselFunction(max_n=max_n, cutoff=cutoff, learnable=learn_basis)
        self.threebody_bond_expansion = (
            RadialBesselFunction(max_n=max_n, cutoff=threebody_cutoff, learnable=learn_basis)
            if self.use_bond_graph
            else None
        )
        self.angle_expansion = FourierExpansion(max_f=max_f, learnable=learn_basis) if self.use_bond_graph else None

        # --- embeddings ---
        self.include_states = dim_state_feats is not None
        self.state_embedding = nn.Embedding(dim_state_feats, dim_state_embedding) if self.include_states else None  # type: ignore[arg-type]
        self.atom_embedding = nn.Embedding(len(element_types), dim_atom_embedding)

        self.bond_embedding = _EmbedMLP(
            [max_n, dim_bond_embedding],
            activation=activation,
            activate_last=non_linear_bond_embedding,
            bias_last=False,
        )
        self.angle_embedding = (
            _EmbedMLP(
                [2 * max_f + 1, dim_angle_embedding],
                activation=activation,
                activate_last=non_linear_angle_embedding,
                bias_last=False,
            )
            if self.use_bond_graph
            else None
        )

        # --- shared bond distance weights ---
        self.atom_bond_weights = (
            nn.Linear(max_n, dim_atom_embedding, bias=False) if shared_bond_weights in ["bond", "both"] else None
        )
        self.bond_bond_weights = (
            nn.Linear(max_n, dim_bond_embedding, bias=False) if shared_bond_weights in ["bond", "both"] else None
        )
        self.threebody_bond_weights = (
            nn.Linear(max_n, dim_bond_embedding, bias=False)
            if shared_bond_weights in ["three_body_bond", "both"] and self.use_bond_graph
            else None
        )

        # --- atom graph blocks ---
        self.atom_graph_layers = nn.ModuleList(
            [
                CHGNetAtomGraphBlock(
                    num_atom_feats=dim_atom_embedding,
                    num_bond_feats=dim_bond_embedding,
                    atom_hidden_dims=list(atom_conv_hidden_dims),
                    bond_hidden_dims=list(bond_update_hidden_dims) if bond_update_hidden_dims is not None else None,
                    num_state_feats=dim_state_embedding,
                    activation=activation,
                    normalization=normalization,
                    normalize_hidden=normalize_hidden,
                    dropout=conv_dropout,
                    rbf_order=max_n if layer_bond_weights in ["bond", "both"] else 0,
                )
                for _ in range(num_blocks)
            ]
        )

        # --- bond graph (line graph) blocks ---
        self.bond_graph_layers = (
            nn.ModuleList(
                [
                    CHGNetBondGraphBlock(
                        num_atom_feats=dim_atom_embedding,
                        num_bond_feats=dim_bond_embedding,
                        num_angle_feats=dim_angle_embedding,
                        bond_hidden_dims=list(bond_conv_hidden_dims),
                        angle_hidden_dims=list(angle_update_hidden_dims)
                        if angle_update_hidden_dims is not None
                        else None,
                        activation=activation,
                        normalization=normalization,
                        normalize_hidden=normalize_hidden,
                        bond_dropout=conv_dropout,
                        angle_dropout=conv_dropout,
                        rbf_order=max_n if layer_bond_weights in ["three_body_bond", "both"] else 0,
                    )
                    for _ in range(num_blocks - 1)
                ]
            )
            if self.use_bond_graph
            else None
        )

        # --- site-wise readout (e.g. magnetic moments) ---
        self.sitewise_readout = (
            nn.Linear(dim_atom_embedding, num_site_targets) if num_site_targets > 0 else lambda x: None
        )

        # --- final readout MLP ---
        # _EmbedMLP (_MLPNorm) stores only Linear layers at consecutive indices in self.layers.
        input_dim = dim_atom_embedding if readout_field == "atom_feat" else dim_bond_embedding
        _act = activation if activation is not None else nn.SiLU()
        if final_mlp_type == "mlp":
            self.final_layer = _EmbedMLP(
                dims=[input_dim, *final_hidden_dims, num_targets], activation=_act, activate_last=False
            )
        elif final_mlp_type == "gated":
            self.final_layer = _GatedMLPNorm(  # type: ignore[assignment]
                in_feats=input_dim, dims=[*final_hidden_dims, num_targets], activation=_act, activate_last=False
            )
        else:
            raise ValueError(f"Invalid final_mlp_type: {final_mlp_type}")

        self.final_dropout = nn.Dropout(final_dropout) if final_dropout > 0.0 else nn.Identity()

        # --- store hyperparameters ---
        self.element_types = element_types
        self.max_n = max_n
        self.max_f = max_f
        self.cutoff = cutoff
        self.cutoff_exponent = cutoff_exponent
        self.three_body_cutoff = threebody_cutoff
        self.n_blocks = num_blocks
        self.readout_operation = pooling_operation
        self.readout_field = readout_field
        self.task_type = task_type
        self.is_intensive = is_intensive

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        g: Data,
        state_attr: torch.Tensor | None = None,
        l_g: Any | None = None,
        return_all_layer_output: bool = False,
    ) -> torch.Tensor | dict:
        """Forward pass.

        Args:
            g: PyG ``Data`` graph. Must have ``pos``, ``edge_index``,
                ``pbc_offshift``, ``node_type``, and optionally ``batch``.
            state_attr: Global state attributes. Default = None.
            l_g: Pre-built line graph (ignored — line graph is always rebuilt from
                current positions to maintain gradient flow). Kept for API compatibility
                with ``_pes.Potential``.
            return_all_layer_output: Return intermediate features. Default = False.

        Returns:
            Structure-level property tensor, or feature dict when
            ``return_all_layer_output=True``.
        """
        edge_index = g.edge_index
        pos = g.pos
        pbc_offshift = getattr(g, "pbc_offshift", None)
        batch = getattr(g, "batch", None)

        # --- bond vectors & distances (needs grad through pos for forces) ---
        bond_vec, bond_dist = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)
        bond_expansion = self.bond_expansion(bond_dist)
        smooth_cutoff = polynomial_cutoff(bond_expansion, self.cutoff, self.cutoff_exponent)
        bond_expansion = smooth_cutoff * bond_expansion

        # --- embeddings ---
        atom_features = self.atom_embedding(g.node_type)
        bond_features = self.bond_embedding(bond_expansion)
        if self.state_embedding is not None and state_attr is not None:
            state_attr = self.state_embedding(state_attr)
        else:
            state_attr = None

        fea_dict: dict = {
            "bond_expansion": bond_expansion,
            "embedding": {"atom_feat": atom_features, "bond_feat": bond_features, "state_feat": state_attr},
        }

        # --- directed line graph (bond graph) ---
        if self.use_bond_graph:
            pbc_offset = getattr(g, "pbc_offset", torch.zeros(edge_index.size(1), 3, device=pos.device))
            with torch.no_grad():
                (lg_edge_index, lg_bond_vec, lg_bond_dist, _lg_pbc_offset, lg_src_bond_sign) = (
                    create_directed_line_graph(edge_index, pbc_offset, bond_vec, bond_dist, self.three_body_cutoff)
                )

            num_lg_nodes = lg_bond_dist.size(0)

            # Radial expansion for three-body bonds
            tb_expansion = self.threebody_bond_expansion(lg_bond_dist)  # type: ignore[misc]
            smooth_tb = polynomial_cutoff(tb_expansion, self.three_body_cutoff, self.cutoff_exponent)
            lg_bond_expansion = smooth_tb * tb_expansion

            # Map line-graph nodes back to atom-graph edge ids
            # lg nodes correspond to the first num_lg_nodes bonds within three-body cutoff
            valid_mask = bond_dist <= self.three_body_cutoff
            bond_index = valid_mask.nonzero(as_tuple=False).squeeze(1)[:num_lg_nodes]

            # Center atom = dst atom of the src bond in each lg edge
            # (src bond of each lg edge, using bond_index to get the atom-graph edge id)
            if lg_edge_index.size(1) > 0:
                src_bond_ids = bond_index[lg_edge_index[0].long()]
                # dst in atom graph = neighbor = center atom for the angle
                center_atom_index = edge_index[1][src_bond_ids]
            else:
                center_atom_index = torch.zeros(0, dtype=torch.long, device=pos.device)

            # Compute angles
            lg_src_idx = lg_edge_index[0].long()
            lg_dst_idx = lg_edge_index[1].long()
            if lg_edge_index.size(1) > 0:
                cos_theta = compute_theta(lg_bond_vec, lg_src_bond_sign, lg_src_idx, lg_dst_idx, directed=True)
                cos_theta = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
                theta = torch.acos(cos_theta)
            else:
                theta = torch.zeros(0, device=pos.device)

            angle_expansion = self.angle_expansion(theta)  # type: ignore[misc]
            angle_features = self.angle_embedding(angle_expansion)  # type: ignore[misc]

            fea_dict["angle_expansion"] = angle_expansion
            fea_dict["embedding"]["angle_feat"] = angle_features
        else:
            lg_edge_index = None
            lg_bond_expansion = None
            bond_index = None
            center_atom_index = None
            angle_features = None

        # --- shared weights ---
        atom_bond_weights = self.atom_bond_weights(bond_expansion) if self.atom_bond_weights is not None else None
        bond_bond_weights = self.bond_bond_weights(bond_expansion) if self.bond_bond_weights is not None else None
        threebody_bond_weights = (
            self.threebody_bond_weights(lg_bond_expansion)
            if self.threebody_bond_weights is not None and lg_bond_expansion is not None
            else None
        )

        # --- message passing ---
        for i in range(self.n_blocks - 1):
            atom_features, bond_features, state_attr = self.atom_graph_layers[i](
                edge_index,
                atom_features,
                bond_features,
                bond_expansion,
                state_attr,
                batch,
                atom_bond_weights,
                bond_bond_weights,
            )
            if self.use_bond_graph:
                bond_features, angle_features = self.bond_graph_layers[i](  # type: ignore[index]
                    lg_edge_index,
                    bond_features,
                    angle_features,
                    atom_features,
                    bond_index,
                    center_atom_index,
                    lg_bond_expansion,
                    threebody_bond_weights,
                )
            fea_dict[f"gc_{i + 1}"] = {
                "atom_feat": atom_features,
                "bond_feat": bond_features,
                "angle_feat": angle_features,
                "state_feat": state_attr,
            }

        # --- site-wise readout before last atom block ---
        magmom = self.sitewise_readout(atom_features)
        fea_dict["magmom"] = magmom
        if hasattr(g, "magmom"):
            g.magmom = magmom
        else:
            g.magmom = magmom

        # --- last atom graph block ---
        atom_features, bond_features, state_attr = self.atom_graph_layers[-1](
            edge_index,
            atom_features,
            bond_features,
            bond_expansion,
            state_attr,
            batch,
            atom_bond_weights,
            bond_bond_weights,
        )
        fea_dict[f"gc_{self.n_blocks}"] = {
            "atom_feat": atom_features,
            "bond_feat": bond_features,
            "angle_feat": angle_features,
            "state_feat": state_attr,
        }

        # --- graph-level readout ---
        num_graphs = (int(batch.max().item()) + 1) if batch is not None else 1

        if self.readout_field == "atom_feat":
            per_node = self.final_dropout(self.final_layer(atom_features))
            structure_properties = self._pool(per_node, batch, num_graphs)
        elif self.readout_field == "bond_feat":
            per_edge = self.final_dropout(self.final_layer(bond_features))
            # pool over edges: use src atom batch assignment
            edge_batch = batch[edge_index[0]] if batch is not None else None
            structure_properties = self._pool(per_edge, edge_batch, num_graphs)
        else:  # angle_feat
            per_angle = self.final_dropout(self.final_layer(angle_features))
            lg_src_batch = batch[edge_index[0][bond_index[lg_src_idx]]] if batch is not None else None  # type: ignore[index]
            structure_properties = self._pool(per_angle, lg_src_batch, num_graphs)

        structure_properties = torch.squeeze(structure_properties)
        fea_dict["final"] = structure_properties

        if return_all_layer_output:
            return fea_dict
        return structure_properties

    def _pool(
        self,
        feat: torch.Tensor,
        batch: torch.Tensor | None,
        num_graphs: int,
    ) -> torch.Tensor:
        """Pool node/edge features to graph level."""
        if batch is None:
            return feat.sum(dim=0, keepdim=True) if self.readout_operation == "sum" else feat.mean(dim=0, keepdim=True)
        out = scatter_add(feat, batch.long(), dim=0, dim_size=num_graphs)
        if self.readout_operation == "mean":
            counts = torch.bincount(batch.long(), minlength=num_graphs).float().clamp(min=1)
            out = out / counts.unsqueeze(1)
        return out

    # ------------------------------------------------------------------
    # Convenience method
    # ------------------------------------------------------------------
    def predict_structure(  # type: ignore[override]
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
        return_features: bool = False,
        output_layers: list | None = None,
    ) -> torch.Tensor | dict:
        """Predict property directly from a Pymatgen Structure.

        Args:
            structure: Pymatgen Structure or Molecule.
            state_feats: Optional state attributes.
            graph_converter: Graph converter. Defaults to ``Structure2Graph``.
            return_features: Return intermediate features. Default = False.
            output_layers: Layer names to return (used when return_features=True).

        Returns:
            Predicted property tensor, or feature dict.
        """
        allowed_output_layers = ["bond_expansion", "angle_expansion", "embedding", "magmom", "final"] + [
            f"gc_{i + 1}" for i in range(self.n_blocks)
        ]

        if not return_features:
            output_layers = ["final"]
        elif output_layers is None:
            output_layers = allowed_output_layers
        elif not isinstance(output_layers, list) or set(output_layers).difference(allowed_output_layers):
            raise ValueError(f"Invalid output_layers. Must be a sublist of {allowed_output_layers}.")

        if graph_converter is None:
            from matgl.ext._pymatgen import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)  # type: ignore[arg-type]

        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        g.pos = g.frac_coords @ lat[0]
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)

        if return_features:
            model_output = self(g=g, state_attr=state_feats, return_all_layer_output=True)
            return {k: v for k, v in model_output.items() if k in output_layers}

        return self(g=g, state_attr=state_feats).detach()
