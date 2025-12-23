"""Implementation of the Crystal Hamiltonian Graph neural Network (CHGNet) model.

CHGNet is a graph neural network model that includes 3 body interactions through a
global line graph, and also includes charge information by including training and
prediction of site-wise magnetic moments.

The CHGNet model is described in the following paper: https://doi.org/10.1038/s42256-023-00716-3

This implementation is contributed by Bowen Deng and Luis Barroso-Luque
Date: March 2024
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch
from dgl import readout_edges, readout_nodes
from torch import nn

from matgl.config import DEFAULT_ELEMENTS
from matgl.graph._compute_dgl import (
    compute_pair_vector_and_distance,
    compute_theta,
    create_line_graph,
    ensure_line_graph_compatibility,
)
from matgl.layers._activations import ActivationFunction
from matgl.layers._basis import FourierExpansion, RadialBesselFunction
from matgl.layers._core_dgl import GatedMLPNorm, MLPNorm
from matgl.layers._graph_convolution_dgl import CHGNetAtomGraphBlock, CHGNetBondGraphBlock
from matgl.utils.cutoff import polynomial_cutoff

from ._core import MatGLModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    import dgl

    from matgl.graph._converters_dgl import GraphConverter

logger = logging.getLogger(__name__)

DEFAULT_ELEMENTS = (*list(DEFAULT_ELEMENTS[:83]), "Po", "At", "Rn", "Fr", "Ra", *list(DEFAULT_ELEMENTS[83:]))


class CHGNet(MatGLModel):
    """Main CHGNet model."""

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
        normalization: Literal["graph", "layer"] | None = None,
        normalize_hidden: bool = False,
        is_intensive: bool = False,
        num_targets: int = 1,
        num_site_targets: int = 1,
        task_type: Literal["regression", "classification"] = "regression",
        error_handling: bool = True,
        **kwargs,
    ):
        """
        Args:
            element_types (list(str)): List of element types to consider in the model.
                If None, defaults to DEFAULT_ELEMENTS.
                Default = None
            dim_atom_embedding (int): Dimension of the atom embedding.
                Default = 64
            dim_bond_embedding (int): Dimension of the bond embedding.
                Default = 64
            dim_angle_embedding (int): Dimension of the angle embedding.
                Default = 64
            dim_state_embedding (int): Dimension of the state embedding.
                Default = None
            dim_state_feats (int): Dimension of the state features.
                Default = None
            non_linear_bond_embedding (bool): Whether to use a non-linear embedding for the
                bond features (single layer MLP).
                Default = False
            non_linear_angle_embedding (bool): Whether to use a non-linear embedding for the
                angle features (single layer MLP).
                Default = False
            cutoff (float): cutoff radius of atom graph.
                Default = 6.0
            threebody_cutoff (float): cutoff radius for bonds included in bond graph for
                three body interactions.
                Default = 3.0
            cutoff_exponent (int): exponent for the smoothing cutoff function.
                Default = 5
            max_n (int): maximum number of basis functions for the bond length radial expansion.
                Default = 9
            max_f (int): maximum number of basis functions for the angle Fourier expansion.
                Default = 4
            learn_basis (bool): whether to learn the frequencies used in basis functions
                or use fixed basis functions.
                Default = True
            num_blocks (int): number of graph convolution blocks.
                Default = 4
            shared_bond_weights (str): whether to share bond weights among layers in the atom
                and bond graph blocks.
                Default = "both"
            layer_bond_weights (str): whether to use independent weights in each convolution layer.
                Default = None
            atom_conv_hidden_dims (Sequence(int)): hidden dimensions for the atom graph convolution layers.
                Default = (64, )
            bond_update_hidden_dims: (Sequence(int)) hidden dimensions for the atom to
                bond message passing layer in atom graph. None means no atom to bond
                message passing layer in atom graph.
                Default = None
            bond_conv_hidden_dims (Sequence(int)): hidden dimensions for the bond graph convolution layers.
                Default = (64, )
            angle_update_hidden_dims (Sequence(int)): hidden dimensions for the angle layer.
                Default = ()
            conv_dropout (float): dropout probability for the graph convolution layers.
                Default = 0.0
            final_mlp_type (str): type of readout block, options are "gated" for a Gated MLP and "mlp".
                Default = "mlp"
            final_hidden_dims (Sequence(int)): hidden dimensions for the readout MLP.
                Default = (64, 64)
            final_dropout (float): dropout probability for the readout MLP.
                Default = 0.0
            pooling_operation (str): type of readout pooling operation to use.
                Default = "sum"
            readout_field (str): field to readout from the graph.
                Default = "atom_feat"
            activation_type (str): activation function to use.
                Default = "swish"
            normalization (str): Normalization type to use in update functions.
                Either "graph" or "layer". If None, no normalization is applied.
                Default = None
            normalize_hidden (bool): whether to normalize the hidden layers in
                convolution update functions.
                Default = False
            is_intensive (bool): whether the target is intensive or extensive.
                Default = False
            num_targets (int): number of targets to predict.
                Default = 1
            num_site_targets (int): number of site-wise targets to predict. (ie magnetic moments)
                Default = 1
            task_type (str): type of task to perform, either "regression" pr "classification"
                Default = "regression"
            error_handling (bool): whether enables numerical tolerance in the error
                handling of line graph construction. When error handling is triggered
                in very rare cases (<0.001%), a small amount (1e-6) of perturbation is
                added to line graph cutoff to avoid inconsistent line graph.
            **kwargs: additional keyword arguments.
        """
        super().__init__()

        self.save_args(locals(), kwargs)

        if task_type == "classification":
            raise NotImplementedError("classification with CHGNet not yet implemented")

        if is_intensive:
            raise NotImplementedError("intensive targets with CHGNet not yet implemented")

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
                f"Angle Readout requires threebody_cutoff > 0, "
                f"but CHGNet is initialized with threebody_cutoff={threebody_cutoff}"
            )

        # basis expansions for bond lengths, triple interaction bond lengths and angles
        self.bond_expansion = RadialBesselFunction(max_n=max_n, cutoff=cutoff, learnable=learn_basis)
        self.threebody_bond_expansion = (
            RadialBesselFunction(max_n=max_n, cutoff=threebody_cutoff, learnable=learn_basis)
            if self.use_bond_graph
            else None
        )
        self.angle_expansion = FourierExpansion(max_f=max_f, learnable=learn_basis) if self.use_bond_graph else None

        # embedding block for atom, bond, angle, and optional state features
        self.include_states = dim_state_feats is not None
        self.state_embedding = nn.Embedding(dim_state_feats, dim_state_embedding) if self.include_states else None  # type: ignore[arg-type]
        self.atom_embedding = nn.Embedding(len(element_types), dim_atom_embedding)
        self.bond_embedding = MLPNorm(
            [max_n, dim_bond_embedding], activation=activation, activate_last=non_linear_bond_embedding, bias_last=False
        )
        self.angle_embedding = (
            MLPNorm(
                [2 * max_f + 1, dim_angle_embedding],
                activation=activation,
                activate_last=non_linear_angle_embedding,
                bias_last=False,
            )
            if self.use_bond_graph
            else None
        )

        # shared message bond distance smoothing weights
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

        # operations involving the graph (i.e. atom graph) to update atom and bond features
        self.atom_graph_layers = nn.ModuleList(
            [
                CHGNetAtomGraphBlock(
                    num_atom_feats=dim_atom_embedding,
                    num_bond_feats=dim_bond_embedding,
                    atom_hidden_dims=atom_conv_hidden_dims,
                    bond_hidden_dims=bond_update_hidden_dims,
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

        # operations involving the line graph (i.e. bond graph) to update bond and angle features
        self.bond_graph_layers = (
            nn.ModuleList(
                [
                    CHGNetBondGraphBlock(
                        num_atom_feats=dim_atom_embedding,
                        num_bond_feats=dim_bond_embedding,
                        num_angle_feats=dim_angle_embedding,
                        bond_hidden_dims=bond_conv_hidden_dims,
                        angle_hidden_dims=angle_update_hidden_dims,
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

        self.sitewise_readout = (
            nn.Linear(dim_atom_embedding, num_site_targets) if num_site_targets > 0 else lambda x: None
        )

        input_dim = dim_atom_embedding if readout_field == "atom_feat" else dim_bond_embedding
        if final_mlp_type == "mlp":
            self.final_layer = MLPNorm(
                dims=[input_dim, *final_hidden_dims, num_targets], activation=activation, activate_last=False
            )
        elif final_mlp_type == "gated":
            self.final_layer = GatedMLPNorm(  # type: ignore[assignment]
                in_feats=input_dim, dims=[*final_hidden_dims, num_targets], activate_last=False
            )
        else:
            raise ValueError(f"Invalid final MLP type: {final_mlp_type}")

        self.element_types = element_types
        self.max_n = max_n
        self.max_f = max_f
        self.cutoff = cutoff
        self.cutoff_exponent = cutoff_exponent
        self.three_body_cutoff = threebody_cutoff

        self.n_blocks = num_blocks
        self.readout_operation = pooling_operation
        self.readout_field = readout_field
        self.readout_type = final_mlp_type

        self.task_type = task_type
        self.is_intensive = is_intensive

    def forward(
        self,
        g: dgl.DGLGraph,
        state_attr: torch.Tensor | None = None,
        l_g: dgl.DGLGraph | None = None,
        error_handling: bool = True,
    ):
        """Forward pass of the model.

        Args:
            g (dgl.DGLGraph): Input g.
            state_attr (torch.Tensor, optional): State features. Defaults to None.
            l_g (dgl.DGLGraph, optional): Line graph. Defaults to None and is computed internally.
            error_handling (bool, optional): Whether to allow numerical tolerance when an error occurs in
                l_g construction. Defaults to True.

        Returns:
            torch.Tensor: Model output.
        """
        # compute bond vectors and distances and add to g, needs to be computed here to register gradients
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["bond_vec"] = bond_vec.to(g.device)
        g.edata["bond_dist"] = bond_dist.to(g.device)
        bond_expansion = self.bond_expansion(bond_dist)
        smooth_cutoff = polynomial_cutoff(bond_expansion, self.cutoff, self.cutoff_exponent)
        g.edata["bond_expansion"] = smooth_cutoff * bond_expansion

        # compute state, atom, bond and angle embeddings
        atom_features = self.atom_embedding(g.ndata["node_type"])
        bond_features = self.bond_embedding(g.edata["bond_expansion"])
        if self.state_embedding is not None and state_attr is not None:
            state_attr = self.state_embedding(state_attr)
        else:
            state_attr = None

        # create bond graph (line graph) with necessary node and edge data
        if self.use_bond_graph:
            if l_g is None:
                bond_graph = create_line_graph(g, self.three_body_cutoff, directed=True, error_handling=error_handling)
            else:
                # need to ensure the line graph matches the graph
                bond_graph = ensure_line_graph_compatibility(g, l_g, self.three_body_cutoff, directed=True)

            bond_graph.ndata["bond_index"] = bond_graph.ndata["edge_ids"]
            threebody_bond_expansion = self.threebody_bond_expansion(bond_graph.ndata["bond_dist"])  # type: ignore[misc]
            smooth_cutoff = polynomial_cutoff(threebody_bond_expansion, self.three_body_cutoff, self.cutoff_exponent)
            bond_graph.ndata["bond_expansion"] = smooth_cutoff * threebody_bond_expansion
            # the center atom is the dst atom of the src bond or the reverse (the src atom of the dst bond)
            # need to use "bond_index" just to be safe always
            bond_indices = bond_graph.ndata["bond_index"][bond_graph.edges()[0]]
            bond_graph.edata["center_atom_index"] = g.edges()[1][bond_indices]
            bond_graph.apply_edges(compute_theta)
            bond_graph.edata["angle_expansion"] = self.angle_expansion(bond_graph.edata["theta"])  # type: ignore[misc]
            angle_features = self.angle_embedding(bond_graph.edata["angle_expansion"])  # type: ignore[misc]
        else:
            bond_graph = None
            angle_features = None

        # shared message weights
        atom_bond_weights = (
            self.atom_bond_weights(g.edata["bond_expansion"]) if self.atom_bond_weights is not None else None
        )
        bond_bond_weights = (
            self.bond_bond_weights(g.edata["bond_expansion"]) if self.bond_bond_weights is not None else None
        )
        threebody_bond_weights = (
            self.threebody_bond_weights(bond_graph.ndata["bond_expansion"])
            if self.threebody_bond_weights is not None
            else None
        )

        # message passing layers
        for i in range(self.n_blocks - 1):
            atom_features, bond_features, state_attr = self.atom_graph_layers[i](
                g, atom_features, bond_features, state_attr, atom_bond_weights, bond_bond_weights
            )
            if self.use_bond_graph:
                bond_features, angle_features = self.bond_graph_layers[i](  # type: ignore
                    bond_graph, atom_features, bond_features, angle_features, threebody_bond_weights
                )

        # site wise target readout
        g.ndata["magmom"] = self.sitewise_readout(atom_features)

        # last atom graph message passing layer
        atom_features, bond_features, state_attr = self.atom_graph_layers[-1](
            g, atom_features, bond_features, state_attr, atom_bond_weights, bond_bond_weights
        )

        # readout
        if self.readout_field == "atom_feat":
            g.ndata["atom_feat"] = self.final_layer(atom_features)
            structure_properties = readout_nodes(g, "atom_feat", op=self.readout_operation)
        elif self.readout_field == "bond_feat":
            g.edata["bond_feat"] = self.final_layer(bond_features)
            structure_properties = readout_edges(g, "bond_feat", op=self.readout_operation)
        else:  # self.readout_field == "angle_feat":
            bond_graph.edata["angle_feat"] = self.final_layer(angle_features)
            structure_properties = readout_edges(bond_graph, "angle_feat", op=self.readout_operation)

        structure_properties = torch.squeeze(structure_properties)
        return structure_properties

    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
        error_handling: bool = True,
    ):
        """Convenience method to directly predict property from structure.

        Args:
            structure: An input crystal/molecule.
            state_feats (torch.tensor): Graph attributes
            graph_converter: Object that implements a get_graph_from_structure.
            error_handling (bool, optional): Whether to allow numerical tolerance when an error occurs in
                l_g construction. Defaults to True.

        Returns:
            output (torch.tensor): output property
        """
        if graph_converter is None:
            from matgl.ext._pymatgen_dgl import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)

        graph, lattice, state_feats_default = graph_converter.get_graph(structure)
        graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lattice[0])
        graph.ndata["pos"] = graph.ndata["frac_coords"] @ lattice[0]
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)
        return self(g=graph, state_attr=state_feats, error_handling=error_handling)
