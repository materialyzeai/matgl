"""Implementation of Materials 3-body Graph Network (M3GNet) model.

The main improvement over MEGNet is the addition of many-body interactios terms, which improves efficiency of
representation of local interactions for applications such as interatomic potentials. For more details on M3GNet,
please refer to::

    Chen, C., Ong, S.P. _A universal graph deep learning interatomic potential for the periodic table._ Nature
    Computational Science, 2023, 2, 718-728. DOI: 10.1038/s43588-022-00349-3.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import dgl
import torch
from torch import nn

from matgl.config import DEFAULT_ELEMENTS
from matgl.graph._compute_dgl import (
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
    ReduceReadOut,
    Set2SetReadOut,
    SphericalBesselWithHarmonics,
    ThreeBodyInteractions,
    WeightedAtomReadOut,
    WeightedReadOut,
)
from matgl.utils.cutoff import polynomial_cutoff

from ._core import MatGLModel

if TYPE_CHECKING:
    from matgl.graph._converters_dgl import GraphConverter

logger = logging.getLogger(__file__)


class M3GNet(MatGLModel):
    """The main M3GNet model."""

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
        """Initialize the M3GNet model.

        Args:
            element_types (tuple): List of elements appearing in the dataset. Default to DEFAULT_ELEMENTS.
            dim_node_embedding (int): Number of embedded atomic features
            dim_edge_embedding (int): Number of edge features
            dim_state_embedding (int): Number of hidden neurons in state embedding
            dim_state_feats (int): Number of state features after linear layer
            ntypes_state (int): Number of state labels
            max_n (int): Number of radial basis expansion
            max_l (int): Number of angular expansion
            nblocks (int): Number of convolution blocks
            rbf_type (str): Radial basis function. choose from 'Gaussian' or 'SphericalBessel'
            is_intensive (bool): Whether the prediction is intensive
            readout_type (str): Readout function type, `set2set`, `weighted_atom` (default) or `reduce_atom`.
            task_type (str): `classification` or `regression` (default).
            cutoff (float): Cutoff radius of the graph
            threebody_cutoff (float): Cutoff radius for 3 body interaction
            units (int): Number of neurons in each MLP layer
            ntargets (int): Number of target properties
            use_smooth (bool): Whether using smooth Bessel functions
            use_phi (bool): Whether using phi angle
            field (str): Using either "node_feat" or "edge_feat" for Set2Set and Reduced readout
            niters_set2set (int): Number of set2set iterations
            nlayers_set2set (int): Number of set2set layers
            include_state (bool): Whether to include states features
            activation_type (str): Activation type. choose from 'swish', 'tanh', 'sigmoid', 'softplus2', 'softexp'
            dropout (float): Dropout probability to apply in graph layers during training
            **kwargs: For future flexibility. Not used at the moment.
        """
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
            {
                ThreeBodyInteractions(
                    update_network_atom=MLP(
                        dims=[dim_node_embedding, degree],
                        activation=nn.Sigmoid(),
                        activate_last=True,
                    ),
                    update_network_bond=GatedMLP(in_feats=degree, dims=[dim_edge_embedding], use_bias=False),
                )
                for _ in range(nblocks)
            }
        )

        dim_state_feats = dim_state_embedding

        self.graph_layers = nn.ModuleList(
            {
                M3GNetBlock(
                    degree=degree_rbf,
                    activation=activation,
                    conv_hiddens=[units, units],
                    dim_node_feats=dim_node_embedding,
                    dim_edge_feats=dim_edge_embedding,
                    dim_state_feats=dim_state_feats,
                    include_state=include_state,
                    dropout=dropout,
                )
                for _ in range(nblocks)
            }
        )
        if is_intensive:
            input_feats = dim_node_embedding if field == "node_feat" else dim_edge_embedding
            if readout_type == "set2set":
                self.readout = Set2SetReadOut(
                    in_feats=input_feats,
                    n_iters=niters_set2set,
                    n_layers=nlayers_set2set,
                    field=field,
                )
                readout_feats = 2 * input_feats + dim_state_feats if include_state else 2 * input_feats  # type: ignore
            elif readout_type == "weighted_atom":
                self.readout = WeightedAtomReadOut(in_feats=input_feats, dims=[units, units], activation=activation)  # type: ignore[assignment]
                readout_feats = units + dim_state_feats if include_state else units  # type: ignore
            else:
                self.readout = ReduceReadOut("mean", field=field)  # type: ignore
                readout_feats = input_feats + dim_state_feats if include_state else input_feats  # type: ignore

            dims_final_layer = [readout_feats, units, units, ntargets]
            self.final_layer = MLP(dims_final_layer, activation, activate_last=False)
            if task_type == "classification":
                self.sigmoid = nn.Sigmoid()

        else:
            if task_type == "classification":
                raise ValueError("Classification task cannot be extensive.")
            self.final_layer = WeightedReadOut(
                in_feats=dim_node_embedding,
                dims=[units, units],
                num_targets=ntargets,  # type: ignore
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

    def forward(
        self,
        g: dgl.DGLGraph,
        state_attr: torch.Tensor | None = None,
        l_g: dgl.DGLGraph | None = None,
        return_all_layer_output: bool = False,
    ):
        """Performs message passing and updates node representations.

        Args:
            g : DGLGraph for a batch of graphs.
            state_attr: State attrs for a batch of graphs.
            l_g : DGLGraph for a batch of line graphs.
            return_all_layer_output: Whether to return outputs of all M3GNet layers. By default, only the final layer
                output is returned.

        Returns:
            output: Output property for a batch of graphs.
        """
        node_types = g.ndata["node_type"]
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.edata["bond_vec"] = bond_vec
        g.edata["bond_dist"] = bond_dist

        expanded_dists = self.bond_expansion(g.edata["bond_dist"])
        if l_g is None:
            l_g = create_line_graph(g, self.threebody_cutoff)
        else:
            l_g = ensure_line_graph_compatibility(g, l_g, self.threebody_cutoff)
        l_g.apply_edges(compute_theta_and_phi)
        g.edata["rbf"] = expanded_dists
        three_body_basis = self.basis_expansion(l_g)
        three_body_cutoff = polynomial_cutoff(g.edata["bond_dist"], self.threebody_cutoff)
        node_feat, edge_feat, state_feat = self.embedding(node_types, g.edata["rbf"], state_attr)
        fea_dict = {
            "bond_expansion": expanded_dists,
            "three_body_basis": three_body_basis,
            "embedding": {
                "node_feat": node_feat,
                "edge_feat": edge_feat,
                "state_feat": state_feat,
            },
        }
        edge_dst_atom = g.edges()[1]
        line_edge_index = torch.stack(list(l_g.edges()), dim=0)
        n_triple_ij = l_g.ndata["n_triple_ij"]
        num_bonds = g.num_edges()
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
            edge_feat, node_feat, state_feat = self.graph_layers[i](g, edge_feat, node_feat, state_feat)
            fea_dict[f"gc_{i + 1}"] = {
                "node_feat": node_feat,
                "edge_feat": edge_feat,
                "state_feat": state_feat,
            }
        g.ndata["node_feat"] = node_feat
        g.edata["edge_feat"] = edge_feat
        if self.is_intensive:
            field_vec = self.readout(g)
            readout_vec = torch.hstack([field_vec, state_feat]) if self.include_state else field_vec  # type: ignore
            fea_dict["readout"] = readout_vec
            output = self.final_layer(readout_vec)
            if self.task_type == "classification":
                output = self.sigmoid(output)
        else:
            g.ndata["atomic_properties"] = self.final_layer(g)
            fea_dict["readout"] = g.ndata["atomic_properties"]
            output = dgl.readout_nodes(g, "atomic_properties", op="sum")
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
        """Convenience method to featurize or predict properties of a structure with M3GNet model.

        Args:
            structure: An input crystal/molecule.
            state_feats (torch.tensor): Graph attributes.
            graph_converter: Object that implements a get_graph_from_structure.
            output_layers: List of names for the layer of GNN as output. Choose from "bond_expansion", "embedding",
                "three_body_basis", "gc_1", "gc_2", "gc_3", "readout", and "final". By default, all M3GNet layer
                outputs are returned. Ignored if `return_features` is False.
            return_features (bool): If True, return specified layer outputs. If False, only return final output.

        Returns:
            output (dict or torch.tensor): M3GNet intermediate and final layer outputs for a structure, or final
                predicted property if `return_features` is False.
        """
        allowed_output_layers = [
            "bond_expansion",
            "embedding",
            "three_body_basis",
            "readout",
            "final",
        ] + [f"gc_{i + 1}" for i in range(self.n_blocks)]

        if not return_features:
            output_layers = ["final"]
        elif output_layers is None:
            output_layers = allowed_output_layers
        elif not isinstance(output_layers, list) or set(output_layers).difference(allowed_output_layers):
            raise ValueError(f"Invalid output_layers, it must be a sublist of {allowed_output_layers}.")

        if graph_converter is None:
            from matgl.ext._pymatgen_dgl import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)  # type: ignore

        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lat[0])
        g.ndata["pos"] = g.ndata["frac_coords"] @ lat[0]

        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)

        model_output = self(g=g, state_attr=state_feats, return_all_layer_output=True)

        if not return_features:
            return model_output["final"].detach()

        return {k: v for k, v in model_output.items() if k in output_layers}
