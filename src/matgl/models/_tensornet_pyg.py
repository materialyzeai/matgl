"""Implementation of TensorNet model.

A Cartesian based equivariant GNN model. For more details on TensorNet,
please refer to::

    G. Simeon, G. de. Fabritiis, _TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular
    Potentials. _arXiv, June 10, 2023, 10.48550/arXiv.2306.06482.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch_geometric.data import Batch, Data

import matgl
from matgl.config import DEFAULT_ELEMENTS
from matgl.graph._compute_pyg import (
    compute_pair_vector_and_distance,
)
from matgl.layers import (
    MLP,
    ActivationFunction,
    BondExpansion,
)
from matgl.layers._embedding_pyg import TensorEmbedding
from matgl.layers._graph_convolution_pyg import TensorNetInteraction
from matgl.layers._readout_pyg import (
    ReduceReadOut,
    WeightedAtomReadOut,
    WeightedReadOut,
)
from matgl.utils.maths import decompose_tensor, scatter_add, tensor_norm

from ._core import MatGLModel

if TYPE_CHECKING:
    from matgl.graph._converters_pyg import GraphConverter

logger = logging.getLogger(__file__)


class TensorNet(MatGLModel):
    """The main TensorNet model. The official implementation can be found in https://github.com/torchmd/torchmd-net."""

    __version__ = 1

    def __init__(
        self,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        units: int = 64,
        ntypes_state: int | None = None,
        dim_state_embedding: int = 0,
        dim_state_feats: int | None = None,
        include_state: bool = False,
        nblocks: int = 2,
        num_rbf: int = 32,
        max_n: int = 3,
        max_l: int = 3,
        rbf_type: Literal["Gaussian", "SphericalBessel"] = "Gaussian",
        use_smooth: bool = False,
        activation_type: Literal["swish", "tanh", "sigmoid", "softplus2", "softexp"] = "swish",
        cutoff: float = 5.0,
        equivariance_invariance_group: str = "O(3)",
        dtype: torch.dtype = matgl.float_th,
        width: float = 0.5,
        readout_type: Literal["weighted_atom", "reduce_atom"] = "weighted_atom",
        task_type: Literal["classification", "regression"] = "regression",
        niters_set2set: int = 3,
        nlayers_set2set: int = 3,
        field: Literal["node_feat", "edge_feat"] = "node_feat",
        is_intensive: bool = True,
        ntargets: int = 1,
        **kwargs,
    ):
        r"""

        Args:
            element_types (tuple): List of elements appearing in the dataset. Default to DEFAULT_ELEMENTS.
            units (int, optional): Hidden embedding size.
                (default: :obj:`64`)
            ntypes_state (int): Number of state labels
            dim_state_embedding (int): Number of hidden neurons in state embedding
            dim_state_feats (int): Number of state features after linear layer
            include_state (bool): Whether to include states features
            nblocks (int, optional): The number of interaction layers.
                (default: :obj:`2`)
            num_rbf (int, optional): The number of radial basis Gaussian functions :math:`\mu`.
                (default: :obj:`32`)
            max_n (int): maximum of n in spherical Bessel functions
            max_l (int): maximum of l in spherical Bessel functions
            rbf_type (str): Radial basis function. choose from 'Gaussian' or 'SphericalBessel'
            use_smooth (bool): Whether to use the smooth version of SphericalBessel functions.
                This is particularly important for the smoothness of PES.
            activation_type (str): Activation type. choose from 'swish', 'tanh', 'sigmoid', 'softplus2', 'softexp'
            cutoff (float): cutoff distance for interatomic interactions.
            equivariance_invariance_group (string, optional): Group under whose action on input
                positions internal tensor features will be equivariant and scalar predictions
                will be invariant. O(3) or SO(3).
               (default :obj:`"O(3)"`)
            dtype (torch.dtype): data type for all variables
            width (float): the width of Gaussian radial basis functions
            readout_type (str): Readout function type, `set2set`, `weighted_atom` (default) or `reduce_atom`.
            task_type (str): `classification` or `regression` (default).
            niters_set2set (int): Number of set2set iterations
            nlayers_set2set (int): Number of set2set layers
            field (str): Using either "node_feat" or "edge_feat" for Set2Set and Reduced readout
            is_intensive (bool): Whether the prediction is intensive
            ntargets (int): Number of target properties
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

        self.element_types = element_types  # type: ignore

        self.bond_expansion = BondExpansion(
            cutoff=cutoff,
            rbf_type=rbf_type,
            final=cutoff + 1.0,
            num_centers=num_rbf,
            width=width,
            smooth=use_smooth,
            max_n=max_n,
            max_l=max_l,
        )

        assert equivariance_invariance_group in ["O(3)", "SO(3)"], "Unknown group representation. Choose O(3) or SO(3)."

        self.units = units
        self.equivariance_invariance_group = equivariance_invariance_group
        self.num_layers = nblocks
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.activation = activation
        self.cutoff = cutoff
        self.dim_state_embedding = dim_state_embedding
        self.dim_state_feats = dim_state_feats
        self.include_state = include_state
        self.ntypes_state = ntypes_state
        self.task_type = task_type

        # make sure the number of radial basis functions correct for tensor embedding
        if rbf_type == "SphericalBessel":
            num_rbf = max_n

        self.tensor_embedding = TensorEmbedding(
            units=units,
            degree_rbf=num_rbf,
            activation=activation,
            ntypes_node=len(element_types),
            cutoff=cutoff,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            {
                TensorNetInteraction(num_rbf, units, activation, cutoff, equivariance_invariance_group, dtype)
                for _ in range(nblocks)
                if nblocks != 0
            }
        )

        self.out_norm = nn.LayerNorm(3 * units, dtype=dtype)
        self.linear = nn.Linear(3 * units, units, dtype=dtype)
        if is_intensive:
            input_feats = units
            if readout_type == "weighted_atom":
                self.readout = WeightedAtomReadOut(in_feats=input_feats, dims=[units, units], activation=activation)  # type:ignore[assignment]
                readout_feats = units
            else:
                self.readout = ReduceReadOut("mean", field=field)  # type: ignore
                readout_feats = input_feats  # type: ignore

            dims_final_layer = [readout_feats, units, units, ntargets]
            self.final_layer = MLP(dims_final_layer, activation, activate_last=False)
            if task_type == "classification":
                self.sigmoid = nn.Sigmoid()

        else:
            if task_type == "classification":
                raise ValueError("Classification task cannot be extensive.")
            self.final_layer = WeightedReadOut(
                in_feats=units,
                dims=[units, units],
                num_targets=ntargets,  # type: ignore
            )

        self.is_intensive = is_intensive
        self.reset_parameters()

    def reset_parameters(self):
        self.tensor_embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()

    def forward(
        self,
        g: Data,
        state_attr: torch.Tensor | None = None,
        return_all_layer_output: bool = False,
        **kwargs,
    ):
        """

        Args:
            g : PyG Graph for a batch of graphs.
            state_attr: State attrs for a batch of graphs.
            return_all_layer_output: Whether to return outputs of all layers.
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            output: output: Output property for a batch of graphs
        """
        # Obtain graph, with distances and relative position vectors
        bond_vec, bond_dist = compute_pair_vector_and_distance(g.pos, g.edge_index, g.pbc_offshift)
        g.bond_vec = bond_vec
        g.bond_dist = bond_dist

        # Expand distances with radial basis functions
        g.edge_attr = self.bond_expansion(g.bond_dist)
        # Embedding layer
        X, _ = self.tensor_embedding(g.node_type, g.edge_index, g.edge_attr, g.bond_dist, g.bond_vec, state_attr)

        fea_dict = {
            "edge_attr": g.edge_attr,
            "embedding": X,
        }

        # Interaction layers
        for i, layer in enumerate(self.layers):
            X = layer(g.edge_index, g.bond_dist, g.edge_attr, X)
            fea_dict[f"gc_{i + 1}"] = X

        scalars, skew_metrices, traceless_tensors = decompose_tensor(X)

        x = torch.cat((tensor_norm(scalars), tensor_norm(skew_metrices), tensor_norm(traceless_tensors)), dim=-1)
        x = self.out_norm(x)
        x = self.linear(x)

        g.node_feat = x
        fea_dict["readout"] = x

        if self.is_intensive:
            node_vec = self.readout(g)
            vec = node_vec  # type: ignore
            output = self.final_layer(vec)
            if self.task_type == "classification":
                output = self.sigmoid(output)
            output = torch.squeeze(output)
        else:
            atomic_energies = self.final_layer(g.node_feat)
            if isinstance(g, Batch) and hasattr(g, "batch") and g.batch is not None:
                # edge case, if we do squeeze() directly, we will get torch.size([]) and it will crash in the training.
                if atomic_energies.shape == (1, 1):
                    atomic_energies = atomic_energies.squeeze(-1)
                else:
                    atomic_energies = atomic_energies.squeeze()
                # Batch case: Use scatter_add with batch tensor
                output = scatter_add(atomic_energies, g.batch, dim_size=g.num_graphs)
            else:
                # Single graph case: Sum all energies (equivalent to scatter_add with all nodes in one graph)
                output = torch.sum(atomic_energies, dim=0, keepdim=True).squeeze()

        fea_dict["final"] = output
        if return_all_layer_output:
            return fea_dict
        return output

    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
        output_layers: list | None = None,
        return_features: bool = False,
    ):
        """Convenience method to directly predict property from structure.

        Args:
            structure: An input crystal/molecule.
            state_feats (torch.tensor): Graph attributes
            graph_converter: Object that implements a get_graph_from_structure.
            output_layers: List of names for the layer of GNN as output.
            return_features (bool): If True, return specified layer outputs. If False, only return final output.

        Returns:
            output (torch.tensor): output property
        """
        allowed_output_layers = [
            "edge_attr",
            "embedding",
            "readout",
            "final",
        ] + [f"gc_{i + 1}" for i in range(self.num_layers)]

        if not return_features:
            output_layers = ["final"]
        elif output_layers is None:
            output_layers = allowed_output_layers
        elif not isinstance(output_layers, list) or set(output_layers).difference(allowed_output_layers):
            raise ValueError(f"Invalid output_layers, it must be a sublist of {allowed_output_layers}.")

        if graph_converter is None:
            from matgl.ext._pymatgen_pyg import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)  # type: ignore
        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        g.pos = g.frac_coords @ lat[0]
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)

        if return_features:
            model_output = self(g=g, state_attr=state_feats, return_all_layer_output=True)
            return {
                k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in model_output.items()
                if k in output_layers
            }

        return self(g=g, state_attr=state_feats).detach()
