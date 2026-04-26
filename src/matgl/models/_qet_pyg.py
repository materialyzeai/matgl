"""Implementation of QET model (PyG backend).

A Cartesian based equivariant GNN model with charge equilibration and electrostatic potential.
For more details on TensorNet, please refer to::

    G. Simeon, G. de. Fabritiis, _TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular
    Potentials. _arXiv, June 10, 2023, 10.48550/arXiv.2306.06482.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch
from ase.data import atomic_numbers, covalent_radii
from torch import nn
from torch_geometric.data import Batch, Data

import matgl
from matgl.config import DEFAULT_ELEMENTS
from matgl.electrostatics._elec_pot_pyg import ElectrostaticPotential
from matgl.electrostatics._fast_qeq_pyg import LinearQeq
from matgl.graph._compute_pyg import compute_pair_vector_and_distance
from matgl.layers import (
    MLP,
    ActivationFunction,
    BondExpansion,
)
from matgl.layers._embedding_pyg import TensorEmbedding
from matgl.layers._graph_convolution_pyg import TensorNetInteraction
from matgl.layers._readout_pyg import WeightedReadOut
from matgl.utils.maths import decompose_tensor, scatter_add, tensor_norm

from ._core import MatGLModel

if TYPE_CHECKING:
    from matgl.graph._converters_pyg import GraphConverter

logger = logging.getLogger(__file__)


class QET(MatGLModel):
    """The main QET model (PyG backend)."""

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
        readout_type: Literal["set2set", "weighted_atom", "reduce_atom"] = "weighted_atom",
        task_type: Literal["classification", "regression"] = "regression",
        niters_set2set: int = 3,
        nlayers_set2set: int = 3,
        field: Literal["node_feat", "edge_feat"] = "node_feat",
        is_intensive: bool = True,
        ntargets: int = 1,
        is_sigma_train: bool = False,
        is_hardness_envs: bool = False,
        include_magmom: bool = False,
        return_features: bool = False,
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
            include_magmom (bool): Whether the magmom is returned
            is_hardness_envs (bool): Whether the hardness is environment dependent
            is_sigma_train (bool): Whether the sigma is trainable
            return_features (bool): Whether the atomic features are returned
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

        if element_types is None:
            self.element_types = DEFAULT_ELEMENTS
        else:
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

        # Hardness: element-wise parameter or environment-dependent MLP
        self.hardness_readout: nn.Parameter | nn.Module
        if is_hardness_envs is False:
            hardness = torch.ones(len(element_types))
            self.hardness_readout = torch.nn.Parameter(data=hardness)
        else:
            self.hardness_readout = MLP(dims=[units, units, units, 1], activation=nn.Softplus(), activate_last=True)

        if is_sigma_train:
            sigma = torch.ones(len(element_types))
            self.sigma = torch.nn.Parameter(data=sigma)
        else:
            self.register_buffer(
                "sigma", torch.tensor([covalent_radii[atomic_numbers[i]] for i in element_types], dtype=matgl.float_th)
            )

        self.chi_readout = MLP(dims=[units, units, units, 1], activation=nn.SiLU(), activate_last=True)
        if include_magmom:
            self.magmom_readout = MLP(
                dims=[units, units, units, 1], activation=nn.SiLU(), activate_last=False, bias_last=False
            )

        self.is_hardness_envs = is_hardness_envs
        self.qeq = LinearQeq()
        self.elec_pot = ElectrostaticPotential(element_types=element_types, cutoff=cutoff)
        self.norm = nn.LayerNorm(units + 3) if include_magmom else nn.LayerNorm(units + 2)
        # short-range energy readout
        self.final_layer = WeightedReadOut(
            in_feats=(units + 3 if include_magmom else units + 2),
            dims=[units, units],
            num_targets=ntargets,  # type: ignore
        )
        self.include_magmom = include_magmom
        self.return_features = return_features
        self.reset_parameters()

    def reset_parameters(self):
        self.tensor_embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()

    def forward(
        self,
        g: Data,
        total_charge: torch.Tensor | None = None,
        state_attr: torch.Tensor | None = None,
        ext_pot: torch.Tensor | None = None,
        **kwargs,
    ):
        """

        Args:
            g: PyG Data object for a batch of graphs.
            total_charge: total charge for a batch of graphs.
            state_attr: State attrs for a batch of graphs.
            ext_pot: External potential for a batch of graphs, shape (num_atoms,).
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            output: Output property for a batch of graphs.
        """
        # Obtain graph, with distances and relative position vectors
        bond_vec, bond_dist = compute_pair_vector_and_distance(g.pos, g.edge_index, g.pbc_offshift)
        g.bond_vec = bond_vec
        g.bond_dist = bond_dist

        # Expand distances with radial basis functions
        g.edge_attr = self.bond_expansion(g.bond_dist)

        # Embedding from edge-wise tensors to node-wise tensors
        X, _ = self.tensor_embedding(g.node_type, g.edge_index, g.edge_attr, g.bond_dist, g.bond_vec, state_attr)

        # Interaction layers
        for layer in self.layers:
            X = layer(g.edge_index, g.bond_dist, g.edge_attr, X)

        scalars, skew_metrices, traceless_tensors = decompose_tensor(X)
        x = torch.cat((tensor_norm(scalars), tensor_norm(skew_metrices), tensor_norm(traceless_tensors)), dim=-1)
        x = self.out_norm(x)
        x = self.linear(x)

        # Electronegativity: chi + optional external potential
        chi = torch.squeeze(self.chi_readout(x))
        if ext_pot is not None:
            chi = chi + ext_pot

        # Magmom
        if self.include_magmom:
            magmom = torch.squeeze(self.magmom_readout(x))  # type: ignore[operator]

        # Hardness per atom
        if self.is_hardness_envs:
            hardness = torch.squeeze(self.hardness_readout(x))  # type: ignore[operator]
        else:
            hardness = torch.squeeze(self.hardness_readout[g.node_type])  # type: ignore[index]

        # Sigma per atom
        sigma = torch.squeeze(self.sigma[g.node_type])

        # Determine batch indices
        batch = g.batch if (isinstance(g, Batch) and hasattr(g, "batch") and g.batch is not None) else None
        num_nodes = x.shape[0]

        # Charge equilibration
        charge = self.qeq(
            chi=chi,
            hardness=hardness,
            batch=batch,
            total_charge=total_charge,
        )

        # Electrostatic potential
        elec_pot = self.elec_pot(
            charge=charge,
            sigma=sigma,
            bond_dist=g.bond_dist,
            edge_index=g.edge_index,
            num_nodes=num_nodes,
        )

        # Combine node features with charge, electrostatic potential, and optionally magmom
        if self.include_magmom:
            combined_node_feat = torch.hstack(
                [x, charge.unsqueeze(dim=1), elec_pot.unsqueeze(dim=1), magmom.unsqueeze(dim=1)]
            )
        else:
            combined_node_feat = torch.hstack([x, charge.unsqueeze(dim=1), elec_pot.unsqueeze(dim=1)])

        node_feat = self.norm(combined_node_feat)
        atomic_energy = self.final_layer(node_feat)

        if self.return_features:
            return node_feat, atomic_energy

        # Aggregate per-atom energies to per-graph total energies
        if batch is not None:
            if atomic_energy.shape == (1, 1):
                atomic_energy = atomic_energy.squeeze(-1)
            else:
                atomic_energy = atomic_energy.squeeze()
            num_graphs = g.num_graphs if hasattr(g, "num_graphs") else int(batch.max().item()) + 1
            e_total = scatter_add(atomic_energy, batch, dim_size=num_graphs)
        else:
            e_total = torch.sum(atomic_energy, dim=0, keepdim=True).squeeze()

        return torch.squeeze(e_total)

    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
    ):
        """Convenience method to directly predict property from structure.

        Args:
            structure: An input crystal/molecule.
            state_feats (torch.tensor): Graph attributes
            total_charge: total charge of a structure
            graph_converter: Object that implements a get_graph_from_structure.

        Returns:
            output (torch.tensor): output property
        """
        if graph_converter is None:
            from matgl.ext._pymatgen_pyg import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)  # type: ignore
        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        g.pos = g.frac_coords @ lat[0]
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)
        if self.return_features:
            node_features, atomic_energies = self(g=g, state_attr=state_feats, total_charge=total_charge)
            return node_features.detach(), atomic_energies.detach()
        return self(g=g, state_attr=state_feats, total_charge=total_charge).detach()
