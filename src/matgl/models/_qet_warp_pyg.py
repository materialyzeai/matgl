"""Implementation of QET model (Warp/pure-PyTorch backend).

A Cartesian based equivariant GNN model with charge equilibration and electrostatic potential.
Uses the pure-PyTorch TensorNet Warp implementation for accelerated message passing.

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

import matgl
from matgl.config import DEFAULT_ELEMENTS
from matgl.electrostatics._elec_pot_pyg import ElectrostaticPotential
from matgl.electrostatics._fast_qeq_pyg import LinearQeq
from matgl.layers import (
    MLP,
    ActivationFunction,
    BondExpansion,
)
from matgl.layers._readout_torch import WeightedReadOut
from matgl.models._tensornetwarp_pyg import TensorEmbedding, TensorNetInteraction, compute_pair_vector_and_distance
from matgl.ops import fn_tensor_norm3, graph_transform
from matgl.utils.maths import scatter_add

from ._core import MatGLModel

if TYPE_CHECKING:
    from matgl.graph._converters_pyg import GraphConverter

logger = logging.getLogger(__file__)


class QET(MatGLModel):
    """The main QET model (Warp/pure-PyTorch backend)."""

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
            readout_type (str): Readout function type, `weighted_atom` (default) or `reduce_atom`.
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
            [
                TensorNetInteraction(num_rbf, units, activation, cutoff, equivariance_invariance_group, dtype)
                for _ in range(nblocks)
                if nblocks != 0
            ]
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
        g: torch.Tensor | dict,
        total_charge: torch.Tensor | None = None,
        state_attr: torch.Tensor | None = None,
        ext_pot: torch.Tensor | None = None,
        **kwargs,
    ):
        """

        Args:
            g: Either a PyG Data object or a dict with keys:
                - 'node_type' or 'z': Node types, shape (num_nodes,)
                - 'pos': Node positions, shape (num_nodes, 3)
                - 'edge_index': Edge indices, shape (2, num_edges)
                - 'pbc_offshift': Optional PBC offsets, shape (num_edges, 3)
                - 'batch': Optional batch indices, shape (num_nodes,)
            total_charge: total charge for a batch of graphs.
            state_attr: State attrs for a batch of graphs (not used currently).
            ext_pot: External potential for a batch of graphs, shape (num_atoms,).
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            output: Output property for a batch of graphs.
        """
        # Handle both dict and PyG Data object
        if isinstance(g, dict):
            z = g.get("node_type", g.get("z"))
            pos = g["pos"]
            edge_index = g["edge_index"]
            pbc_offshift = g.get("pbc_offshift", None)
            batch = g.get("batch", None)
            num_graphs = g.get("num_graphs", None)
        else:
            z = getattr(g, "node_type", getattr(g, "z", None))
            pos = g.pos  # type: ignore[attr-defined]
            edge_index = g.edge_index  # type: ignore[attr-defined]
            pbc_offshift = getattr(g, "pbc_offshift", None)
            batch = getattr(g, "batch", None)
            num_graphs = getattr(g, "num_graphs", None)

        # Obtain bond vectors and distances
        bond_vec, bond_dist = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)

        # Prepare CSR/CSC graph indices for warp message passing
        row_data, row_indices, row_indptr, col_data, col_indices, col_indptr = graph_transform(
            edge_index.int(),
            z.shape[0],  # type: ignore[union-attr]
        )

        # Expand distances with radial basis functions
        edge_attr = self.bond_expansion(bond_dist)

        # Embedding layer (warp TensorEmbedding uses CSR format)
        X = self.tensor_embedding(z, edge_index, bond_dist, bond_vec, edge_attr, col_data, col_indptr)

        # Interaction layers
        for layer in self.layers:
            X = layer(
                X,
                edge_index,
                bond_dist,
                edge_attr,
                row_data,
                row_indices,
                row_indptr,
                col_data,
                col_indices,
                col_indptr,
            )

        # Compute I, A, S norms and project to hidden units
        x = fn_tensor_norm3(X)
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
            hardness = torch.squeeze(self.hardness_readout[z])  # type: ignore[index]

        # Sigma per atom
        sigma = torch.squeeze(self.sigma[z])

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
            bond_dist=bond_dist,
            edge_index=edge_index,
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
            batch_long = batch.to(torch.long)
            if num_graphs is None:
                num_graphs = int(batch_long.max().item()) + 1
            return scatter_add(atomic_energy, batch_long, dim_size=num_graphs)  # type: ignore[arg-type]
        return torch.sum(atomic_energy, dim=0, keepdim=True).squeeze()

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
