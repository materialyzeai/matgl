"""Implementation of TensorNet model.

A Cartesian based equivariant GNN model. For more details on TensorNet,
please refer to::

    G. Simeon, G. de. Fabritiis, _TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular
    Potentials. _arXiv, June 10, 2023, 10.48550/arXiv.2306.06482.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import torch
from ase.data import atomic_numbers, covalent_radii
from torch import nn

import matgl
from matgl.config import DEFAULT_ELEMENTS
from matgl.electrostatics._elec_pot_pyg import ElectrostaticPotential
from matgl.electrostatics._fast_qeq_pyg import LinearQeq
from matgl.layers import MLP
from matgl.layers._readout_torch import WeightedReadOut
from matgl.models._tensornet_pyg import TensorNet
from matgl.utils.maths import scatter_add

try:
    from matgl.layers._embedding_warp import TensorEmbedding as TensorEmbeddingWarp
    from matgl.layers._graph_convolution_warp import TensorNetInteraction as TensorNetInteractionWarp
    from matgl.ops import fn_tensor_norm3, graph_transform

    _warp_available = True
except ImportError:
    _warp_available = False

if TYPE_CHECKING:
    from matgl.graph._converters_pyg import GraphConverter

logger = logging.getLogger(__file__)


class QET(TensorNet):
    """The main QET model."""

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
        # readout_type: Literal["set2set", "weighted_atom", "reduce_atom"] = "weighted_atom",
        # task_type: Literal["classification", "regression"] = "regression",
        # niters_set2set: int = 3,
        # nlayers_set2set: int = 3,
        field: Literal["node_feat", "edge_feat"] = "node_feat",
        # is_intensive: bool = True,
        ntargets: int = 1,
        is_sigma_train: bool = False,
        is_hardness_envs: bool = False,
        include_magmom: bool = False,
        return_features: bool = False,
        use_warp: bool | None = None,
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
            include_magmom (bool): Whether the magmom is returned (not implemented yet)
            is_hardness_envs (bool): Whether the hardness is environment dependent
            is_sigma_train (bool): Whether the sigma is trainable
            return_features (bool): Whether the atomic features are returned
            use_warp (bool | None): Whether to use warp-accelerated kernels from ``nvalchemi-toolkit-ops``.
                ``None`` (default) auto-detects: warp is used when the package is installed.
                ``True`` raises ``ImportError`` if the package is not available.
                ``False`` forces the plain PyG implementation.
            **kwargs: For future flexibility. Not used at the moment.

        """
        # QET reuses TensorNet feature extraction, but always keeps an atomic-energy style readout.
        super().__init__(
            element_types=element_types,
            units=units,
            ntypes_state=ntypes_state,
            dim_state_embedding=dim_state_embedding,
            dim_state_feats=dim_state_feats,
            include_state=include_state,
            nblocks=nblocks,
            num_rbf=num_rbf,
            max_n=max_n,
            max_l=max_l,
            rbf_type=rbf_type,
            use_smooth=use_smooth,
            activation_type=activation_type,
            cutoff=cutoff,
            equivariance_invariance_group=equivariance_invariance_group,
            dtype=dtype,
            width=width,
            readout_type="weighted_atom",
            task_type="regression",
            field=field,
            is_intensive=False,
            ntargets=ntargets,
            use_warp=use_warp,
            **kwargs,
        )

        self.save_args(locals(), kwargs)

        ## QET-specific heads
        self.element_types = element_types
        self.is_hardness_envs = is_hardness_envs
        self.include_magmom = include_magmom
        self.return_features = return_features

        # check whether hardness is environment dependent property
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

        self.qeq = LinearQeq()
        self.elec_pot = ElectrostaticPotential(element_types=element_types, cutoff=cutoff)

        self.norm = nn.LayerNorm(units + 3) if include_magmom else nn.LayerNorm(units + 2)
        # short-range energy
        self.final_layer = WeightedReadOut(
            in_feats=(units + 3 if include_magmom else units + 2),  # 1 for atomic charge, 1  for elec_pot, 1 for magmom
            dims=[units, units],
            num_targets=ntargets,  # type: ignore
        )

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for name in ["hardness_readout", "chi_readout", "magmom_readout"]:
            module = getattr(self, name, None)
            if module is not None and hasattr(module, "reset_parameters"):
                module.reset_parameters()

        # if hasattr(self, "hardness_readout") and isinstance(self.hardness_readout, nn.Module):
        #     self.hardness_readout.reset_parameters()
        # if hasattr(self, "chi_readout"):
        #     self.chi_readout.reset_parameters()
        # if hasattr(self, "magmom_readout"):
        #     self.magmom_readout.reset_parameters()  # type: ignore[union-attr]
        # # if hasattr(self, "norm"):
        # #     self.norm.reset_parameters()

    def forward(
        self,
        g: Any,
        total_charge: torch.Tensor | None = None,
        state_attr: torch.Tensor | None = None,
        ext_pot: torch.Tensor | None = None,
        **kwargs,
    ):
        """

        Args:
            g : PyG Data object or dict with keys 'node_type'/'z', 'pos', 'edge_index',
               and optionally 'pbc_offshift', 'batch', 'num_graphs'.
            total_charge: total charge for a batch of graphs.
            state_attr: State attrs for a batch of graphs.
            ext_pot: External potential for a batch of graphs (N_batch, Natoms).
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            output: output: Output property for a batch of graphs
        """
        fea_dict = super().forward_features(g, state_attr)
        x = fea_dict["readout"]

        #### QET specific
        ## electronegativity: chi + external potential
        chi = (
            torch.squeeze(self.chi_readout(x)) + ext_pot if ext_pot is not None else torch.squeeze(self.chi_readout(x))
        )  # (num_nodes, 1)

        ## magmom
        # magmom = None
        if self.include_magmom:
            magmom = torch.squeeze(self.magmom_readout(x))  # (num_nodes, 1)

        ## hardness
        node_type = getattr(g, "node_type", getattr(g, "z", None))
        if node_type is None:
            raise AttributeError("QET expects `node_type` or `z` on the input graph.")
        node_type = node_type.to(torch.long)

        if self.is_hardness_envs:
            hardness = torch.squeeze(self.hardness_readout(x))  # type: ignore[operator]
        else:
            hardness = torch.squeeze(self.hardness_readout[node_type])  # type: ignore[index]

        ## sigma
        sigma = torch.squeeze(self.sigma[node_type])

        ## nihang
        # g.chi = chi
        # g.hardness = hardness
        # g.sigma = sigma
        # if magmom is not None:
        #     g.magmom = magmom

        # g = self.qeq(g=g, total_charge=total_charge)
        # g = self.elec_pot(g)

        charge = self.qeq(
            g=g,
            total_charge=total_charge,
            chi=chi,
            hardness=hardness,
        )

        elec_pot = self.elec_pot(g, charge=charge, sigma=sigma)

        combined_node_feat = (
            torch.hstack([x, charge.unsqueeze(dim=1), elec_pot.unsqueeze(dim=1), magmom.unsqueeze(dim=1)])
            if self.include_magmom
            else torch.hstack([x, charge.unsqueeze(dim=1), elec_pot.unsqueeze(dim=1)])
        )

        node_feat = self.norm(combined_node_feat)  # (N_atoms, units+2 or units+3)
        atomic_energies = self.final_layer(node_feat)

        if self.return_features:
            return node_feat, atomic_energies

        batch = getattr(g, "batch", None)
        num_graphs = getattr(g, "num_graphs", None)

        if batch is not None:
            # edge case: avoid losing the batch dimension on size-(1,1) outputs
            if atomic_energies.shape == (1, 1):
                atomic_energies = atomic_energies.squeeze(-1)
            else:
                atomic_energies = atomic_energies.squeeze()
            batch_long = batch.to(torch.long)
            if num_graphs is None:
                num_graphs = int(batch_long.max().item()) + 1
            e_total = scatter_add(atomic_energies, batch_long, dim_size=num_graphs)
        else:
            e_total = torch.sum(atomic_energies, dim=0, keepdim=True).squeeze()

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
            from matgl.ext.pymatgen import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)  # type: ignore
        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lat[0])
        g.ndata["pos"] = g.ndata["frac_coords"] @ lat[0]
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)
        if self.return_features:
            node_features, atomic_energies = self(g=g, state_attr=state_feats, total_charge=total_charge)
            return node_features.detach(), atomic_energies.detach()
        return self(g=g, state_attr=state_feats, total_charge=total_charge).detach()
