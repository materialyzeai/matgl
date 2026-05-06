"""PyG implementation of the QET model.

QET extends :class:`matgl.models._tensornet_pyg.TensorNet` (PyG) with
per-atom electronegativity / hardness / sigma readouts, a closed-form
charge-equilibration solver
(:class:`matgl.electrostatics._fast_qeq_pyg.LinearQeq`) and a
Gaussian-smeared Coulomb electrostatic potential
(:class:`matgl.electrostatics._elec_pot_pyg.ElectrostaticPotential`). The
TensorNet feature extractor is reused via ``forward_features``.
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
from matgl.utils.maths import scatter_add

from ._tensornet_pyg import TensorNet

if TYPE_CHECKING:
    from matgl.graph._converters_pyg import GraphConverter

logger = logging.getLogger(__file__)


class QET(TensorNet):
    """The main QET model (PyG backend).

    A subclass of :class:`TensorNet` (PyG) that reuses the TensorNet feature
    extraction stack (bond expansion, tensor embedding, interaction layers,
    decomposition) and adds a charge-equilibration head producing per-atom
    electronegativity, hardness, sigma, equilibrated charges and electrostatic
    potential, before running an atomic-energy readout over
    ``[node_feat, charge, elec_pot, magmom?]``.
    """

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
        ntargets: int = 1,
        is_sigma_train: bool = False,
        is_hardness_envs: bool = False,
        include_magmom: bool = False,
        return_features: bool = False,
        use_warp: bool | None = None,
        **kwargs,
    ):
        r"""Initialize the QET (PyG) model.

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
            ntargets (int): Number of target properties
            include_magmom (bool): Whether the magmom is returned (not implemented yet)
            is_hardness_envs (bool): Whether the hardness is environment dependent
            is_sigma_train (bool): Whether the sigma is trainable
            return_features (bool): Whether the atomic features are returned
            use_warp (bool | None): Whether to use warp-accelerated kernels from ``nvalchemi-toolkit-ops``.
                Same semantics as :class:`TensorNet`.
            **kwargs: For future flexibility. Not used at the moment.
        """
        # QET ignores intensive / readout-shape kwargs — it is always extensive
        # and always applies a WeightedReadOut over the wider concatenated
        # node feature. Silently drop them so callers / saved checkpoints that
        # pass them through don't crash on duplicate kwargs.
        for legacy in ("is_intensive", "readout_type", "task_type", "field"):
            kwargs.pop(legacy, None)
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
            is_intensive=False,
            ntargets=ntargets,
            use_warp=use_warp,
            **kwargs,
        )
        # Re-record the user-facing args so IOMixIn round-trips QET, not TensorNet.
        self.save_args(locals(), kwargs)

        self.is_hardness_envs = is_hardness_envs
        self.include_magmom = include_magmom
        self.return_features = return_features

        self.hardness_readout: nn.Module | nn.Parameter = (
            MLP(dims=[units, units, units, 1], activation=nn.Softplus(), activate_last=True)
            if is_hardness_envs
            else nn.Parameter(torch.ones(len(element_types)))
        )

        if is_sigma_train:
            self.sigma = nn.Parameter(torch.ones(len(element_types)))
        else:
            self.register_buffer(
                "sigma",
                torch.tensor([covalent_radii[atomic_numbers[i]] for i in element_types], dtype=matgl.float_th),
            )

        self.chi_readout = MLP(dims=[units, units, units, 1], activation=nn.SiLU(), activate_last=True)
        if include_magmom:
            self.magmom_readout = MLP(
                dims=[units, units, units, 1], activation=nn.SiLU(), activate_last=False, bias_last=False
            )

        self.qeq = LinearQeq()
        self.elec_pot = ElectrostaticPotential(element_types=element_types, cutoff=cutoff)
        extra_feats = 3 if include_magmom else 2  # +1 charge, +1 elec_pot, (+1 magmom)
        self.norm = nn.LayerNorm(units + extra_feats)
        # Replaces the parent's WeightedReadOut, which is built over the narrower units-wide feature.
        self.final_layer = WeightedReadOut(  # type: ignore[assignment]
            in_feats=units + extra_feats, dims=[units, units], num_targets=ntargets
        )

    def forward(  # type: ignore[override]
        self,
        g: Any,
        total_charge: torch.Tensor | None = None,
        state_attr: torch.Tensor | None = None,
        ext_pot: torch.Tensor | None = None,
        **kwargs,
    ):
        """Forward pass for QET (PyG).

        Args:
            g: PyG ``Data`` / ``Batch``-like object with ``node_type`` (or ``z``),
                ``pos``, ``edge_index``, and optionally ``pbc_offshift``,
                ``batch``, ``num_graphs``.
            total_charge: total charge for a batch of graphs.
            state_attr: State attrs for a batch of graphs.
            ext_pot: External potential, broadcastable to per-node.
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            Per-graph total energy (or ``(node_feat, atomic_energies)`` when
            ``return_features=True``).
        """
        fea_dict = self.forward_features(g, state_attr)
        x = fea_dict["readout"]

        chi = torch.squeeze(self.chi_readout(x))
        if ext_pot is not None:
            chi = chi + ext_pot

        node_type = getattr(g, "node_type", getattr(g, "z", None))
        if node_type is None:
            raise AttributeError("QET expects `node_type` or `z` on the input graph.")
        node_type = node_type.to(torch.long)

        if self.is_hardness_envs:
            hardness = torch.squeeze(self.hardness_readout(x))  # type: ignore[operator]
        else:
            hardness = torch.squeeze(self.hardness_readout[node_type])  # type: ignore[index]

        sigma = torch.squeeze(self.sigma[node_type])

        charge = self.qeq(g=g, total_charge=total_charge, chi=chi, hardness=hardness)
        elec_pot = self.elec_pot(g, charge=charge, sigma=sigma)
        g.charge = charge
        g.elec_pot = elec_pot

        feats = [x, charge.unsqueeze(dim=1), elec_pot.unsqueeze(dim=1)]
        if self.include_magmom:
            magmom = torch.squeeze(self.magmom_readout(x))
            feats.append(magmom.unsqueeze(dim=1))
        node_feat = self.norm(torch.hstack(feats))
        atomic_energies = self.final_layer(node_feat)

        if self.return_features:
            return node_feat, atomic_energies

        batch = getattr(g, "batch", None)
        if batch is None:
            batch = atomic_energies.new_zeros(atomic_energies.shape[0], dtype=torch.long)
            num_graphs = 1
        else:
            batch = batch.to(torch.long)
            num_graphs = int(getattr(g, "num_graphs", int(batch.max()) + 1))
        return torch.squeeze(scatter_add(atomic_energies.squeeze(-1), batch, dim_size=num_graphs))

    def predict_structure(  # type: ignore[override]
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        total_charge: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
    ):
        """Convenience method to directly predict a property from a structure.

        Args:
            structure: An input crystal/molecule.
            state_feats: Graph attributes
            total_charge: Total charge of the structure
            graph_converter: Object that implements ``get_graph``.

        Returns:
            output (torch.Tensor): output property
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
