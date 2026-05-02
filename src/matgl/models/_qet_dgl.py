"""Implementation of QET model.

QET extends :class:`matgl.models._tensornet_dgl.TensorNet` with per-atom
electronegativity / hardness / sigma readouts, a closed-form
charge-equilibration solver (:class:`matgl.electrostatics._fast_qeq_dgl.LinearQeq`)
and a Gaussian-smeared Coulomb electrostatic potential
(:class:`matgl.electrostatics._elec_pot_dgl.ElectrostaticPotential`). The
TensorNet feature extractor is reused via ``forward_features``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import dgl
import torch
from ase.data import atomic_numbers, covalent_radii
from torch import nn

import matgl
from matgl.config import DEFAULT_ELEMENTS
from matgl.electrostatics._elec_pot_dgl import ElectrostaticPotential
from matgl.electrostatics._fast_qeq_dgl import LinearQeq
from matgl.layers import MLP
from matgl.layers._readout_dgl import WeightedReadOut

from ._tensornet_dgl import TensorNet

if TYPE_CHECKING:
    from matgl.graph.converters import GraphConverter

logger = logging.getLogger(__file__)


class QET(TensorNet):
    """The main QET model.

    A subclass of :class:`TensorNet` that reuses the TensorNet feature
    extraction stack (bond expansion, tensor embedding, interaction layers,
    decomposition) and adds a charge-equilibration head that produces
    per-atom electronegativity, hardness, sigma, equilibrated charges and
    electrostatic potential, before running an atomic-energy readout over
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
            readout_type (str): Accepted for IOMixIn compatibility; QET always uses an
                atomic-energy ``WeightedReadOut`` over the concatenated node features.
            task_type (str): Accepted for IOMixIn compatibility; QET is always regression.
            niters_set2set (int): Accepted for IOMixIn compatibility; unused by QET.
            nlayers_set2set (int): Accepted for IOMixIn compatibility; unused by QET.
            field (str): Accepted for IOMixIn compatibility; unused by QET.
            is_intensive (bool): Accepted for IOMixIn compatibility; QET is always extensive.
            ntargets (int): Number of target properties
            include_magmom (bool): Whether the magmom is returned (not implemented yet)
            is_hardness_envs (bool): Whether the hardness is environment dependent
            is_sigma_train (bool): Whether the sigma is trainable
            return_features (bool): Whether the atomic features are returned
            **kwargs: For future flexibility. Not used at the moment.

        """
        # Defer reset_parameters until after QET-specific heads are built so the
        # random init stream order matches the pre-subclass implementation
        # (otherwise the parent's reset would shift the stream and break
        # existing checkpoints / regression tests).
        self._qet_init_complete = False
        # QET reuses TensorNet feature extraction. is_intensive/task_type/readout_type
        # are forced to the QET-specific path; the user-facing signature preserves the
        # other args verbatim so existing model.json files round-trip via IOMixIn.
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
            niters_set2set=niters_set2set,
            nlayers_set2set=nlayers_set2set,
            field=field,
            is_intensive=False,
            ntargets=ntargets,
            **kwargs,
        )
        # Re-record the user-facing constructor args so IOMixIn round-trips QET, not TensorNet.
        self.save_args(locals(), kwargs)

        self.is_hardness_envs = is_hardness_envs
        self.include_magmom = include_magmom
        self.return_features = return_features

        self.hardness_readout: nn.Parameter | nn.Module
        if not is_hardness_envs:
            self.hardness_readout = torch.nn.Parameter(data=torch.ones(len(element_types)))
        else:
            self.hardness_readout = MLP(dims=[units, units, units, 1], activation=nn.Softplus(), activate_last=True)

        if is_sigma_train:
            self.sigma = torch.nn.Parameter(data=torch.ones(len(element_types)))
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
        self.norm = nn.LayerNorm(units + 3) if include_magmom else nn.LayerNorm(units + 2)
        # QET reads atomic energies from the wider concatenated node_feat.
        self.final_layer = WeightedReadOut(
            in_feats=(units + 3 if include_magmom else units + 2),  # +1 charge, +1 elec_pot, (+1 magmom)
            dims=[units, units],
            num_targets=ntargets,  # type: ignore
        )

        # All heads built; now do the deferred reset_parameters at the same
        # point in the random stream as the original non-subclassed QET.
        self._qet_init_complete = True
        self.reset_parameters()

    def _build_readout(self, *args, **kwargs) -> None:
        """Skip the parent's readout build; QET constructs its own ``final_layer``
        in :meth:`__init__` over the wider concatenated feature. Suppressing the
        parent build also keeps the random init order identical to the original
        non-subclassed QET, so saved checkpoints / regression tests still match.
        """
        return

    def reset_parameters(self) -> None:
        """Reset trainable parameters of the inherited TensorNet stack.

        While ``self._qet_init_complete`` is ``False`` (i.e. QET is still inside
        its own ``__init__``), this is a no-op so that the parent's automatic
        ``reset_parameters()`` call inside ``super().__init__`` does not perturb
        the random stream. QET re-invokes this method at the end of its own
        ``__init__`` once all heads have been built.
        """
        if not getattr(self, "_qet_init_complete", False):
            return
        super().reset_parameters()

    def forward(  # type: ignore[override]
        self,
        g: dgl.DGLGraph,
        total_charge: torch.Tensor | None = None,
        state_attr: torch.Tensor | None = None,
        ext_pot: torch.Tensor | None = None,
        **kwargs,
    ):
        """

        Args:
            g : DGLGraph for a batch of graphs.
            total_charge: total charge for a batch of graphs.
            state_attr: State attrs for a batch of graphs.
            ext_pot: External potential for a batch of graphs (N_batch, Natoms).
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            output: Output property for a batch of graphs
        """
        if ext_pot is not None:
            g.ndata["chi_ext"] = ext_pot

        fea_dict = self.forward_features(g=g, state_attr=state_attr)
        x = fea_dict["readout"]

        g.ndata["chi"] = (
            torch.squeeze(self.chi_readout(x)) + g.ndata["chi_ext"]
            if "chi_ext" in g.ndata
            else torch.squeeze(self.chi_readout(x))
        )

        if self.include_magmom:
            g.ndata["magmom"] = torch.squeeze(self.magmom_readout(x))

        if self.is_hardness_envs:
            g.ndata["hardness"] = torch.squeeze(self.hardness_readout(x))  # type: ignore[operator]
        else:
            g.ndata["hardness"] = torch.squeeze(self.hardness_readout[g.ndata["node_type"]])  # type: ignore[index]

        g.ndata["sigma"] = torch.squeeze(self.sigma[g.ndata["node_type"]])

        g = self.qeq(g=g, total_charge=total_charge)
        g = self.elec_pot(g)

        combined_node_feat = (
            torch.hstack(
                [
                    x,
                    g.ndata["charge"].unsqueeze(dim=1),
                    g.ndata["elec_pot"].unsqueeze(dim=1),
                    g.ndata["magmom"].unsqueeze(dim=1),
                ]
            )
            if self.include_magmom
            else torch.hstack([x, g.ndata["charge"].unsqueeze(dim=1), g.ndata["elec_pot"].unsqueeze(dim=1)])
        )
        g.ndata["node_feat"] = self.norm(combined_node_feat)
        g.ndata["atomic_energy"] = self.final_layer(g)

        if self.return_features:
            return g.ndata["node_feat"], g.ndata["atomic_energy"]

        return torch.squeeze(dgl.readout_nodes(g, "atomic_energy", op="sum"))

    def predict_structure(  # type: ignore[override]
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
