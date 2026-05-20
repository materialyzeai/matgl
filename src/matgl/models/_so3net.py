"""PyG implementation of SO3Net.

A simple spherical harmonic based equivariant GNN. For more details on SO3Net,
please refer to::

    K.T. Schuett, S.S.P. Hessmann, N.W.A. Gebauer, J. Lederer, M. Gastegger. SchNetPack 2.0:
    A neural network toolbox for atomistic machine learning. J. Chem. Phys. 2023, 158 (14): 144801.
    10.1063/5.0138367.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn

import matgl.layers._so3 as so3
from matgl.config import DEFAULT_ELEMENTS
from matgl.graph._compute import compute_pair_vector_and_distance
from matgl.layers import (
    MLP,
    ActivationFunction,
    build_gated_equivariant_mlp,
)
from matgl.layers._basis import RadialBesselFunction
from matgl.layers._readout import Set2SetReadOut
from matgl.layers._readout_torch import (
    ReduceReadOut,
    WeightedAtomReadOut,
    WeightedReadOut,
)
from matgl.utils.cutoff import polynomial_cutoff
from matgl.utils.maths import scatter_add

from ._core import MatGLModel

if TYPE_CHECKING:
    from matgl.graph._converters import GraphConverter

logger = logging.getLogger(__name__)


def _resolve_batch(g: Any, num_nodes: int, device: torch.device) -> tuple[torch.Tensor, int]:
    """Return ``(batch, num_graphs)`` for a PyG ``Data``/``Batch`` (or a single graph)."""
    batch = getattr(g, "batch", None)
    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        num_graphs = 1
    else:
        batch = batch.to(device=device, dtype=torch.long)
        num_graphs_attr = getattr(g, "num_graphs", None)
        if num_graphs_attr is None:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        else:
            num_graphs = int(num_graphs_attr)
    return batch, num_graphs


class SO3Net(MatGLModel):
    """SO(3)-equivariant representation using spherical harmonics and Clebsch-Gordon tensor products.

    PyG counterpart of :class:`matgl.models._so3net.SO3Net`. The official reference
    implementation lives in https://github.com/atomistic-machine-learning/schnetpack.
    """

    __version__ = 1

    def __init__(
        self,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        dim_node_embedding: int = 64,
        units: int = 64,
        dim_state_embedding: int = 0,
        ntypes_state: int | None = None,
        dim_state_feats: int | None = None,
        nblocks: int = 3,
        nmax: int = 5,
        lmax: int = 3,
        cutoff: float = 5.0,
        rbf_learnable: bool = False,
        target_property: Literal["atomwise", "dipole_moment", "polarizability", "graph"] = "atomwise",
        task_type: Literal["classification", "regression"] = "regression",
        readout_type: Literal["set2set", "weighted_atom", "reduce_atom"] = "weighted_atom",
        niters_set2set: int = 3,
        nlayers_set2set: int = 3,
        nlayers_readout: int = 2,
        is_intensive: bool = True,
        include_state: bool = False,
        use_vector_representation: bool = False,
        correct_charges: bool = False,
        predict_dipole_magnitude: bool = False,
        activation_type: Literal["swish", "tanh", "sigmoid", "softplus2", "softexp"] = "swish",
        ntargets: int = 1,
        return_vector_representation: bool = False,
        **kwargs,
    ):
        """Initialize the SO3Net (PyG) model.

        Args:
            element_types (tuple): List of elements appearing in the dataset. Default to DEFAULT_ELEMENTS.
            dim_node_embedding (int): Number of embedded atomic features.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            units (int): Number of neurons in each MLP layer.
            dim_state_embedding (int): Number of hidden neurons in state embedding.
            ntypes_state (int): Number of state labels.
            dim_state_feats (int): Number of state features after linear layer.
            nblocks (int): number of interaction blocks.
            nmax (int): number of radial basis functions.
            lmax (int): maximum angular momentum of spherical harmonics basis.
            cutoff (float): Cutoff radius of the graph.
            rbf_learnable (bool): whether radial basis functions are trained or not.
            target_property (Literal): Target properties including atomwise, dipole_moment, polarizability and graph.
            task_type (Literal): `classification` or `regression` (default).
            readout_type (Literal): Readout function type, `set2set`, `weighted_atom` (default) or `reduce_atom`.
            niters_set2set (int): Number of Set2Set iterations.
            nlayers_set2set (int): Number of Set2Set layers.
            nlayers_readout (int): Number of layers for readout.
            is_intensive (bool): Whether the prediction is intensive.
            include_state (bool): Whether to include states features.
            use_vector_representation (bool): Whether to use node vector features.
            correct_charges (bool): Whether to correct the sum of atomic charges to the total charge.
            predict_dipole_magnitude (bool): Whether to predict the magnitude of dipole moment.
            activation_type (Literal): Activation type. choose from 'swish', 'tanh', 'sigmoid', 'softplus2', 'softexp'.
            ntargets (int): Number of target properties.
            return_vector_representation (bool): Whether to return the output node vectors.
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

        self.dim_node_embedding = dim_node_embedding
        self.nblocks = nblocks
        self.lmax = lmax
        self.cutoff = cutoff
        self.radial_basis = RadialBesselFunction(max_n=nmax, cutoff=cutoff, learnable=rbf_learnable)
        self.return_vector_representation = return_vector_representation
        self.element_types = element_types or DEFAULT_ELEMENTS
        self.target_property = target_property
        self.task_type = task_type
        self.include_state = include_state
        self.correct_charges = correct_charges
        self.is_intensive = is_intensive
        self.use_vector_representation = use_vector_representation
        self.predict_dipole_magnitude = predict_dipole_magnitude

        self.embedding = nn.Embedding(len(self.element_types), dim_node_embedding, padding_idx=0)

        self.sphharm = so3.RealSphericalHarmonics(lmax=lmax)

        self.so3convs = nn.ModuleList(
            [so3.SO3Convolution(lmax, dim_node_embedding, self.radial_basis.max_n) for _ in range(self.nblocks)]
        )
        self.mixings1 = nn.ModuleList(
            [nn.Linear(dim_node_embedding, dim_node_embedding, bias=False) for _ in range(self.nblocks)]
        )
        self.mixings2 = nn.ModuleList(
            [nn.Linear(dim_node_embedding, dim_node_embedding, bias=False) for _ in range(self.nblocks)]
        )
        self.mixings3 = nn.ModuleList(
            [nn.Linear(dim_node_embedding, dim_node_embedding, bias=False) for _ in range(self.nblocks)]
        )
        self.gatings = nn.ModuleList(
            [so3.SO3ParametricGatedNonlinearity(dim_node_embedding, lmax) for _ in range(self.nblocks)]
        )

        self.so3product = so3.SO3TensorProduct(lmax)

        dim_state_feats = dim_state_embedding

        if target_property == "atomwise":
            if is_intensive:
                dim_final_layers = [dim_node_embedding, units, units, ntargets]
                self.final_layer = MLP(
                    dims=dim_final_layers, activation=activation, activate_last=False, bias_last=True
                )
            else:
                if task_type == "classification":
                    raise ValueError("Classification task cannot be extensive.")
                self.final_layer = WeightedReadOut(  # type: ignore[assignment]
                    in_feats=dim_node_embedding,
                    dims=[units, units],
                    num_targets=ntargets,
                )
        elif target_property == "graph":
            input_feats = dim_node_embedding
            if readout_type == "set2set":
                self.readout = Set2SetReadOut(
                    in_feats=input_feats, n_iters=niters_set2set, n_layers=nlayers_set2set, field="node_feat"
                )
                readout_feats = 2 * input_feats + dim_state_feats if include_state else 2 * input_feats  # type: ignore
            elif readout_type == "weighted_atom":
                self.readout = WeightedAtomReadOut(  # type:ignore[assignment]
                    in_feats=input_feats, dims=[units, units], activation=activation
                )
                readout_feats = units + dim_state_feats if include_state else units  # type: ignore
            else:
                self.readout = ReduceReadOut("mean", field="node_feat")  # type: ignore[assignment]
                readout_feats = input_feats + dim_state_feats if include_state else input_feats  # type: ignore

            dims_final_layer = [readout_feats, units, units, ntargets]
            self.final_layer = MLP(dims_final_layer, activation, activate_last=False)
            if task_type == "classification":
                self.sigmoid = nn.Sigmoid()
        else:
            dim_readout_layers = [dim_node_embedding, units, units, ntargets]
            if target_property == "polarizability":
                use_vector_representation = True
            if use_vector_representation:
                self.readout = build_gated_equivariant_mlp(  # type: ignore[assignment]
                    n_in=dim_node_embedding,
                    n_out=ntargets,
                    n_hidden=units,
                    n_layers=nlayers_readout,
                    activation=activation,
                    sactivation=activation,
                )
            else:
                self.readout = MLP(  # type: ignore[assignment]
                    dims=dim_readout_layers, activation=activation, activate_last=True, bias_last=True
                )

    def forward(self, g: Any, total_charges: torch.Tensor | None = None, **kwargs):
        """Performs message passing and updates node representations.

        Args:
            g: PyG ``Data``/``Batch`` with ``node_type`` (or ``z``), ``pos``,
                ``edge_index``, and optionally ``pbc_offshift``, ``batch``,
                ``num_graphs``. ``pos`` is required for ``dipole_moment`` and
                ``polarizability`` targets.
            total_charges: Per-graph total charges (only used when
                ``correct_charges=True`` with ``target_property='dipole_moment'``).
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            Property tensor for a batch of graphs. Shape depends on ``target_property``.
        """
        atomic_numbers = getattr(g, "node_type", getattr(g, "z", None))
        if atomic_numbers is None:
            raise AttributeError("SO3Net expects `node_type` or `z` on the input graph.")
        atomic_numbers = atomic_numbers.to(torch.long)

        pos = g.pos
        edge_index = g.edge_index
        pbc_offshift = getattr(g, "pbc_offshift", None)

        idx_i = edge_index[0].to(torch.long)
        idx_j = edge_index[1].to(torch.long)

        r_ij, _ = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)
        dir_ij = r_ij / d_ij
        Yij = self.sphharm(dir_ij)
        radial_ij = torch.squeeze(self.radial_basis(d_ij))
        cutoff_ij = polynomial_cutoff(d_ij, cutoff=self.cutoff)

        x0 = self.embedding(atomic_numbers)[:, None]

        x = so3.scalar2rsh(x0, int(self.lmax))
        for so3conv, mixing1, mixing2, gating, mixing3 in zip(
            self.so3convs, self.mixings1, self.mixings2, self.gatings, self.mixings3, strict=False
        ):
            dx = so3conv(x, radial_ij, Yij, cutoff_ij, idx_i, idx_j)
            ddx = mixing1(dx)
            dx = dx + self.so3product(dx, ddx)
            dx = mixing2(dx)
            dx = gating(dx)
            dx = mixing3(dx)
            x = x + dx

        scalar_representation = x[:, 0]
        if self.return_vector_representation or self.target_property == "polarizability":
            # extract cartesian vector from multipoles: [y, z, x] -> [x, y, z]
            vector_representation = torch.roll(x[:, 1:4], 1, 1)
        else:
            vector_representation = None

        batch, num_graphs = _resolve_batch(g, num_nodes=atomic_numbers.shape[0], device=atomic_numbers.device)

        if self.target_property == "atomwise":
            if self.is_intensive:
                output = self.final_layer(scalar_representation)
            else:
                atomic_properties = self.final_layer(scalar_representation)  # type: ignore[operator]
                output = scatter_add(atomic_properties, batch, dim_size=num_graphs)
            return torch.squeeze(output)

        if self.target_property == "graph":
            if self.is_intensive:
                if isinstance(self.readout, Set2SetReadOut):
                    node_vec = self.readout(scalar_representation, batch)
                else:
                    node_vec = self.readout(scalar_representation, batch)  # type: ignore[operator]
                output = self.final_layer(node_vec)
                if self.task_type == "classification":
                    output = self.sigmoid(output)
                return output
            raise NotImplementedError("target_property='graph' with is_intensive=False is not supported.")

        if self.target_property == "dipole_moment":
            natoms = torch.bincount(batch, minlength=num_graphs)
            if self.use_vector_representation:
                assert vector_representation is not None
                charges, atomic_dipoles = self.readout(  # type: ignore[operator]
                    (scalar_representation, vector_representation)
                )
                atomic_dipoles = torch.squeeze(atomic_dipoles, -1)
            else:
                charges = self.readout(scalar_representation)  # type: ignore[operator]
                atomic_dipoles = None
            if self.correct_charges:
                if total_charges is None:
                    raise ValueError("`total_charges` is required when `correct_charges=True`.")
                sum_charges = scatter_add(charges, batch, dim_size=num_graphs)
                charges_correction = (sum_charges - total_charges) / natoms
                charges = charges - charges_correction[batch]
            dipole_moment = pos * charges
            if self.use_vector_representation:
                dipole_moment = dipole_moment + atomic_dipoles
                dipole_moment = scatter_add(dipole_moment, batch, dim_size=num_graphs)
                if self.predict_dipole_magnitude:
                    dipole_moment = torch.norm(dipole_moment, dim=1, keepdim=False)
            return torch.squeeze(charges), torch.squeeze(dipole_moment)

        # polarizability
        assert vector_representation is not None
        l0 = scalar_representation
        l1 = vector_representation
        dim = l1.shape[-2]

        l0, l1 = self.readout((l0, l1))  # type: ignore[operator]

        # isotropic on diagonal
        alpha = l0[..., 0:1]
        size = list(alpha.shape)
        size[-1] = dim
        alpha = alpha.expand(*size)
        alpha = torch.diag_embed(alpha)

        # add anisotropic components
        mur = l1[..., None, 0] * pos[..., None, :]
        alpha_c = mur + mur.transpose(-2, -1)
        alpha = alpha + alpha_c

        alpha = scatter_add(alpha, batch, dim_size=num_graphs)
        return torch.squeeze(alpha)

    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
    ):
        """Convenience method to directly predict property from structure.

        Args:
            structure: An input crystal/molecule.
            state_feats (torch.tensor): Graph attributes.
            graph_converter: Object that implements ``get_graph``.

        Returns:
            output (torch.tensor): output property
        """
        if graph_converter is None:
            from matgl.ext._pymatgen import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)  # type: ignore
        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        g.pos = g.frac_coords @ lat[0]
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)
        return self(g=g, state_attr=state_feats).detach()
