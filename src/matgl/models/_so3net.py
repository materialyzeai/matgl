"""DGL implementation of SO3Net.

A simple spherical harmonic based equivariant GNNs. For more details on SO3Net,
please refer to::

    K.T. Schütt, S.S.P. Hessmann, N.W.A. Gebauer, J. Lederer, M. Gastegger. _SchNetPack 2.0: A neural network toolbox
    for atomistic machine learning. _J. Chem. Phys. 2023, 158 (14): 144801. 10.1063/5.0138367.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import dgl
import torch
import torch.nn as nn

import matgl.layers._so3 as so3
from matgl.config import DEFAULT_ELEMENTS
from matgl.graph._compute_dgl import compute_pair_vector_and_distance
from matgl.layers import (
    MLP,
    ActivationFunction,
    ReduceReadOut,
    Set2SetReadOut,
    WeightedAtomReadOut,
    WeightedReadOut,
    build_gated_equivariant_mlp,
)
from matgl.layers._basis import RadialBesselFunction
from matgl.utils.cutoff import polynomial_cutoff

from ._core import MatGLModel

if TYPE_CHECKING:
    from matgl.graph._converters_dgl import GraphConverter

logger = logging.getLogger(__file__)


class SO3Net(MatGLModel):
    """A simple SO3-equivariant representation using spherical harmonics and Clebsch-Gordon tensor products.

    The official implementation can be found in
    https://github.com/atomistic-machine-learning/schnetpack.

    __version__ = 1
    """

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
        """Initialize the SO3Net model.

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
            readout_type (Literal): Readout function type, `Set2Set`, `weighted_atom` (default) or `reduce_atom`.
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

        self.embedding = nn.Embedding(len(element_types), dim_node_embedding, padding_idx=0)

        self.sphharm = so3.RealSphericalHarmonics(lmax=lmax)

        self.so3convs = nn.ModuleList(
            {so3.SO3Convolution(lmax, dim_node_embedding, self.radial_basis.max_n) for _ in range(self.nblocks)}
        )
        self.mixings1 = nn.ModuleList(
            {nn.Linear(dim_node_embedding, dim_node_embedding, bias=False) for _ in range(self.nblocks)}
        )
        self.mixings2 = nn.ModuleList(
            {nn.Linear(dim_node_embedding, dim_node_embedding, bias=False) for _ in range(self.nblocks)}
        )
        self.mixings3 = nn.ModuleList(
            {nn.Linear(dim_node_embedding, dim_node_embedding, bias=False) for _ in range(self.nblocks)}
        )
        self.gatings = nn.ModuleList(
            {so3.SO3ParametricGatedNonlinearity(dim_node_embedding, lmax) for _ in range(self.nblocks)}
        )

        self.so3product = so3.SO3TensorProduct(lmax)

        dim_state_feats = dim_state_embedding

        if target_property == "atomwise":
            # intensive atomic property
            if is_intensive:
                dim_final_layers = [dim_node_embedding, units, units, ntargets]
                self.final_layer = MLP(
                    dims=dim_final_layers, activation=activation, activate_last=False, bias_last=True
                )

            else:  # atomic energy
                if task_type == "classification":
                    raise ValueError("Classification task cannot be extensive.")
                self.final_layer = WeightedReadOut(
                    in_feats=dim_node_embedding,
                    dims=[units, units],
                    num_targets=ntargets,  # type: ignore
                )
        else:  # graph property, dipole_moment or polarizability
            if target_property == "graph":
                input_feats = dim_node_embedding
                if readout_type == "set2set":
                    self.readout = Set2SetReadOut(
                        in_feats=input_feats, n_iters=niters_set2set, n_layers=nlayers_set2set, field="node_feat"
                    )
                    readout_feats = 2 * input_feats + dim_state_feats if include_state else 2 * input_feats  # type: ignore
                elif readout_type == "weighted_atom":
                    self.readout = WeightedAtomReadOut(in_feats=input_feats, dims=[units, units], activation=activation)  # type:ignore[assignment]
                    readout_feats = units + dim_state_feats if include_state else units  # type: ignore
                else:
                    self.readout = ReduceReadOut("mean", field="node_feat")  # type: ignore
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
                    self.readout = build_gated_equivariant_mlp(  # type: ignore
                        n_in=dim_node_embedding,
                        n_out=ntargets,
                        n_hidden=units,
                        n_layers=nlayers_readout,
                        activation=activation,
                        sactivation=activation,
                    )

                else:
                    self.readout = MLP(  # type: ignore
                        dims=dim_readout_layers, activation=activation, activate_last=True, bias_last=True
                    )

    def forward(self, g: dgl.DGLGraph, total_charges: torch.Tensor | None = None, **kwargs):
        """Performs message passing and updates node representations.

        Args:
            g : DGLGraph for a batch of graphs.
            total_charges: a list of total charges of systems
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            output: Output property for a batch of graphs
        """
        # get tensors from input dictionary
        atomic_numbers = g.ndata["node_type"]
        g.edata["bond_vec"], g.edata["bond_dist"] = compute_pair_vector_and_distance(g)
        r_ij = g.edata["bond_vec"]
        idx_i = g.edges()[0]
        idx_j = g.edges()[1]

        # compute atom and pair features
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

        g.ndata["scalar_representation"] = x[:, 0]
        g.ndata["multipole_representation"] = x

        # extract cartesian vector from multipoles: [y, z, x] -> [x, y, z]
        if self.return_vector_representation:
            g.ndata["vector_representation"] = torch.roll(x[:, 1:4], 1, 1)

        if self.target_property == "atomwise":
            g.ndata["node_feat"] = x[:, 0]
            if self.is_intensive:
                output = self.final_layer(g.ndata["node_feat"])
            else:
                g.ndata["atomic_properties"] = self.final_layer(g)
                output = dgl.readout_nodes(g, "atomic_properties", op="sum")
            return torch.squeeze(output)

        if self.target_property == "graph":
            g.ndata["node_feat"] = x[:, 0]
            if self.is_intensive:
                node_vec = self.readout(g)
                output = self.final_layer(node_vec)
                if self.task_type == "classification":
                    output = self.sigmoid(output)
                return output

        if self.target_property == "dipole_moment":
            natoms = g.batch_num_nodes()
            if self.use_vector_representation:
                charges, atomic_dipoles = self.readout(  # type: ignore
                    (g.ndata["scalar_representation"], g.ndata["vector_representation"])
                )  # type: ignore
                atomic_dipoles = torch.squeeze(atomic_dipoles, -1)
            else:
                charges = self.readout(g.ndata["scalar_representation"])  # type: ignore
                atomic_dipoles = 0.0
            g.ndata["charges"] = charges
            if self.correct_charges:
                sum_charges = dgl.readout_nodes(g, "charges")
                charges_correction = (sum_charges - total_charges) / natoms
                charges = charges - torch.repeat_interleave(charges_correction, natoms, dim=0)
            dipole_moment = g.ndata["pos"] * charges
            if self.use_vector_representation:
                dipole_moment = dipole_moment + atomic_dipoles
                g.ndata["dipole_moment"] = dipole_moment
                # sum over atoms
                dipole_moment = dgl.readout_nodes(g, "dipole_moment", op="sum")

                if self.predict_dipole_magnitude:
                    dipole_moment = torch.norm(dipole_moment, dim=1, keepdim=False)
            return torch.squeeze(charges), torch.squeeze(dipole_moment)

        #    else:  # polarizability
        positions = g.ndata["pos"]
        l0 = g.ndata["scalar_representation"]
        l1 = g.ndata["vector_representation"]
        dim = l1.shape[-2]

        l0, l1 = self.readout((l0, l1))  # type: ignore

        # isotropic on diagonal
        alpha = l0[..., 0:1]
        size = list(alpha.shape)
        size[-1] = dim
        alpha = alpha.expand(*size)
        alpha = torch.diag_embed(alpha)

        # add anisotropic components
        mur = l1[..., None, 0] * positions[..., None, :]
        alpha_c = mur + mur.transpose(-2, -1)
        alpha = alpha + alpha_c

        # sum over atoms
        g.ndata["alpha"] = alpha
        alpha = dgl.readout_nodes(g, "alpha", op="sum")

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
            state_feats (torch.tensor): Graph attributes
            graph_converter: Object that implements a get_graph_from_structure.

        Returns:
            output (torch.tensor): output property
        """
        if graph_converter is None:
            from matgl.ext._pymatgen_dgl import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)  # type: ignore
        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lat[0])
        g.ndata["pos"] = g.ndata["frac_coords"] @ lat[0]
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)
        return self(g=g, state_attr=state_feats).detach()
