"""GRACE (Graph Atomic Cluster Expansion) PyG implementation.

Single-layer GRACE potential that follows the canonical recipe from the
original gracemaker (TensorFlow) implementation:

    bond geometry → Chebyshev radial basis → real spherical harmonics
    → single-particle ACE basis ``A_i`` → Clebsch-Gordan products
    ``A, A⊗A, ..., A^{max_order}`` → ``L=0`` invariant collection
    → MLP energy readout → per-atom (or summed) energy.

The PyG-native implementation rides on matgl's existing equivariant
primitives (:class:`matgl.layers._so3.RealSphericalHarmonics`,
:class:`matgl.layers._so3.SO3TensorProduct`) and graph utilities
(:func:`matgl.graph._compute_pyg.compute_pair_vector_and_distance`,
:func:`matgl.utils.maths.scatter_add`). The GRACE-specific bits — Chebyshev
radial basis, learnable ``R_{nl}`` expansion, single-particle aggregation,
multi-order product chain — live in :mod:`matgl.layers._grace`.

References:
    Bochkarev, Lysogorskiy, Drautz. *Graph Atomic Cluster Expansion for
    Semilocal Interactions beyond Equivariant Message Passing.* Phys. Rev. X
    14, 021036 (2024).

    Lysogorskiy, Bochkarev, Drautz. *Graph atomic cluster expansion for
    foundational machine learning interatomic potentials.* arXiv:2508.17936
    (2025).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn

from matgl.config import DEFAULT_ELEMENTS
from matgl.graph._compute_pyg import compute_pair_vector_and_distance
from matgl.layers import MLP, ActivationFunction
from matgl.layers._grace import (
    ChebyshevRadialBasis,
    GraceACEStack,
    GraceSPBasis,
    LinearRadialFunction,
    collect_invariants,
)
from matgl.layers._so3 import RealSphericalHarmonics
from matgl.utils.maths import scatter_add

from ._core import MatGLModel

if TYPE_CHECKING:
    from matgl.graph._converters_pyg import GraphConverter


class GRACE(MatGLModel):
    """Single-layer GRACE potential (PyG backend).

    The model returns a scalar total energy per graph (extensive output)
    summed over per-atom contributions. It exposes the attributes
    ``cutoff: float`` and ``element_types: tuple[str, ...]`` required by
    :class:`matgl.apps.pes.Potential` to compute forces and stresses by
    autograd.
    """

    __version__ = 1

    def __init__(
        self,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        cutoff: float = 5.0,
        n_rad_base: int = 8,
        n_rad_max: int = 16,
        lmax: int = 2,
        embedding_size: int = 16,
        max_order: int = 3,
        readout_hidden: tuple[int, ...] = (64,),
        activation_type: Literal["swish", "tanh", "sigmoid", "softplus2", "softexp"] = "swish",
        cutoff_exponent: int = 5,
        avg_n_neigh: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the GRACE-1L PyG model.

        Args:
            element_types: ordered tuple of element symbols this model knows
                about. Indexes into the chemical embedding table.
            cutoff: real-space cutoff radius in Å.
            n_rad_base: number of Chebyshev radial basis functions ``g_k``.
            n_rad_max: number of learned radial channels ``n``.
            lmax: angular cutoff for spherical harmonics and ACE products.
            embedding_size: width of the per-element scalar embedding ``z``.
            max_order: ACE order; ``max_order=3`` builds ``{A, A⊗A, A⊗A⊗A}``.
                Must be ``>= 1``.
            readout_hidden: hidden widths of the scalar-output MLP.
            activation_type: activation for the readout MLP. One of
                ``"swish", "tanh", "sigmoid", "softplus2", "softexp"``.
            cutoff_exponent: polynomial degree of the cutoff envelope.
            avg_n_neigh: typical neighbor count, used to normalize the
                single-particle basis sum.
            **kwargs: reserved for future flexibility.
        """
        super().__init__()
        self.save_args(locals(), kwargs)

        if max_order < 1:
            raise ValueError("max_order must be >= 1")
        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError as err:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from err

        self.element_types = element_types  # required by Potential
        self.cutoff = float(cutoff)
        self.lmax = int(lmax)
        self.n_rad_max = int(n_rad_max)
        self.max_order = int(max_order)

        self.radial_basis = ChebyshevRadialBasis(nfunc=n_rad_base, cutoff=cutoff, cutoff_exponent=cutoff_exponent)
        self.radial_function = LinearRadialFunction(nfunc=n_rad_base, n_rad_max=n_rad_max, lmax=lmax)
        self.spherical = RealSphericalHarmonics(lmax=lmax)
        self.sp_basis = GraceSPBasis(
            lmax=lmax,
            n_rad_max=n_rad_max,
            n_elements=len(element_types),
            embedding_size=embedding_size,
            avg_n_neigh=avg_n_neigh,
        )
        self.ace_stack = GraceACEStack(lmax=lmax, max_order=max_order)

        readout_in = self.max_order * self.n_rad_max
        self.readout = MLP(
            dims=[readout_in, *readout_hidden, 1],
            activation=activation,
            activate_last=False,
            bias_last=True,
        )

    def forward(self, g: Any, state_attr: torch.Tensor | None = None, **kwargs: Any) -> torch.Tensor:
        """Compute the total energy of a (possibly batched) PyG graph.

        Args:
            g: PyG ``Data`` / ``Batch`` with ``node_type``, ``pos``,
                ``edge_index`` and (for periodic systems) ``pbc_offshift``.
                ``batch`` and ``num_graphs`` are honored when present.
            state_attr: unused, accepted for signature compatibility with
                other matgl PyG models.
            **kwargs: reserved.

        Returns:
            Scalar ``torch.Tensor`` (or ``[num_graphs]`` if batched) total
            energy. The model is extensive: the per-atom energies produced by
            the readout MLP are summed via ``scatter_add`` per graph.
        """
        del state_attr, kwargs
        node_type = getattr(g, "node_type", getattr(g, "z", None))
        if node_type is None:
            raise ValueError("Graph must carry `node_type` (or `z`) attribute with atomic-type indices.")

        pos = g.pos
        edge_index = g.edge_index
        pbc_offshift = getattr(g, "pbc_offshift", None)
        num_nodes = pos.shape[0]

        bond_vec, bond_dist = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)
        rhat = bond_vec / bond_dist.unsqueeze(-1).clamp_min(1e-10)

        basis_values = self.radial_basis(bond_dist)
        radial_nl = self.radial_function(basis_values)  # [E, (lmax+1)^2, n_rad_max]
        spherical_lm = self.spherical(rhat)  # [E, (lmax+1)^2]

        a_node = self.sp_basis(
            radial_nl=radial_nl,
            spherical_lm=spherical_lm,
            edge_index=edge_index,
            node_type=node_type,
            num_nodes=num_nodes,
        )
        equivariants = self.ace_stack(a_node)
        invariants = collect_invariants(equivariants)
        atomic_energies = self.readout(invariants).view(-1)

        batch = getattr(g, "batch", None)
        num_graphs = getattr(g, "num_graphs", None)
        if batch is None:
            return atomic_energies.sum()
        batch_long = batch.to(torch.long)
        if num_graphs is None:
            num_graphs = int(batch_long.max().item()) + 1
        return scatter_add(atomic_energies, batch_long, dim_size=num_graphs)

    def predict_structure(
        self,
        structure: Any,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
    ) -> torch.Tensor:
        """Convenience method: structure → total energy.

        Args:
            structure: pymatgen ``Structure`` or ``Molecule``.
            state_feats: state attributes (unused; accepted for signature
                compatibility with other matgl models).
            graph_converter: optional ``GraphConverter`` instance. Defaults
                to a fresh ``Structure2Graph`` parameterized with this
                model's ``element_types`` and ``cutoff``.

        Returns:
            Scalar total energy as a detached ``torch.Tensor``.
        """
        del state_feats
        if graph_converter is None:
            from matgl.ext._pymatgen_pyg import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)  # type: ignore[arg-type]
        graph, lattice, _ = graph_converter.get_graph(structure)
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lattice[0])
        graph.pos = graph.frac_coords @ lattice[0]
        return self(g=graph).detach()
