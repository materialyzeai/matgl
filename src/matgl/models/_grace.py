"""GRACE (Graph Atomic Cluster Expansion) PyG implementation.

Multi-layer GRACE potential that follows the canonical recipe from the
original gracemaker (TensorFlow) implementation:

    bond geometry → Chebyshev radial basis → real spherical harmonics
    → single-particle ACE basis ``A_i`` → Clebsch-Gordan products
    ``A, A⊗A, ..., A^{max_order}`` → ``L=0`` invariant collection
    → MLP energy readout → per-atom (or summed) energy.

When ``nblocks > 1`` (the default is ``2``), the per-atom equivariant
descriptors that produced block ``k``'s energy are also truncated to
``indicator_lmax``, mixed across cluster orders by a learned linear
projection on the radial axis, and fed forward as block ``k+1``'s
*equivariant* chemical indicator (replacing the per-element scalar
indicator). Each block contributes its own per-atom energy; the total is
the sum.

The PyG-native implementation rides on matgl's existing equivariant
primitives (:class:`matgl.layers._so3.RealSphericalHarmonics`,
:class:`matgl.layers._so3.SO3TensorProduct`) and graph utilities
(:func:`matgl.graph._compute_pyg.compute_pair_vector_and_distance`,
:func:`matgl.utils.maths.scatter_add`). The GRACE-specific bits — Chebyshev
radial basis, learnable ``R_{nl}`` expansion, single-particle aggregation
(scalar and equivariant variants), multi-order product chain — live in
:mod:`matgl.layers._grace`.

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
    GraceSPBasisEquivariant,
    LinearRadialFunction,
    collect_invariants,
)
from matgl.layers._so3 import RealSphericalHarmonics
from matgl.utils.maths import scatter_add

from ._core import MatGLModel

if TYPE_CHECKING:
    from matgl.graph._converters_pyg import GraphConverter


class GRACE(MatGLModel):
    """Multi-layer GRACE potential (PyG backend).

    A stack of ``nblocks`` ACE-style blocks: the first block uses a
    per-element scalar chemical indicator; subsequent blocks consume the
    previous block's equivariant descriptor as their chemical indicator
    (CG-coupled with ``R(r) Y(r̂)`` at the bond level via
    :class:`~matgl.layers._so3.SO3TensorProduct`). All blocks contribute
    per-atom energies, which are summed into the total.

    Setting ``nblocks=1`` recovers the single-layer model (the original
    GRACE-1L preset); ``nblocks=2`` (the default) is GRACE-2L; deeper
    stacks are supported with the same indicator-passing convention.

    For simplicity (and to ride on matgl's fixed-``lmax``
    :class:`~matgl.layers._so3.SO3TensorProduct`) every block shares the
    same ``lmax``; ``indicator_lmax`` controls which ``l`` components flow
    forward between blocks (the rest are dropped). The upstream gracemaker
    GRACE-2L preset uses different ``lmax`` per layer (e.g. ``lmax=4``
    then ``lmax=3``); that generalization is left for a follow-up.

    Exposes ``cutoff: float`` and ``element_types: tuple[str, ...]`` so
    the model composes directly with :class:`matgl.apps.pes.Potential`
    for forces / stresses / hessians via autograd.
    """

    __version__ = 2

    def __init__(
        self,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        cutoff: float = 5.0,
        n_rad_base: int = 8,
        n_rad_max: int = 144,
        lmax: int = 3,
        embedding_size: int = 128,
        max_order: int = 3,
        nblocks: int = 2,
        indicator_lmax: int = 2,
        indicator_n_max: int = 128,
        readout_hidden: tuple[int, ...] = (128,),
        activation_type: Literal["swish", "tanh", "sigmoid", "softplus2", "softexp"] = "swish",
        cutoff_exponent: int = 5,
        avg_n_neigh: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the GRACE model.

        Args:
            element_types: ordered tuple of element symbols this model knows
                about. Indexes into the chemical embedding table.
            cutoff: real-space cutoff radius in Å.
            n_rad_base: number of Chebyshev radial basis functions ``g_k``.
            n_rad_max: number of learned radial channels ``n`` (shared by
                all blocks).
            lmax: angular cutoff for spherical harmonics and ACE products
                (shared by all blocks).
            embedding_size: width of the per-element scalar embedding ``z``
                used by block 0.
            max_order: ACE order; ``max_order=3`` builds ``{A, A⊗A, A⊗A⊗A}``
                inside each block. Must be ``>= 1``.
            nblocks: number of stacked GRACE blocks. ``nblocks=1`` is the
                single-layer GRACE-1L; ``nblocks=2`` (default) is GRACE-2L.
                Must be ``>= 1``.
            indicator_lmax: angular cutoff of the per-block equivariant
                indicator that flows forward to the next block. Must
                satisfy ``0 <= indicator_lmax <= lmax``. Ignored when
                ``nblocks == 1``.
            indicator_n_max: feature width of the equivariant indicator
                after the learned linear mix across cluster orders.
                Ignored when ``nblocks == 1``.
            readout_hidden: hidden widths of the per-block scalar-output
                MLP.
            activation_type: activation for the readout MLPs.
            cutoff_exponent: polynomial degree of the cutoff envelope.
            avg_n_neigh: typical neighbor count, used to normalize the
                single-particle basis sums in every block.
            **kwargs: reserved for future flexibility.
        """
        super().__init__()
        self.save_args(locals(), kwargs)

        if nblocks < 1:
            raise ValueError("nblocks must be >= 1")
        if max_order < 1:
            raise ValueError("max_order must be >= 1")
        if nblocks > 1 and not 0 <= indicator_lmax <= lmax:
            raise ValueError(f"indicator_lmax ({indicator_lmax}) must satisfy 0 <= indicator_lmax <= lmax ({lmax}).")
        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError as err:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from err

        self.element_types = element_types
        self.cutoff = float(cutoff)
        self.lmax = int(lmax)
        self.n_rad_max = int(n_rad_max)
        self.max_order = int(max_order)
        self.nblocks = int(nblocks)
        self.indicator_lmax = int(indicator_lmax)
        self.indicator_n_max = int(indicator_n_max)

        # Shared real-Y_lm path. All blocks share the spherical harmonics;
        # only the radial expansion / SP basis / ACE products / readout are
        # per-block.
        self.spherical = RealSphericalHarmonics(lmax=lmax)

        # Chebyshev basis is parameterless and depends only on (r, cutoff,
        # cutoff_exponent) — those are identical across blocks, so we share
        # one instance and call it once per forward instead of nblocks times.
        # We keep ``self.radial_basis`` as a ``ModuleList`` of length nblocks
        # for backward compatibility with checkpoints saved by previous
        # versions of this model (every entry points at the same module so
        # parameter counts and state-dict keys are unchanged).
        shared_basis = ChebyshevRadialBasis(nfunc=n_rad_base, cutoff=cutoff, cutoff_exponent=cutoff_exponent)
        self.radial_basis = nn.ModuleList([shared_basis for _ in range(self.nblocks)])
        self.radial_function = nn.ModuleList(
            [LinearRadialFunction(nfunc=n_rad_base, n_rad_max=n_rad_max, lmax=lmax) for _ in range(self.nblocks)]
        )

        # Block 0 uses a per-element scalar indicator; blocks 1+ consume the
        # previous block's equivariant descriptor.
        sp_basis_modules: list[nn.Module] = [
            GraceSPBasis(
                lmax=lmax,
                n_rad_max=n_rad_max,
                n_elements=len(element_types),
                embedding_size=embedding_size,
                avg_n_neigh=avg_n_neigh,
            )
        ]
        sp_basis_modules.extend(
            GraceSPBasisEquivariant(
                lmax=lmax,
                n_rad_max=n_rad_max,
                indicator_lmax=indicator_lmax,
                indicator_n_max=indicator_n_max,
                avg_n_neigh=avg_n_neigh,
            )
            for _ in range(1, self.nblocks)
        )
        self.sp_basis = nn.ModuleList(sp_basis_modules)

        # The last product in each block's ACE chain only needs to expose the
        # lm components actually consumed downstream:
        #   * collect_invariants reads only L=0;
        #   * the next-block indicator concat reads up to ``indicator_lmax``.
        # So for non-final blocks we set ``last_lmax_out = indicator_lmax``;
        # the final block only feeds collect_invariants, so ``last_lmax_out=0``
        # is sufficient (cuts the last product's CG buffer 22x for lmax=3).
        ace_stacks: list[nn.Module] = []
        for k in range(self.nblocks):
            is_final_block = k == self.nblocks - 1
            last_lmax_out = 0 if is_final_block else self.indicator_lmax
            ace_stacks.append(GraceACEStack(lmax=lmax, max_order=max_order, last_lmax_out=last_lmax_out))
        self.ace_stack = nn.ModuleList(ace_stacks)

        # ``nblocks - 1`` indicator-mixing linear projections, one between
        # every consecutive pair of blocks. Each maps the
        # ``max_order * n_rad_max`` channels (concatenated across cluster
        # orders, truncated to ``indicator_lmax``) down to
        # ``indicator_n_max``.
        self.indicator_mix = nn.ModuleList(
            [
                nn.Linear(self.max_order * self.n_rad_max, self.indicator_n_max, bias=False)
                for _ in range(self.nblocks - 1)
            ]
        )

        readout_in = self.max_order * self.n_rad_max
        self.readout = nn.ModuleList(
            [
                MLP(
                    dims=[readout_in, *readout_hidden, 1],
                    activation=activation,
                    activate_last=False,
                    bias_last=True,
                )
                for _ in range(self.nblocks)
            ]
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
            Scalar total energy (or ``[num_graphs]`` if batched). The model
            is extensive: per-atom contributions from every block are summed.
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
        spherical_lm = self.spherical(rhat)
        # Chebyshev basis is parameterless so we hoist its ``forward`` out of
        # the per-block loop (every block's ``self.radial_basis[k]`` points
        # at the same shared module).
        basis_values = self.radial_basis[0](bond_dist)

        atomic_energies = torch.zeros(num_nodes, dtype=pos.dtype, device=pos.device)
        indicator: torch.Tensor | None = None
        keep = (self.indicator_lmax + 1) ** 2

        for k in range(self.nblocks):
            radial_nl = self.radial_function[k](basis_values)

            if k == 0:
                a_node = self.sp_basis[k](
                    radial_nl=radial_nl,
                    spherical_lm=spherical_lm,
                    edge_index=edge_index,
                    node_type=node_type,
                    num_nodes=num_nodes,
                )
            else:
                # ``indicator`` was set in the previous loop iteration when
                # ``k > 0`` because we always reach the ``k < nblocks - 1``
                # branch below in the prior step.
                a_node = self.sp_basis[k](
                    radial_nl=radial_nl,
                    spherical_lm=spherical_lm,
                    indicator=indicator,
                    edge_index=edge_index,
                    num_nodes=num_nodes,
                )

            equivariants = self.ace_stack[k](a_node)
            invariants = collect_invariants(equivariants)
            atomic_energies = atomic_energies + self.readout[k](invariants).view(-1)

            # Build the equivariant indicator for the next block.
            if k < self.nblocks - 1:
                equiv_concat = torch.cat([t[:, :keep, :] for t in equivariants], dim=-1)
                indicator = self.indicator_mix[k](equiv_concat)

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
