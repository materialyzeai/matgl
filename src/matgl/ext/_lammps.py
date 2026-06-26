"""LAMMPS-compatible TorchScript wrapper for MatGL Potentials.

Mirrors the MACE pattern (``mace.cli.create_lammps_model``): produces a
``torch.jit.ScriptModule`` that takes plain tensors and returns a dict of
energy / per-atom energy / forces / virials, ready to be loaded by a LAMMPS
``pair_style`` via ``torch::jit::load``.

Supported architectures (PyG backend only):

    * ``TensorNet`` (extensive head, ``use_warp=False``)
    * ``M3GNet``    (extensive head)

Other matgl models (CHGNet, MEGNet, SO3Net, QET) need follow-up work — see
the LAMMPS plugin README in the matgl repo for status.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from pymatgen.core.periodic_table import Element
from torch import Tensor, nn
from torch.autograd import grad

from matgl.graph._compute_pyg import (
    compute_pair_vector_and_distance,
    compute_theta_and_phi,
    create_line_graph_torch,
)
from matgl.layers._basis import spherical_bessel_smooth
from matgl.utils.cutoff import polynomial_cutoff
from matgl.utils.maths import decompose_tensor, tensor_norm

if TYPE_CHECKING:
    from matgl.apps._pes_pyg import Potential

logger = logging.getLogger(__name__)


_MAX_Z = 119


def _build_z_to_index(element_types: tuple[str, ...]) -> Tensor:
    """Build a length-_MAX_Z lookup buffer mapping atomic number -> internal index."""
    z_to_index = torch.full((_MAX_Z,), -1, dtype=torch.long)
    for idx, sym in enumerate(element_types):
        z = int(Element(sym).Z)
        z_to_index[z] = idx
    return z_to_index


def _y_l0_torch(cos_theta: Tensor, max_l: int) -> Tensor:
    """Pure-tensor port of ``SphericalHarmonicsFunction(use_phi=False)``.

    Returns ``Y_l^0(cos_theta) for l in [0, max_l)`` via the Legendre-polynomial
    recurrence ``P_l = ((2l-1)x P_{l-1} - (l-1) P_{l-2}) / l`` and the
    real-spherical-harmonic normalization ``sqrt((2l+1)/(4π))``. Matches the
    sympy-based reference implementation to fp32 precision.
    """
    pi = 3.141592653589793
    n = int(cos_theta.size(0))
    out = torch.empty((n, max_l), dtype=cos_theta.dtype, device=cos_theta.device)
    if max_l >= 1:
        out[:, 0] = 0.5 * (1.0 / pi) ** 0.5 * torch.ones_like(cos_theta)
    if max_l >= 2:
        out[:, 1] = 0.5 * (3.0 / pi) ** 0.5 * cos_theta
    pim2 = torch.ones_like(cos_theta)
    pim1 = cos_theta
    for lv in range(2, max_l):
        pl = ((2 * lv - 1) * cos_theta * pim1 - (lv - 1) * pim2) / lv
        out[:, lv] = ((2 * lv + 1) / (4 * pi)) ** 0.5 * pl
        pim2 = pim1
        pim1 = pl
    return out


def _m3gnet_three_body_basis_torch(
    triple_bond_lengths: Tensor,
    cos_theta: Tensor,
    max_n: int,
    max_l: int,
    cutoff: float,
) -> Tensor:
    """Pure-tensor port of M3GNet's ``SphericalBesselWithHarmonics`` (use_smooth=True, use_phi=False).

    Combines ``spherical_bessel_smooth(r, cutoff, max_n*max_l)`` with the
    Legendre Y_l^0(cos_theta) basis using the ``combine_sbf_shf`` recipe for
    ``use_phi=False``: each Y_l value is repeated ``max_n`` times to align with
    the SBF column blocks, then multiplied element-wise. Output shape
    ``(num_triples, max_n*max_l)`` matches the reference.
    """
    sbf = spherical_bessel_smooth(triple_bond_lengths, cutoff=cutoff, max_n=max_n * max_l)
    shf = _y_l0_torch(cos_theta, max_l)
    expanded_shf = shf.repeat_interleave(max_n, dim=1)
    return (sbf * expanded_shf).reshape(-1, max_n * max_l)


class _SmoothSBFExpansion(nn.Module):
    """TorchScript-friendly stand-in for ``BondExpansion(rbf_type='SphericalBessel', smooth=True)``.

    ``SphericalBesselFunction``'s forward dispatches through ``@torch.jit.ignore``
    helpers (sympy-lambdified ``funcs`` list), which blocks ``torch.jit.save``.
    The smooth-SBF basis it computes is mathematically identical to
    :func:`matgl.layers._basis.spherical_bessel_smooth`, which is implemented in
    pure tensor ops. This adapter is a drop-in replacement preserving the
    ``(N, max_n)`` output shape.
    """

    # NOTE: no class-level annotations — under ``from __future__ import
    # annotations`` TorchScript's annotation resolver treats them as
    # unresolved string types and errors with "Unknown type annotation".
    # ``__constants__`` makes scripting bake ``cutoff`` and ``max_n`` in as
    # Python literals (their values never change after __init__).

    __constants__ = ["cutoff", "max_n"]  # noqa: RUF012 — TorchScript convention.

    def __init__(self, cutoff: float, max_n: int) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.max_n = int(max_n)

    def forward(self, r: Tensor) -> Tensor:
        return spherical_bessel_smooth(r, cutoff=self.cutoff, max_n=self.max_n)


class _TensorNetKernel(nn.Module):
    """Per-atom raw-energy compute for TensorNet (PyG, no-Warp)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        # Swap SphericalBessel bond_expansion for the pure-tensor equivalent so
        # the kernel survives torch.jit.script.save. Gaussian / ExpNormal /
        # RadialBessel basis modules are already TorchScript-friendly and pass
        # through unchanged.
        # ``model.bond_expansion`` typed as ``Tensor | Module`` via nn.Module's
        # ``__getattr__``; the cast narrows it for mypy without runtime cost.
        from typing import cast

        be = cast("nn.Module", model.bond_expansion)
        if getattr(be, "rbf_type", None) == "SphericalBessel":
            if not bool(be.rbf.smooth):
                raise NotImplementedError(
                    "LAMMPS export of TensorNet currently requires use_smooth=True "
                    "for rbf_type='SphericalBessel' (non-smooth SphericalBesselFunction "
                    "uses sympy-lambdified callables incompatible with torch.jit.script). "
                    "Re-load the checkpoint with smooth=True."
                )
            self.bond_expansion: nn.Module = _SmoothSBFExpansion(
                cutoff=float(be.cutoff),
                max_n=int(be.rbf.max_n),
            )
        else:
            self.bond_expansion = be
        self.tensor_embedding = model.tensor_embedding
        self.layers = model.layers
        self.out_norm = model.out_norm
        self.linear = model.linear
        self.final_layer = model.final_layer

    def forward(
        self,
        atom_types: Tensor,
        edge_index: Tensor,
        pbc_offshift: Tensor,
        pos_s: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """Returns per-atom raw energies (pre-std/mean, pre-element-ref)."""
        bond_vec, bond_dist = compute_pair_vector_and_distance(pos_s, edge_index, pbc_offshift)
        edge_attr = self.bond_expansion(bond_dist)

        x_tensor, _ = self.tensor_embedding(atom_types, edge_index, edge_attr, bond_dist, bond_vec, None)
        for layer in self.layers:
            x_tensor = layer(edge_index, bond_dist, edge_attr, x_tensor)

        scalars, skew, traceless = decompose_tensor(x_tensor)
        x = torch.cat(
            (tensor_norm(scalars), tensor_norm(skew), tensor_norm(traceless)),
            dim=-1,
        )
        x = self.out_norm(x)
        x = self.linear(x)
        return self.final_layer(x).view(-1)


class _M3GNetKernel(nn.Module):
    """Per-atom raw-energy compute for M3GNet (PyG, extensive).

    Drops ``model.basis_expansion`` (``SphericalBesselWithHarmonics``) entirely
    — its ``sbf`` / ``shf`` submodules carry sympy-lambdified Python lists that
    don't survive ``torch.jit.script``. Instead we recompute the three-body
    basis with :func:`_m3gnet_three_body_basis_torch`, which handles M3GNet's
    one combination (``use_smooth=True``, ``use_phi=False``) in pure tensor
    ops.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        # Skip ``bond_expansion`` and ``basis_expansion`` — both wrap
        # ``SphericalBesselFunction``, which uses sympy-lambdified Python
        # callables that don't survive ``torch.jit.script.save``. We
        # recompute their outputs in pure tensor ops below.
        self.embedding = model.embedding
        self.three_body_interactions = model.three_body_interactions
        self.graph_layers = model.graph_layers
        self.final_layer = model.final_layer
        self.threebody_cutoff: float = float(model.threebody_cutoff)
        self.n_blocks: int = int(model.n_blocks)
        self.basis_max_n: int = int(model.basis_expansion.max_n)
        self.basis_max_l: int = int(model.basis_expansion.max_l)
        self.basis_cutoff: float = float(model.cutoff)
        # bond_expansion's RBF — stored separately because both
        # SphericalBesselFunction (for SphericalBessel rbf_type) and the
        # alternatives need different reconstruction.
        self.bond_max_n: int = int(model.bond_expansion.rbf.max_n)
        # Validate the supported configuration:
        if model.bond_expansion.rbf_type != "SphericalBessel":
            raise NotImplementedError(
                "LAMMPS export of M3GNet currently requires "
                "rbf_type='SphericalBessel'; got "
                f"{model.bond_expansion.rbf_type!r}."
            )
        if not bool(model.bond_expansion.rbf.smooth):
            raise NotImplementedError(
                "LAMMPS export of M3GNet currently requires use_smooth=True for the bond expansion."
            )
        if not bool(model.basis_expansion.use_smooth):
            raise NotImplementedError(
                "LAMMPS export of M3GNet currently requires use_smooth=True "
                "for the three-body basis. Re-train or re-load with "
                "use_smooth=True."
            )
        if bool(model.basis_expansion.use_phi):
            raise NotImplementedError(
                "LAMMPS export of M3GNet currently requires use_phi=False (matches the standard PES configuration)."
            )

    def forward(
        self,
        atom_types: Tensor,
        edge_index: Tensor,
        pbc_offshift: Tensor,
        pos_s: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """Returns per-atom raw energies (pre-std/mean, pre-element-ref)."""
        bond_vec, bond_dist = compute_pair_vector_and_distance(pos_s, edge_index, pbc_offshift)
        # Tensor-only smooth-SBF (replaces ``self.bond_expansion`` for the
        # ``rbf_type='SphericalBessel'`` + ``smooth=True`` configuration).
        expanded_dists = spherical_bessel_smooth(bond_dist, cutoff=self.basis_cutoff, max_n=self.bond_max_n)

        # Line graph (3-body): tensor-only build; no PyG Data, no numpy.
        l_g = create_line_graph_torch(edge_index, bond_dist, bond_vec, num_nodes, self.threebody_cutoff)
        angles = compute_theta_and_phi(l_g["bond_vec"], l_g["bond_dist"], l_g["line_edge_index"])

        # Tensor-only spherical-Bessel x spherical-harmonic basis.
        three_body_basis = _m3gnet_three_body_basis_torch(
            angles["triple_bond_lengths"],
            angles["cos_theta"],
            self.basis_max_n,
            self.basis_max_l,
            self.basis_cutoff,
        )

        three_body_cutoff = polynomial_cutoff(bond_dist, self.threebody_cutoff)

        node_feat, edge_feat, state_feat = self.embedding(atom_types, expanded_dists, None)

        edge_dst_atom = edge_index[1]
        line_edge_index = l_g["line_edge_index"]
        n_triple_ij = l_g["n_triple_ij"]
        num_bonds = int(edge_index.size(1))

        # TorchScript can't index ModuleList with a non-literal int, so we
        # iterate over the two lists in parallel via zip(...). ``strict=True``
        # would be safer but TorchScript doesn't accept the kwarg here, and
        # ``three_body_interactions`` and ``graph_layers`` are constructed
        # together in M3GNet's ``__init__`` so they're always the same length.
        for tbi, gl in zip(self.three_body_interactions, self.graph_layers):  # noqa: B905
            edge_feat = tbi(
                edge_dst_atom,
                line_edge_index,
                n_triple_ij,
                num_bonds,
                three_body_basis,
                three_body_cutoff,
                node_feat,
                edge_feat,
            )
            edge_feat, node_feat, state_feat = gl(
                edge_index,
                edge_feat,
                node_feat,
                state_feat,
                expanded_dists,
                None,  # node_batch — single graph
                None,  # edge_batch
                num_nodes,
                1,  # num_graphs
            )

        return self.final_layer(node_feat).view(-1)


class LAMMPSMatGLModel(nn.Module):
    """TorchScript-friendly wrapper around a MatGL ``Potential`` for LAMMPS.

    Takes plain tensors (Cartesian positions, edge_index, integer image shifts,
    cell, atomic numbers, ghost mask) and returns a dict of total local energy,
    per-atom node energies, forces, and virials. Replicates the autograd
    machinery from :class:`matgl.apps.pes.Potential` but driven by Cartesian
    inputs so LAMMPS doesn't have to materialize fractional coordinates.

    Architecture-specific feature compute lives in a small inner ``kernel``
    module (``_TensorNetKernel`` or ``_M3GNetKernel``), so the strain / autograd
    machinery here is single-source.

    Limitations:
        * Inner model must be ``TensorNet`` or ``M3GNet`` (PyG backend,
          extensive head, no-Warp for TensorNet).
        * Per-atom virials and the Hessian path are not exported.
        * ``data_mean`` is added once to ``total_energy_local``; multi-rank
          LAMMPS therefore requires ``data_mean == 0`` for correctness. The
          standard MatGL PES checkpoints satisfy this — element offsets are
          carried by ``element_refs`` instead.
    """

    # NOTE: deliberately *no* class-level Tensor annotations on the registered
    # buffers — they trip TorchScript's annotation resolver (it sees them as
    # unresolved string annotations under ``from __future__ import annotations``
    # and fails with "Unknown type annotation"). Buffer types are still
    # inferred correctly from ``register_buffer`` calls. mypy access to
    # ``self.data_mean`` etc. is narrowed locally with ``cast`` where needed.

    def __init__(
        self,
        potential: Potential,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        # Imports kept local so the public ``matgl.ext`` namespace doesn't
        # hard-require these submodules at import time.
        from matgl.models._m3gnet_pyg import M3GNet
        from matgl.models._tensornet_pyg import TensorNet

        model = potential.model

        if isinstance(model, TensorNet):
            if getattr(model, "_use_warp", False):
                raise ValueError(
                    "Inner TensorNet was constructed with use_warp=True. Re-load "
                    "or rebuild it with use_warp=False before exporting for "
                    "LAMMPS — Warp custom-ops are not TorchScript-compatible."
                )
            self.kernel: nn.Module = _TensorNetKernel(model)
        elif isinstance(model, M3GNet):
            self.kernel = _M3GNetKernel(model)
        else:
            raise NotImplementedError(
                "LAMMPSMatGLModel currently supports TensorNet (PyG) and "
                f"M3GNet (PyG); got {type(model).__name__}. CHGNet/MEGNet/"
                "SO3Net/QET are not yet exported."
            )

        if model.is_intensive:
            raise ValueError(
                f"Inner {type(model).__name__} has is_intensive=True. "
                "LAMMPS expects an extensive PES; only is_intensive=False "
                "is supported."
            )

        if potential.calc_repuls:
            raise NotImplementedError("ZBL repulsion (calc_repuls=True) export is not yet supported.")

        if potential.calc_magmom or potential.calc_charge:
            raise NotImplementedError("Magmom / charge heads are not exported for LAMMPS.")

        # Bake in the cutoff and group as Python attrs (immutable in script).
        self.cutoff: float = float(model.cutoff)
        self.r_max: float = float(model.cutoff)
        self.n_species: int = len(model.element_types)

        # Buffers carried over from Potential. ``Potential`` stores these
        # via ``register_buffer``, so mypy sees them as ``Tensor | Module``;
        # the ``cast`` keeps the typing tight without any runtime cost.
        from typing import cast

        data_mean = cast("Tensor", potential.data_mean).detach().to(dtype)
        data_std = cast("Tensor", potential.data_std).detach().to(dtype)
        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_std", data_std)

        # Per-element reference energies (1D, indexed by internal element idx).
        if potential.element_refs is not None:
            ref = potential.element_refs.property_offset.detach().to(dtype)
            if ref.dim() != 1:
                raise NotImplementedError("State-conditional element_refs (>1D) not supported.")
            if ref.numel() != self.n_species:
                if ref.numel() < self.n_species:
                    ref = torch.cat([ref, torch.zeros(self.n_species - ref.numel(), dtype=dtype)])
                else:
                    ref = ref[: self.n_species]
        else:
            ref = torch.zeros(self.n_species, dtype=dtype)
        self.register_buffer("element_refs", ref)

        # Z -> internal element index lookup. C++ passes Z; we translate.
        self.register_buffer("z_to_index", _build_z_to_index(model.element_types))

        # Atomic numbers in element_types order — useful for inspection.
        atomic_numbers = torch.tensor([int(Element(s).Z) for s in model.element_types], dtype=torch.long)
        self.register_buffer("atomic_numbers", atomic_numbers)

        if abs(float(data_mean.item() if data_mean.ndim == 0 else 0.0)) > 1e-8:
            logger.warning(
                "Exported model has non-zero data_mean=%s. Multi-rank LAMMPS "
                "runs will double-count this offset; use single-rank Kokkos.",
                float(data_mean.item()) if data_mean.ndim == 0 else data_mean,
            )

        # Cast the wrapper to the target dtype. The inner model is already the
        # checkpoint dtype; the conversion above ensures buffers match.
        self.to(dtype)

    def forward(
        self,
        positions: Tensor,
        edge_index: Tensor,
        unit_shifts: Tensor,
        cell: Tensor,
        atomic_numbers: Tensor,
        local_or_ghost: Tensor,
        compute_virials: bool,
    ) -> dict[str, Tensor]:
        """Energy / forces / virials for a single LAMMPS configuration.

        Args:
            positions: Cartesian coordinates, shape (N, 3). N includes
                ghost atoms when running in domain-decomposed mode.
            edge_index: COO edges, shape (2, E), int64.
            unit_shifts: Integer image vectors per edge, shape (E, 3),
                int64. The destination atom's effective position is
                ``positions[dst] + unit_shifts @ cell``.
            cell: Lattice basis as row vectors, shape (3, 3).
            atomic_numbers: Per-atom Z, shape (N,), int64.
            local_or_ghost: True for owned atoms, False for ghosts; shape
                (N,), bool. Only owned atoms contribute to the energy sum.
            compute_virials: Whether to compute the virial tensor.

        Returns:
            dict with keys ``total_energy_local`` (scalar), ``node_energy``
            (N,), ``forces`` (N, 3), and ``virials`` (3, 3).
        """
        atom_types = self.z_to_index[atomic_numbers]

        strain = torch.zeros((3, 3), dtype=positions.dtype, device=positions.device)
        if compute_virials:
            strain.requires_grad_(True)

        eye = torch.eye(3, dtype=positions.dtype, device=positions.device)
        deformation = eye + strain
        pos_s = positions @ deformation
        cell_s = cell @ deformation
        pos_s.requires_grad_(True)

        pbc_offshift = unit_shifts.to(positions.dtype) @ cell_s

        num_nodes = int(positions.size(0))
        atomic_energies_raw = self.kernel(atom_types, edge_index, pbc_offshift, pos_s, num_nodes)

        node_energy = self.data_std * atomic_energies_raw + self.element_refs[atom_types]
        masked = node_energy.masked_fill(~local_or_ghost, 0.0)
        total_energy_local = masked.sum() + self.data_mean

        # TorchScript demands the explicit Optional[Tensor] typing on grad_outputs.
        grad_outputs: list[torch.Tensor | None] = [torch.ones_like(total_energy_local)]

        if compute_virials:
            grads = grad(
                outputs=[total_energy_local],
                inputs=[pos_s, strain],
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False,
            )
            pos_grad = grads[0]
            strain_grad = grads[1]
            forces = -pos_grad if pos_grad is not None else torch.zeros_like(positions)
            virials = (
                strain_grad
                if strain_grad is not None
                else torch.zeros((3, 3), dtype=positions.dtype, device=positions.device)
            )
        else:
            grads = grad(
                outputs=[total_energy_local],
                inputs=[pos_s],
                grad_outputs=grad_outputs,
                create_graph=False,
                retain_graph=False,
            )
            pos_grad = grads[0]
            forces = -pos_grad if pos_grad is not None else torch.zeros_like(positions)
            virials = torch.zeros((3, 3), dtype=positions.dtype, device=positions.device)

        return {
            "total_energy_local": total_energy_local,
            "node_energy": node_energy,
            "forces": forces,
            "virials": virials,
        }


def export_lammps_model(
    potential: Potential,
    output_path: str,
    dtype: torch.dtype = torch.float32,
    script: bool = True,
) -> LAMMPSMatGLModel:
    """Wrap a :class:`Potential` and save a LAMMPS-loadable artifact.

    Args:
        potential: A trained MatGL ``Potential`` (TensorNet or M3GNet on the
            PyG backend, extensive head).
        output_path: Where to write the ``.pt`` file. The C++ pair_style
            loads this with ``torch::jit::load``.
        dtype: Wrapper buffer dtype. Forces ``float32`` or ``float64``.
        script: When True (default), runs ``torch.jit.script`` and saves the
            script module. When False, saves the eager wrapper as a regular
            PyTorch checkpoint (useful for debugging — not loadable from C++).

    Returns:
        The wrapper instance (eager).
    """
    wrapper = LAMMPSMatGLModel(potential=potential, dtype=dtype)
    wrapper.eval()

    if script:
        scripted = torch.jit.script(wrapper)
        scripted.save(output_path)
    else:
        torch.save(wrapper, output_path)

    return wrapper
