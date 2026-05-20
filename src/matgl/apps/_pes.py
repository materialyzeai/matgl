""":class:`Potential` wraps an energy-predicting PyG graph model.

Returns energies, forces, stresses, and (optionally) Hessian / partial
charges / magnetic moments. See :mod:`matgl.apps.pes` for unit conventions.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.autograd import grad
from torch_geometric.data import Batch, Data

import matgl
from matgl.layers._atom_ref import AtomRef
from matgl.layers._zbl import NuclearRepulsion
from matgl.utils.io import IOMixIn

if TYPE_CHECKING:
    import numpy as np

# 1 eV/Å³ = 160.21766208 GPa. Stress is autograd of energy w.r.t. strain (eV)
# divided by volume (Å³), giving eV/Å³; multiply by this constant for GPa.
EV_PER_ANG3_TO_GPA = 160.21766208


class Potential(nn.Module, IOMixIn):
    """Interatomic potential wrapping a PyG energy model.

    ``Potential`` takes any PyG graph model that maps a graph to a scalar
    per-graph energy (M3GNet, CHGNet, TensorNet, ...) and produces forces,
    stress, and optionally the Hessian via PyTorch autograd. The wrapped
    model's ``__call__`` is expected to accept the keyword arguments
    ``g``, ``state_attr``, and ``l_g`` (and, when ``calc_charge=True``,
    additionally ``total_charge`` and ``ext_pot``), and return a scalar
    energy tensor of shape ``(num_graphs,)``.

    Outputs are denormalised with ``data_std * E_pred + data_mean`` and,
    if ``element_refs`` is supplied, shifted by a per-atomic-number
    reference summed over the structure (see :class:`AtomRef`). The ZBL
    repulsion (:class:`NuclearRepulsion`) is optionally added when
    ``calc_repuls=True``; it requires ``model.cutoff`` and
    ``model.element_types`` to be defined.

    Units (matching matgl's conventions):

    * energy: eV per structure;
    * forces: eV/A;
    * stress: GPa, compressive-negative -- see the "Model Training"
      section of the project README;
    * Hessian (when ``calc_hessian=True``): eV/A^2, shape
      ``(3*num_atoms, 3*num_atoms)``.

    Save/load goes through :class:`~matgl.utils.io.IOMixIn`: ``self.save_args(locals())``
    in ``__init__`` records the constructor arguments, so the standard
    ``model.pt`` / ``state.pt`` / ``model.json`` triple round-trips the
    wrapped model and all options. ``__version__`` is bumped whenever
    serialised checkpoints would otherwise become invalid.
    """

    __version__ = 3

    # Class-level annotations narrow ``nn.Module.__getattr__``'s ``Tensor | Module``
    # return type to ``Tensor`` for these registered buffers, so mypy accepts
    # ``self.data_mean.device`` and ``self._eye3 + st`` below.
    data_mean: torch.Tensor
    data_std: torch.Tensor
    _eye3: torch.Tensor

    def __init__(
        self,
        model: nn.Module,
        data_mean: torch.Tensor | float = 0.0,
        data_std: torch.Tensor | float = 1.0,
        element_refs: torch.Tensor | np.ndarray | None = None,
        calc_forces: bool = True,
        calc_stresses: bool = True,
        calc_hessian: bool = False,
        calc_magmom: bool = False,
        calc_charge: bool = False,
        calc_repuls: bool = False,
        zbl_trainable: bool = False,
        debug_mode: bool = False,
        compile_model: bool = False,
        compile_mode: str = "reduce-overhead",
    ):
        """Initialize Potential from a model and elemental references.

        Args:
            model: Model for predicting energies.
            data_mean: Mean of target.
            data_std: Std dev of target.
            element_refs: Element reference values for each element.
            calc_forces: Enable force calculations.
            calc_stresses: Enable stress calculations.
            calc_hessian: Enable hessian calculations.
            calc_magmom: Enable site-wise property calculation.
            calc_charge: Enable charge property calculation
            calc_repuls: Whether the ZBL repulsion is included
            zbl_trainable: Whether zbl repulsion is trainable
            debug_mode: Return gradient of total energy with respect to atomic positions and lattices for checking
            compile_model: If True, wrap ``model`` with ``torch.compile`` using
                ``dynamic=True``. Off by default; opt in for inference / MD /
                training where the ~1.6-2x reduction in per-step graph-kernel
                overhead pays off. Compatible with training (``model.train()`` +
                ``create_graph=True``) because we disable AOTAutograd's
                donated-buffer optimization. Automatically falls back to eager
                when ``calc_hessian=True`` because torch.compile cannot trace
                double-backward.
            compile_mode: ``mode`` argument forwarded to ``torch.compile``.
                Defaults to ``"reduce-overhead"``.
        """
        super().__init__()
        self.save_args(locals())
        # torch.compile cannot handle the Hessian path because AOTAutograd has no
        # support for double-backward through compiled functions. Silently fall
        # back to eager when the caller asks for both compile and Hessian — the
        # Hessian path is rare and would otherwise crash deep inside the engine
        # with "torch.compile with aot_autograd does not currently support double
        # backward".
        if compile_model and not calc_hessian:
            # AOTAutograd's donated-buffer pass assumes ``create_graph=False`` and
            # ``retain_graph=False`` on the backward call, which is incompatible
            # with our training path (force-loss double-backward). Disable it so
            # the compiled module survives ``model.train()`` invocations.
            # See ``torch._functorch.config.donated_buffer`` and the error
            # "This backward function was compiled with non-empty donated buffers".
            import torch._functorch.config as _ftconfig

            _ftconfig.donated_buffer = False  # type: ignore[attr-defined]
            self.model = torch.compile(model, mode=compile_mode, dynamic=True)
        else:
            self.model = model
        self.calc_forces = calc_forces
        self.calc_stresses = calc_stresses
        self.calc_hessian = calc_hessian
        self.calc_magmom = calc_magmom
        self.element_refs: AtomRef | None
        self.debug_mode = debug_mode
        self.calc_repuls = calc_repuls
        self.calc_charge = calc_charge

        if calc_repuls:
            cutoff: float = self.model.cutoff  # type: ignore[assignment,attr-defined]
            self.repuls = NuclearRepulsion(cutoff, trainable=zbl_trainable)

        if element_refs is not None:
            if not isinstance(element_refs, torch.Tensor):
                element_refs = torch.tensor(element_refs, dtype=matgl.float_th)
            self.element_refs = AtomRef(property_offset=element_refs)
        else:
            self.element_refs = None
        # for backward compatibility
        if data_mean is None:
            data_mean = 0.0
        if not isinstance(data_mean, torch.Tensor):
            data_mean = torch.tensor(data_mean, dtype=matgl.float_th)
        if not isinstance(data_std, torch.Tensor):
            data_std = torch.tensor(data_std, dtype=matgl.float_th)

        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_std", data_std)
        # Identity used in strain expansion `lat @ (I + ε)`. Registering as a buffer
        # avoids allocating a fresh 3x3 every forward and follows .to(device) moves.
        self.register_buffer("_eye3", torch.eye(3, dtype=matgl.float_th), persistent=False)

    def forward(
        self,
        g: Data,
        lat: torch.Tensor,
        state_attr: torch.Tensor | np.ndarray | None = None,
        l_g: Data | None = None,
        total_charge: torch.Tensor | None = None,
        ext_pot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Compute energies, forces, stresses, and (optionally) the Hessian.

        Stress is obtained by introducing a symbolic strain tensor
        ``eps`` of shape ``(B, 3, 3)`` and using ``dE/d_eps`` from autograd,
        scaled by ``1/V`` and converted to GPa.

        The input ``g`` is **not mutated**: this method shallow-clones it
        before attaching ``lattice`` / ``pbc_offshift`` / ``pos`` so the
        same graph can be reused across multiple ``Potential.forward``
        calls and shared between callers (e.g. two ``Potential`` instances)
        without re-conversion.

        Args:
            g: PyG graph (or ``Batch``) with the following attributes set by
                the matgl converters. Read-only -- a shallow clone is taken
                internally so the caller's ``g`` is untouched.

                * ``g.frac_coords`` -- fractional coordinates, shape (N, 3);
                * ``g.edge_index`` -- COO connectivity, shape (2, E);
                * ``g.pbc_offset`` -- integer PBC image offsets per edge,
                  shape (E, 3);
                * ``g.batch`` -- per-node graph index when batched.
            lat: lattice in Cartesian frame, shape ``(B, 3, 3)`` (or
                ``(3, 3)`` for a single graph). Units of A.
            state_attr: optional global state features, shape
                ``(B, dim_state)``.
            l_g: optional line graph used by three-body interactions
                (M3GNet/CHGNet/SO3Net). May be ``None`` for two-body
                models such as TensorNet.
            total_charge: optional per-graph total charge, shape ``(B,)``,
                consumed only when ``calc_charge=True``.
            ext_pot: optional per-atom external potential, shape ``(N,)``,
                consumed only when ``calc_charge=True``.

        Returns:
            A tuple whose contents depend on the active ``calc_*`` flags.
            The base form is ``(energies, forces, stresses, hessian)`` --
            quantities not requested are populated with a singleton
            ``torch.zeros(1)`` placeholder rather than being omitted.
            Optional site-wise quantities are appended in fixed order:

            * ``calc_magmom and calc_charge`` -> ``(..., charges, magmoms)``;
            * ``calc_magmom`` -> ``(..., magmoms)``;
            * ``calc_charge`` -> ``(..., charges)``;
            * ``debug_mode`` -> ``(energies, dE/dpos, dE/deps)``
              (3-tuple, bypasses the standard layout).

            Shapes: ``energies (B,)``, ``forces (N, 3)``, ``stresses (3*B, 3)``
            in GPa with compressive-negative sign, ``hessian (3*N, 3*N)``.
        """
        if lat is None:
            raise ValueError(
                "Potential.forward requires a `lat` tensor (lattice in Cartesian frame, "
                "shape (3, 3) or (B, 3, 3)). The PES path always needs a lattice to "
                "compute pbc-aware positions and stress; pass an explicit identity or "
                "the structure's lattice."
            )

        # Shallow-clone the input graph so the in-place attribute assignments below
        # (``g.lattice`` / ``g.pbc_offshift`` / ``g.pos``) do not leak back to the
        # caller. PyG's ``Data.to(device)`` migrates tensors in place and returns
        # ``self``, so it cannot be relied on for isolation. ``copy.copy`` produces
        # a new ``Data`` with its own attribute namespace but shares tensor refs,
        # so this is O(1) — the migration cost (if any) is in the ``.to(device)``
        # call that follows. Skipping the device migration when already co-located
        # also saves a per-step ``.apply(func)`` walk on the ASE/MD hot path.
        device = self.data_mean.device
        g = copy.copy(g)
        if lat.device != device:
            g = g.to(device)
            lat = lat.to(device)
            if isinstance(state_attr, torch.Tensor):
                state_attr = state_attr.to(device)
            if l_g is not None:
                l_g = l_g.to(device)

        batch_size = g.num_graphs if hasattr(g, "num_graphs") else 1
        # st (strain) for stress calculations
        st = lat.new_zeros([batch_size, 3, 3])
        if self.calc_stresses:
            st.requires_grad_(True)

        lattice = lat @ (self._eye3 + st)

        # Attach lattice to edges
        if isinstance(g, Batch):
            edge_batch = g.batch[g.edge_index[0]]  # (num_edges,)
            node_batch = g.batch  # (num_nodes,)
        else:
            # If not batched
            edge_batch = torch.zeros(g.edge_index.size(1), dtype=torch.long, device=lat.device)
            node_batch = torch.zeros(g.num_nodes, dtype=torch.long, device=lat.device)

        g.lattice = lattice[edge_batch]  # (num_edges, 3, 3)

        g.pbc_offshift = (g.pbc_offset.unsqueeze(dim=-1) * g.lattice).sum(dim=1)

        lattice_per_node = lattice[node_batch]

        g.pos = (g.frac_coords.unsqueeze(-1) * lattice_per_node).sum(dim=1)

        if self.calc_forces:
            g.pos.requires_grad_(True)

        # If no derivatives are requested, suppress autograd graph construction entirely.
        # `calc_stresses` already required `st.requires_grad_(True)` above, so we only
        # enter the no_grad context when forces/stresses/hessian are all off.
        needs_autograd = self.calc_forces or self.calc_stresses or self.calc_hessian
        autograd_ctx = nullcontext() if needs_autograd else torch.no_grad()
        with autograd_ctx:
            total_energies = (
                self.model(
                    g=g,
                    state_attr=state_attr,
                    l_g=l_g,
                    total_charge=total_charge,
                    ext_pot=ext_pot,
                )
                if self.calc_charge
                else self.model(g=g, state_attr=state_attr, l_g=l_g)
            )
            total_energies = self.data_std * total_energies + self.data_mean

            if self.calc_repuls:
                total_energies += self.repuls(self.model.element_types, g)  # type: ignore[attr-defined]

            if self.element_refs is not None:
                property_offset = torch.squeeze(self.element_refs(g))
                total_energies += property_offset

        forces = torch.zeros(1)
        stresses = torch.zeros(1)
        hessian = torch.zeros(1)

        grad_vars = [g.pos, st] if self.calc_stresses else [g.pos]

        # create_graph is only needed if we'll backprop through the gradient itself —
        # i.e. during training (force-loss double-backward) or for Hessian. At inference
        # this roughly halves autograd memory and saves wall time. Stress is captured in
        # the same grad() call as forces, so it does not require retain_graph on its own.
        needs_double_back = self.training or self.calc_hessian

        grads: tuple[torch.Tensor, ...] | None = None
        if self.calc_forces:
            grads = grad(
                total_energies,
                grad_vars,
                grad_outputs=torch.ones_like(total_energies),
                create_graph=needs_double_back,
                retain_graph=needs_double_back,
            )
            forces = -grads[0]

        if self.calc_hessian and grads is not None:
            r = grads[0].view(-1)
            s = r.size(0)
            hessian = total_energies.new_zeros((s, s))
            for iatom in range(s):
                tmp = grad([r[iatom]], g.pos, retain_graph=iatom < s - 1)[0]
                if tmp is not None:
                    hessian[iatom] = tmp.view(-1)

        if self.calc_stresses and grads is not None:
            volume = (
                torch.abs(torch.det(lattice.float())).half()
                if matgl.float_th == torch.float16
                else torch.abs(torch.det(lattice))
            )
            # grads[1] is dE/dε with shape either (3, 3) [unbatched] or (B, 3, 3) [batched].
            # Stress = (1/V) * dE/dε in eV/Å³, converted to GPa.
            sts = grads[1]
            if sts.dim() == 3:
                scaled = sts * (EV_PER_ANG3_TO_GPA / volume).view(-1, 1, 1)
                stresses = scaled.reshape(-1, 3)
            else:
                stresses = sts * (EV_PER_ANG3_TO_GPA / volume)

        if self.debug_mode and grads is not None:
            return total_energies, grads[0], grads[1]

        if self.calc_magmom:
            if self.calc_charge:
                return total_energies, forces, stresses, hessian, g.charge, g.magmom
            return total_energies, forces, stresses, hessian, g.magmom

        if self.calc_charge:
            return total_energies, forces, stresses, hessian, g.charge

        return total_energies, forces, stresses, hessian
