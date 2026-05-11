"""Implementation of Interatomic Potentials."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import dgl
import torch
from torch import nn
from torch.autograd import grad

import matgl
from matgl.layers._atom_ref_dgl import AtomRef
from matgl.layers._zbl_dgl import NuclearRepulsion
from matgl.utils.io import IOMixIn

if TYPE_CHECKING:
    import dgl
    import numpy as np

# 1 eV/Å³ = 160.21766208 GPa. Stress is autograd of energy w.r.t. strain (eV)
# divided by volume (Å³), giving eV/Å³; multiply by this constant for GPa.
EV_PER_ANG3_TO_GPA = 160.21766208


class Potential(nn.Module, IOMixIn):
    """A class representing an interatomic potential."""

    __version__ = 3

    # Class-level annotations narrow ``nn.Module.__getattr__``'s ``Tensor | Module``
    # return type to ``Tensor`` for these registered buffers, so mypy accepts
    # ``self._eye3 + st`` below.
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
        """
        super().__init__()
        self.save_args(locals())
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
            cutoff: float = self.model.cutoff  # type: ignore[assignment]
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
        g: dgl.DGLGraph,
        lat: torch.Tensor,
        state_attr: torch.Tensor | None = None,
        l_g: dgl.DGLGraph | None = None,
        total_charge: torch.Tensor | None = None,
        ext_pot: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Compute energies, forces, stresses, and (optionally) the Hessian.

        Args:
            g: DGL graph
            lat: lattice
            state_attr: State attrs
            l_g: Line graph
            total_charge: total charge of the system
            ext_pot: external potential (Natoms).

        Returns:
            (energies, forces, stresses, hessian) or (energies, forces, stresses, hessian, site-wise properties)
        """
        # st (strain) for stress calculations
        st = lat.new_zeros([g.batch_size, 3, 3])
        if self.calc_stresses:
            st.requires_grad_(True)
        lattice = lat @ (self._eye3 + st)
        g.edata["lattice"] = torch.repeat_interleave(lattice, g.batch_num_edges(), dim=0)
        g.edata["pbc_offshift"] = (g.edata["pbc_offset"].unsqueeze(dim=-1) * g.edata["lattice"]).sum(dim=1)
        g.ndata["pos"] = (
            g.ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(lattice, g.batch_num_nodes(), dim=0)
        ).sum(dim=1)
        if self.calc_forces:
            g.ndata["pos"].requires_grad_(True)

        # If no derivatives are requested, suppress autograd graph construction entirely.
        # ``calc_stresses`` already required ``st.requires_grad_(True)`` above, so we only
        # enter the no_grad context when forces/stresses/hessian are all off.
        needs_autograd = self.calc_forces or self.calc_stresses or self.calc_hessian
        autograd_ctx = nullcontext() if needs_autograd else torch.no_grad()
        with autograd_ctx:
            total_energies = (
                self.model(
                    g=g,
                    state_attr=state_attr,
                    l_g=l_g,
                    lat=lat,
                    total_charge=total_charge,
                    ext_pot=ext_pot,
                    lattice=lat,
                )
                if self.calc_charge is True
                else self.model(g=g, l_g=l_g, state_attr=state_attr)
            )

            total_energies = self.data_std * total_energies + self.data_mean

            if self.calc_repuls:
                total_energies += self.repuls(self.model.element_types, g)

            if self.element_refs is not None:
                property_offset = torch.squeeze(self.element_refs(g))
                total_energies += property_offset

        forces = torch.zeros(1)
        stresses = torch.zeros(1)
        hessian = torch.zeros(1)

        grad_vars = [g.ndata["pos"], st] if self.calc_stresses else [g.ndata["pos"]]

        # create_graph is only needed if we'll backprop through the gradient itself —
        # i.e. during training (force-loss double-backward) or for Hessian. At inference
        # this roughly halves autograd memory and saves wall time. Stress is captured in
        # the same grad() call as forces, so it does not require retain_graph on its own.
        needs_double_back = self.training or self.calc_hessian

        if self.calc_forces:
            grads = grad(
                total_energies,
                grad_vars,
                grad_outputs=torch.ones_like(total_energies),
                create_graph=needs_double_back,
                retain_graph=needs_double_back,
            )
            forces = -grads[0]

        if self.calc_hessian:
            r = grads[0].view(-1)
            s = r.size(0)
            hessian = total_energies.new_zeros((s, s))
            for iatom in range(s):
                tmp = grad([r[iatom]], g.ndata["pos"], retain_graph=iatom < s - 1)[0]
                if tmp is not None:
                    hessian[iatom] = tmp.view(-1)

        if self.calc_stresses:
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

        if self.debug_mode:
            return total_energies, grads[0], grads[1]

        if self.calc_magmom:
            if self.calc_charge:
                return total_energies, forces, stresses, hessian, g.ndata["charge"], g.ndata["magmom"]
            return total_energies, forces, stresses, hessian, g.ndata["magmom"]

        if self.calc_charge:
            return total_energies, forces, stresses, hessian, g.ndata["charge"]

        return total_energies, forces, stresses, hessian
