"""Implementation of Interatomic Potentials."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.autograd import grad
from torch_geometric.data import Batch, Data

import matgl
from matgl.layers._atom_ref_pyg import AtomRefPyG
from matgl.layers._zbl_pyg import NuclearRepulsionPyG
from matgl.utils.io import IOMixIn

if TYPE_CHECKING:
    import numpy as np


class Potential(nn.Module, IOMixIn):
    """A class representing an interatomic potential."""

    __version__ = 3

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
        self.element_refs: AtomRefPyG | None
        self.debug_mode = debug_mode
        self.calc_repuls = calc_repuls

        if calc_repuls:
            cutoff: float = self.model.cutoff  # type: ignore[assignment]
            self.repuls = NuclearRepulsionPyG(cutoff, trainable=zbl_trainable)

        if element_refs is not None:
            if not isinstance(element_refs, torch.Tensor):
                element_refs = torch.tensor(element_refs, dtype=matgl.float_th)
            self.element_refs = AtomRefPyG(property_offset=element_refs)
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

    def forward(
        self,
        g: Data,
        lat: torch.Tensor,
        state_attr: torch.Tensor | np.ndarray | None = None,
        l_g: Data | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Args:
            g: PyG graph
            lat: lattice
            state_attr: State attrs
            l_g: PyG Line graph.

        Returns:
            (energies, forces, stresses, hessian) or (energies, forces, stresses, hessian, site-wise properties)
        """
        device = next(self.parameters()).device
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

        lattice = lat @ (torch.eye(3, device=lat.device) + st)

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

        total_energies = self.model(g=g, state_attr=state_attr, l_g=l_g)
        total_energies = self.data_std * total_energies + self.data_mean

        if self.calc_repuls:
            total_energies += self.repuls(self.model.element_types, g)

        if self.element_refs is not None:
            property_offset = torch.squeeze(self.element_refs(g))
            total_energies += property_offset

        forces = torch.zeros(1)
        stresses = torch.zeros(1)
        hessian = torch.zeros(1)

        grad_vars = [g.pos, st] if self.calc_stresses else [g.pos]

        grads: tuple[torch.Tensor, ...] | None = None
        if self.calc_forces:
            grads = grad(
                total_energies,
                grad_vars,
                grad_outputs=torch.ones_like(total_energies),
                create_graph=True,
                retain_graph=True,
            )
            forces = -grads[0]

        if self.calc_hessian and grads is not None:
            r = grads[0].view(-1)
            s = r.size(0)
            hessian = total_energies.new_zeros((s, s))
            for iatom in range(s):
                tmp = grad([r[iatom]], g.pos, retain_graph=iatom < s)[0]
                if tmp is not None:
                    hessian[iatom] = tmp.view(-1)

        if self.calc_stresses and grads is not None:
            volume = (
                torch.abs(torch.det(lattice.float())).half()
                if matgl.float_th == torch.float16
                else torch.abs(torch.det(lattice))
            )
            sts = grads[1]
            scale = 1.0 / volume * 160.21766208
            sts = [i * j for i, j in zip(sts, scale, strict=False)] if sts.dim() == 3 else [sts * scale]  # type:ignore[assignment]
            stresses = torch.cat(sts)  # type:ignore[call-overload]

        if self.debug_mode and grads is not None:
            return total_energies, grads[0], grads[1]

        if self.calc_magmom:
            return total_energies, forces, stresses, hessian, g.ndata["magmom"]

        return total_energies, forces, stresses, hessian
