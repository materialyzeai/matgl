"""TorchSim model interface for MatGL interatomic potentials.

This optional extension adapts a :class:`matgl.apps.pes.Potential` to
TorchSim's batched ``ModelInterface``. Graph connectivity is built directly
from ``SimState`` tensors with TorchSim's neighbor-list implementation, so no
ASE or pymatgen round trip is required inside a simulation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from ase.data import atomic_numbers
from torch_geometric.data import Batch

from matgl.utils.constants import EV_PER_ANG3_TO_GPA

try:
    from torch_sim.models.interface import ModelInterface
    from torch_sim.neighbors import torchsim_nl
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "matgl.ext.torch_sim requires the optional 'torch-sim' dependency and Python 3.12 or newer. "
        "Install it with: pip install 'matgl[torch-sim]'"
    ) from exc

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch_sim.state import SimState

    from matgl.apps.pes import Potential

__all__ = ["MatGLModel"]


class MatGLModel(ModelInterface):
    """Expose a MatGL potential through TorchSim's native model interface.

    Args:
        potential: MatGL potential to evaluate. The adapter configures it for
            inference and synchronizes its force/stress flags with this interface.
        device: Compute device. Defaults to the potential's current device.
        dtype: Floating-point dtype. Defaults to the potential's current dtype.
        compute_forces: Whether to return forces.
        compute_stress: Whether to return stress in eV/Angstrom^3.
        state_attr: Optional fixed MatGL graph state attributes. A ``state_attr``
            system extra or forward keyword takes precedence.
        neighbor_list_fn: TorchSim-compatible neighbor-list callable. Defaults
            to :func:`torch_sim.neighbors.torchsim_nl`.

    Notes:
        TorchSim uses column-vector cells, while MatGL uses row-vector cells;
        this adapter performs that conversion. MatGL stresses are converted
        from GPa to TorchSim's metal-unit convention of eV/Angstrom^3.
    """

    _memory_scales_with = "n_atoms_x_density"
    _default_state_attr: torch.Tensor | None
    _supported_z: torch.Tensor
    _z_to_type: torch.Tensor

    def __init__(
        self,
        potential: Potential,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        compute_forces: bool = True,
        compute_stress: bool = True,
        state_attr: torch.Tensor | None = None,
        neighbor_list_fn: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = torchsim_nl,
    ) -> None:
        """Initialize the TorchSim adapter."""
        super().__init__()
        if getattr(potential.model, "is_intensive", False):
            raise ValueError("MatGLModel requires an extensive total-energy model (is_intensive=False)")
        if not hasattr(potential.model, "cutoff") or not hasattr(potential.model, "element_types"):
            raise TypeError("potential.model must define cutoff and element_types")

        self.potential = potential
        self.neighbor_list_fn = neighbor_list_fn
        self.cutoff = float(potential.model.cutoff)
        self.element_types = tuple(potential.model.element_types)

        inferred_device, inferred_dtype = self._potential_device_dtype()
        self._device = torch.device(device) if device is not None else inferred_device
        self._dtype = dtype or inferred_dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress

        self.potential.eval()
        self.potential.calc_hessian = False
        self._sync_potential_derivatives()
        self.register_buffer("_default_state_attr", state_attr, persistent=False)

        supported_z = torch.tensor([atomic_numbers[symbol] for symbol in self.element_types], dtype=torch.long)
        max_z = int(supported_z.max())
        z_to_type = torch.full((max_z + 1,), -1, dtype=torch.long)
        z_to_type[supported_z] = torch.arange(len(self.element_types))
        self.register_buffer("_z_to_type", z_to_type, persistent=False)
        self.register_buffer("_supported_z", supported_z, persistent=False)
        self.to(device=self._device, dtype=self._dtype)

    def _potential_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        """Return the first floating parameter/buffer device and dtype."""
        for tensor in (*self.potential.parameters(), *self.potential.buffers()):
            if tensor.is_floating_point():
                return tensor.device, tensor.dtype
        return torch.device("cpu"), torch.get_default_dtype()

    def _sync_potential_derivatives(self) -> None:
        """Keep MatGL derivative switches consistent with TorchSim outputs."""
        self.potential.calc_forces = self._compute_forces or self._compute_stress
        self.potential.calc_stresses = self._compute_stress

    @property
    def device(self) -> torch.device:
        """Device used for model evaluation."""
        return self._device

    @device.setter
    def device(self, device: torch.device) -> None:
        self._device = torch.device(device)
        self.to(device=self._device)

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point dtype used for model evaluation."""
        return self._dtype

    @dtype.setter
    def dtype(self, dtype: torch.dtype) -> None:
        self._dtype = dtype
        self.to(dtype=dtype)

    @property
    def compute_forces(self) -> bool:
        """Whether calls return per-atom forces."""
        return self._compute_forces

    @compute_forces.setter
    def compute_forces(self, compute_forces: bool) -> None:
        self._compute_forces = compute_forces
        self._sync_potential_derivatives()

    @property
    def compute_stress(self) -> bool:
        """Whether calls return per-system stresses."""
        return self._compute_stress

    @compute_stress.setter
    def compute_stress(self, compute_stress: bool) -> None:
        self._compute_stress = compute_stress
        self._sync_potential_derivatives()

    def _validate_state(self, state: SimState) -> None:
        if state.device != self.device:
            raise ValueError(f"SimState device {state.device} does not match model device {self.device}")
        if state.dtype != self.dtype:
            raise TypeError(f"SimState dtype {state.dtype} does not match model dtype {self.dtype}")

        numbers = state.atomic_numbers.to(torch.long)
        unknown = numbers[(numbers >= self._z_to_type.numel()) | ~torch.isin(numbers, self._supported_z)]
        if unknown.numel():
            unique = sorted(set(unknown.detach().cpu().tolist()))
            raise ValueError(f"SimState contains unsupported atomic numbers: {unique}")

    def _state_to_graph(self, state: SimState) -> tuple[Batch, torch.Tensor]:
        """Convert one batched TorchSim state to a MatGL PyG batch."""
        self._validate_state(state)
        system_idx = state.system_idx.to(torch.long)
        row_cell = state.row_vector_cell
        pbc = state.pbc.reshape(-1, 3).expand(state.n_systems, -1)
        periodic = pbc.any(dim=1)
        if bool(periodic.any()):
            volumes = torch.linalg.det(row_cell[periodic]).abs()
            if bool((volumes <= torch.finfo(self.dtype).eps).any()):
                raise ValueError("periodic systems require non-singular cells")
        identity = torch.eye(3, dtype=self.dtype, device=self.device)
        lattice = torch.where(periodic[:, None, None], row_cell, identity)

        edge_index, _, pbc_offset = self.neighbor_list_fn(
            positions=state.positions,
            cell=row_cell,
            pbc=state.pbc,
            cutoff=self.cutoff,
            system_idx=system_idx,
        )
        atom_lattice = lattice[system_idx]
        frac_coords = torch.linalg.solve(atom_lattice.mT, state.positions.unsqueeze(-1)).squeeze(-1)
        numbers = state.atomic_numbers.to(torch.long)
        node_type = self._z_to_type[numbers]
        graph = Batch(
            node_type=node_type,
            frac_coords=frac_coords,
            edge_index=edge_index.to(torch.long),
            pbc_offset=pbc_offset,
            batch=system_idx,
            num_nodes=state.n_atoms,
            pbc=pbc,
        )
        return graph, lattice

    @staticmethod
    def _state_extra(state: SimState, key: str) -> torch.Tensor | None:
        """Read an optional TorchSim atom/system extra without assuming its presence."""
        return getattr(state, key) if state.has_extras(key) else None

    @torch.inference_mode(False)
    @torch.enable_grad()
    def forward(self, state: SimState, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Calculate batched energies, forces, stresses, and optional site properties."""
        graph, lattice = self._state_to_graph(state)
        state_attr = kwargs.get("state_attr", self._state_extra(state, "state_attr"))
        if state_attr is None:
            state_attr = self._default_state_attr
        if state_attr is not None:
            state_attr = state_attr.to(device=self.device)
            if state_attr.is_floating_point():
                state_attr = state_attr.to(dtype=self.dtype)

        total_charge = kwargs.get("total_charge", self._state_extra(state, "total_charge"))
        if total_charge is None:
            total_charge = torch.zeros(state.n_systems, dtype=self.dtype, device=self.device)
        graph.total_charge = torch.as_tensor(total_charge, dtype=self.dtype, device=self.device)
        ext_pot = kwargs.get("ext_pot", self._state_extra(state, "external_potential"))
        if ext_pot is not None:
            ext_pot = torch.as_tensor(ext_pot, dtype=self.dtype, device=self.device)
        raw = self.potential(graph, lattice, state_attr, total_charge=graph.total_charge, ext_pot=ext_pot)

        result = {"energy": raw[0].reshape(state.n_systems).detach()}
        if self.compute_forces:
            result["forces"] = raw[1].reshape(state.n_atoms, 3).detach()
        if self.compute_stress:
            stress = raw[2].reshape(state.n_systems, 3, 3).div(EV_PER_ANG3_TO_GPA)
            stress = torch.where(graph.pbc.any(dim=1)[:, None, None], stress, torch.zeros_like(stress))
            result["stress"] = stress.detach()

        if self.potential.calc_charge:
            result["partial_charges"] = raw[4].reshape(state.n_atoms).detach()
            if self.potential.calc_magmom:
                result["magnetic_moments"] = raw[5].reshape(state.n_atoms).detach()
        elif self.potential.calc_magmom:
            result["magnetic_moments"] = raw[4].reshape(state.n_atoms).detach()
        return result
