# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""This module implements the interface to the neighbor list in the nvalchemi-toolkit-ops package."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from nvalchemiops.neighborlist import estimate_max_neighbors, neighbor_list
from nvalchemiops.neighborlist.neighbor_utils import NeighborOverflowError

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core.structure import Molecule, Structure


_default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def neighbor_list_from_structure(
    structure: Structure,
    cutoff: float,
    compute_distances: bool = True,
    density_guess: float = 0.3,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Get the neighbor list from the PyMatGen Structure using the nvalchemi-toolkit-ops package."""
    device = torch.device(device) if device is not None else _default_device
    pbc = torch.tensor([True, True, True], device=device)
    cell = torch.as_tensor(structure.lattice.matrix.copy(), dtype=torch.float32, device=device)
    frac_coords = torch.as_tensor(structure.frac_coords, dtype=torch.float32, device=device)
    positions = frac_coords @ cell

    nblist, _, unit_shifts, _ = _safe_nl(
        positions=positions,
        cutoff=cutoff,
        cell=cell,
        pbc=pbc,
        density_guess=density_guess,
    )

    src_id, dst_id = nblist.unbind(dim=0)
    if compute_distances:
        distances = _compute_distances(
            src_id=src_id,
            dst_id=dst_id,
            positions=positions,
            cell=cell,
            unit_shifts=unit_shifts,
        )
    else:
        distances = None
    # unit_shifts is always valid for periodic structures (pbc=True)
    assert unit_shifts is not None
    return (src_id, dst_id, distances, unit_shifts, positions)


def neighbor_list_from_molecule(
    molecule: Molecule,
    cutoff: float,
    compute_distances: bool = True,
    density_guess: float = 0.3,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Get the neighbor list from the PyMatGen Structure using the nvalchemi-toolkit-ops package."""
    device = torch.device(device) if device is not None else _default_device
    positions = torch.as_tensor(molecule.cart_coords, dtype=torch.float32, device=device)

    nblist, _, _, _ = _safe_nl(
        positions=positions,
        cutoff=cutoff,
        cell=None,
        pbc=None,
        density_guess=density_guess,
    )

    src_id, dst_id = nblist.unbind(dim=0)
    if compute_distances:
        distances = _compute_distances(
            src_id=src_id,
            dst_id=dst_id,
            positions=positions,
            cell=None,
            unit_shifts=None,
        )
    else:
        distances = None

    return (src_id, dst_id, distances, positions)


def neighbor_list_from_ase(
    atoms: Atoms,
    cutoff: float,
    compute_distances: bool = True,
    density_guess: float = 0.3,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, int]:
    """Get the neighbor list from the ASE Atoms using the nvalchemi-toolkit-ops package."""
    device = torch.device(device) if device is not None else _default_device
    positions = torch.as_tensor(atoms.get_positions(), dtype=torch.float32, device=device)
    if atoms.get_pbc().any():
        cell = torch.as_tensor(atoms.get_cell(), dtype=torch.float32, device=device)
        pbc = torch.as_tensor(atoms.get_pbc(), dtype=torch.bool, device=device)
    else:
        cell = None
        pbc = None

    nblist, _, unit_shifts, max_neighbors = _safe_nl(
        positions=positions,
        cutoff=cutoff,
        cell=cell,
        pbc=pbc,
        density_guess=density_guess,
    )
    src_id, dst_id = nblist.unbind(dim=0)
    if compute_distances:
        distances = _compute_distances(
            src_id=src_id,
            dst_id=dst_id,
            positions=positions,
            cell=cell,
            unit_shifts=unit_shifts,
        )
    else:
        distances = None
    if unit_shifts is None:
        unit_shifts = torch.zeros(src_id.shape[0], 3, dtype=torch.float32, device=device)
    return (src_id, dst_id, distances, unit_shifts, positions, max_neighbors)


def _safe_nl(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
    density_guess: float = 0.3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
    """Get the neighbor list using nvalchemi-toolkit-ops with automatic buffer sizing."""
    method = "cell_list" if pbc is not None else "naive"

    # Accept either a density (float) or a pre-computed max_neighbors (int)
    if isinstance(density_guess, int):
        max_neighbors = density_guess
    else:
        max_neighbors = estimate_max_neighbors(cutoff, density_guess, safety_factor=1.0)

    # Safety cap based on density: max_neighbors at 5.0 atoms/A^3 is sufficiently
    # high that exceeding it means the structure is genuinely collapsed.
    max_density = 5.0
    max_max_neighbors = estimate_max_neighbors(cutoff, max_density, safety_factor=1.0)

    while True:
        try:
            ret = neighbor_list(
                positions=positions,
                cell=cell,
                pbc=pbc,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                return_neighbor_list=True,
                method=method,
            )
            break
        except NeighborOverflowError as err:
            # Grow buffer by 1.5x, rounded up to multiple of 16 for alignment
            max_neighbors = ((int(max_neighbors * 1.5) + 15) // 16) * 16
            if max_neighbors > max_max_neighbors:
                raise ValueError(
                    f"Unable to get neighbor list. Structure is too dense. "
                    f"max_neighbors ({max_neighbors}) exceeds cap from "
                    f"density {max_density} atoms/A^3."
                ) from err

    if pbc is not None:
        nblist, neighbor_ptr, unit_shifts = ret
    else:
        nblist, neighbor_ptr = ret
        unit_shifts = None

    return nblist, neighbor_ptr, unit_shifts, max_neighbors


def _compute_distances(
    src_id: torch.Tensor,
    dst_id: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor | None = None,
    unit_shifts: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the distances between the source and destination atoms."""
    vectors = positions[dst_id] - positions[src_id]
    if cell is not None and unit_shifts is not None:
        vectors += unit_shifts @ cell
    return torch.linalg.norm(vectors, dim=1)


def _wrap_positions(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
) -> torch.Tensor:
    """Wrap the positions to be in the unit cell."""
    cell_inv = torch.linalg.inv(cell)
    positions_frac = positions @ cell_inv
    positions_frac[:, pbc] %= 1.0
    return positions_frac @ cell
