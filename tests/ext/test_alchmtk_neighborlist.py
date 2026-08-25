"""Tests for the ``matgl.ext._alchmtk`` neighbor-list helpers."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("nvalchemiops", reason="nvalchemi-toolkit-ops required")

from ase.build import bulk

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the nvalchemiops path needs CUDA")


def test_neighbor_list_from_ase_computes_distances_on_periodic_cells():
    """``compute_distances=True`` (the default) must work under PBC.

    ``_compute_distances`` did ``unit_shifts @ cell`` with the integer shift
    matrix the neighbor-list kernel returns, and integer x float matmul
    raises ``RuntimeError: expected mat1 and mat2 to have the same dtype``.
    Non-periodic structures dodge the path (``unit_shifts is None``), which
    is how it survived smoke tests. Beyond surviving, the values must match
    a float64 recomputation from the returned tensors.
    """
    from matgl.ext._alchmtk import neighbor_list_from_ase

    atoms = bulk("NaCl", "rocksalt", a=5.64).repeat((3, 3, 3))
    src, dst, dist, shifts, pos, _ = neighbor_list_from_ase(atoms, cutoff=5.0, compute_distances=True, device="cuda")

    assert dist is not None
    assert dist.shape[0] == src.shape[0]
    cell = torch.as_tensor(atoms.get_cell().array, dtype=torch.float64, device=pos.device)
    ref = torch.linalg.norm(
        pos[dst.long()].double() + shifts.double() @ cell - pos[src.long()].double(),
        dim=1,
    )
    assert torch.allclose(dist.double(), ref, atol=1e-4)
    assert float(dist.max()) <= 5.0 + 1e-4
