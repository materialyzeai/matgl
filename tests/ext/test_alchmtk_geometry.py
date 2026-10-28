"""Precision of the geometry the alchmtk (nvalchemiops) neighbour-list path hands to the graph.

``nvalchemiops`` is GPU-only and usually absent in CI, so a brute-force stand-in for
``nvalchemiops.neighborlist`` is injected and ``matgl.ext._alchmtk`` is re-imported against it. The
stand-in only replaces the neighbour search; everything matgl does around it is exercised for real.
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest
import torch
from ase.build import bulk, fcc111
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.optimization.neighbors import find_points_in_spheres

import matgl
from matgl.apps.pes import Potential
from matgl.models import TensorNet

SEARCH_DTYPES: list[torch.dtype] = []


class _NeighborOverflowError(Exception):
    pass


def _fake_neighbor_list(positions, cell, pbc, cutoff, max_neighbors, return_neighbor_list, method):
    SEARCH_DTYPES.append(positions.dtype)
    pos = positions.detach().cpu().double().numpy()
    if cell is None:
        lattice = np.eye(3) * 1000.0
        pbc_arr = np.zeros(3, dtype=np.int64)
    else:
        lattice = cell.detach().cpu().double().numpy().reshape(3, 3)
        pbc_arr = pbc.detach().cpu().numpy().astype(np.int64)
    src, dst, images, dist = find_points_in_spheres(pos, pos, r=cutoff, pbc=pbc_arr, lattice=lattice, tol=1e-8)
    keep = (src != dst) | (dist > 1e-8)
    nblist = torch.as_tensor(np.stack([src[keep], dst[keep]]), dtype=torch.int32)
    neighbor_ptr = torch.zeros(len(pos) + 1, dtype=torch.int32)
    if pbc is None:
        return nblist, neighbor_ptr
    return nblist, neighbor_ptr, torch.as_tensor(images[keep], dtype=torch.int32)


@pytest.fixture
def alchmtk(monkeypatch):
    """Install the stand-in ``nvalchemiops`` and route the converters through the alchmtk branch."""
    neighborlist = types.ModuleType("nvalchemiops.neighborlist")
    neighborlist.neighbor_list = _fake_neighbor_list
    neighborlist.estimate_max_neighbors = lambda cutoff, density, safety_factor=1.0: 64
    neighbor_utils = types.ModuleType("nvalchemiops.neighborlist.neighbor_utils")
    neighbor_utils.NeighborOverflowError = _NeighborOverflowError
    monkeypatch.setitem(sys.modules, "nvalchemiops", types.ModuleType("nvalchemiops"))
    monkeypatch.setitem(sys.modules, "nvalchemiops.neighborlist", neighborlist)
    monkeypatch.setitem(sys.modules, "nvalchemiops.neighborlist.neighbor_utils", neighbor_utils)
    monkeypatch.delitem(sys.modules, "matgl.ext._alchmtk", raising=False)
    module = importlib.import_module("matgl.ext._alchmtk")
    monkeypatch.setattr(module, "_default_device", torch.device("cpu"))

    import matgl.ext.ase as ext_ase
    import matgl.ext.pymatgen as ext_pmg

    monkeypatch.setattr(ext_pmg, "_alchmtk_available", True)
    monkeypatch.setattr(ext_pmg, "neighbor_list_from_structure", module.neighbor_list_from_structure, raising=False)
    monkeypatch.setattr(ext_pmg, "neighbor_list_from_molecule", module.neighbor_list_from_molecule, raising=False)
    monkeypatch.setattr(ext_ase, "_alchmtk_available", True)
    monkeypatch.setattr(ext_ase, "neighbor_list_from_ase", module.neighbor_list_from_ase, raising=False)
    SEARCH_DTYPES.clear()
    yield module
    sys.modules.pop("matgl.ext._alchmtk", None)


@pytest.fixture
def float64_mode():
    matgl.set_default_dtype("float", 64)
    try:
        yield
    finally:
        matgl.set_default_dtype("float", 32)


def _triclinic():
    lattice = Lattice.from_parameters(4.1, 4.7, 5.3, 81.3, 97.9, 107.1)
    return Structure(lattice, ["Mo", "S", "S"], [[0.013, 0.021, 0.037], [0.333, 0.671, 0.249], [0.71, 0.29, 0.77]])


def test_neighbor_lists_return_float64_geometry(alchmtk):
    structure = _triclinic()
    *_, positions = alchmtk.neighbor_list_from_structure(structure, cutoff=4.0)
    assert positions.dtype == torch.float64
    np.testing.assert_allclose(positions.numpy(), structure.cart_coords, rtol=0, atol=1e-12)

    mol = Molecule(["C", "H", "H"], [[0.0, 0.0, 0.0], [1.0912345678, 0.0, 0.0], [0.0, 1.0987654321, 0.0]])
    *_, positions = alchmtk.neighbor_list_from_molecule(mol, cutoff=3.0)
    assert positions.dtype == torch.float64
    assert np.array_equal(positions.numpy(), mol.cart_coords)

    atoms = bulk("NaCl", "rocksalt", a=5.64).repeat((2, 2, 2))
    src, _, distances, shifts, positions, _ = alchmtk.neighbor_list_from_ase(atoms, cutoff=4.0)
    assert positions.dtype == torch.float64
    assert distances.dtype == torch.float64
    assert np.array_equal(positions.numpy(), atoms.get_positions())
    # The neighbour search itself still runs in float32.
    assert set(SEARCH_DTYPES) == {torch.float32}
    assert len(src) == len(shifts)


def test_structure2graph_float64(alchmtk, float64_mode):
    from matgl.ext.pymatgen import Structure2Graph

    structure = _triclinic()
    graph, lat, _ = Structure2Graph(element_types=("S", "Mo"), cutoff=4.0).get_graph(structure)
    assert graph.frac_coords.dtype == torch.float64
    assert np.array_equal(graph.frac_coords.numpy(), structure.frac_coords)
    assert np.array_equal(lat[0].numpy(), structure.lattice.matrix)


def test_molecule2graph_float64(alchmtk, float64_mode):
    from matgl.ext.pymatgen import Molecule2Graph

    mol = Molecule(["C", "H", "H"], [[0.0, 0.0, 0.0], [1.0912345678, 0.0, 0.0], [0.0, 1.0987654321, 0.0]])
    graph, _, _ = Molecule2Graph(element_types=("H", "C"), cutoff=3.0).get_graph(mol)
    assert np.array_equal(graph.frac_coords.numpy(), mol.cart_coords)


@pytest.mark.parametrize(
    "make_atoms",
    [lambda: bulk("Cu", "fcc", a=3.61, cubic=True), lambda: fcc111("Cu", (2, 2, 3), vacuum=6.0)],
    ids=["bulk", "slab"],
)
def test_atoms2graph_float64(alchmtk, float64_mode, make_atoms):
    from matgl.ext.ase import Atoms2Graph

    atoms = make_atoms()
    atoms.rattle(0.05, seed=1)
    graph, lat, _ = Atoms2Graph(element_types=("Cu",), cutoff=4.0).get_graph(atoms)
    assert lat.shape == (1, 3, 3)
    np.testing.assert_allclose(graph.frac_coords.numpy(), atoms.get_scaled_positions(wrap=False), rtol=0, atol=1e-14)
    np.testing.assert_allclose((graph.frac_coords @ lat[0]).numpy(), atoms.get_positions(), rtol=0, atol=1e-12)


def test_double_potential_forces_match_finite_differences(alchmtk, float64_mode):
    """With float64 geometry, a ``.double()`` Potential's forces agree with central differences.

    The 1e-5 A step is only ~20 float32 ulps at these coordinates, so float32-rounded geometry fails this.
    """
    from matgl.ext.ase import Atoms2Graph

    torch.manual_seed(0)
    model = TensorNet(element_types=("Na", "Cl"), is_intensive=False, use_warp=False, dtype=torch.float64)
    potential = Potential(model=model).double()
    converter = Atoms2Graph(element_types=("Na", "Cl"), cutoff=5.0)
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    atoms.rattle(0.05, seed=3)

    def energy_and_forces(a):
        graph, lat, state = converter.get_graph(a)
        energy, forces, *_ = potential(graph, lat, torch.as_tensor(state))
        return float(energy), forces.detach().numpy()

    _, forces = energy_and_forces(atoms)
    step = 1e-5
    for atom_idx, axis in [(0, 0), (1, 2)]:
        plus, minus = atoms.copy(), atoms.copy()
        plus.positions[atom_idx, axis] += step
        minus.positions[atom_idx, axis] -= step
        fd = -(energy_and_forces(plus)[0] - energy_and_forces(minus)[0]) / (2 * step)
        assert forces[atom_idx, axis] == pytest.approx(fd, rel=1e-5, abs=1e-10)
