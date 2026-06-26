"""Tests for the LAMMPS TorchScript export wrapper.

The wrapper has to give the same energy/forces/stresses as
``Potential.forward`` for any periodic configuration. We exercise this with
a small randomly-initialized TensorNet on a Mo-S supercell, comparing the
wrapper's Cartesian-driven path to the canonical PyG ``Data``-driven path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("LAMMPS export only supports PyG backend", allow_module_level=True)

from pymatgen.core import Lattice, Structure
from pymatgen.optimization.neighbors import find_points_in_spheres

from matgl.apps._pes_pyg import Potential
from matgl.ext._lammps import LAMMPSMatGLModel
from matgl.ext._pymatgen_pyg import Structure2Graph
from matgl.models._m3gnet_pyg import M3GNet
from matgl.models._tensornet_pyg import TensorNet


def _build_lammps_inputs(structure: Structure, element_types: tuple[str, ...], cutoff: float, dtype: torch.dtype):
    """Build the tensor inputs LAMMPS would produce for a single configuration.

    Mirrors the CPU-fallback path in ``Atoms2Graph.get_graph`` — pymatgen's
    ``find_points_in_spheres`` produces (src, dst, image, dist), which is
    exactly the (edge_index, unit_shifts) pair LAMMPS gives us at the C++
    boundary.
    """
    lattice = np.array(structure.lattice.matrix)
    cart = structure.cart_coords
    src, dst, images, dist = find_points_in_spheres(
        cart,
        cart,
        r=float(cutoff),
        pbc=np.array([1, 1, 1], dtype=np.int64),
        lattice=lattice,
        tol=1.0e-8,
    )
    keep = (src != dst) | (dist > 1e-8)
    src = src[keep]
    dst = dst[keep]
    images = images[keep]

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    unit_shifts = torch.tensor(images, dtype=torch.long)
    positions = torch.tensor(cart, dtype=dtype)
    cell = torch.tensor(lattice, dtype=dtype)

    z_per_atom = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
    local_or_ghost = torch.ones(len(structure), dtype=torch.bool)
    return positions, edge_index, unit_shifts, cell, z_per_atom, local_or_ghost


def _build_tensornet_potential(rbf_type: str = "Gaussian") -> tuple[Potential, tuple[str, ...]]:
    torch.manual_seed(0)
    element_types = ("Mo", "S")
    model = TensorNet(
        element_types=element_types,
        is_intensive=False,
        units=16,
        nblocks=1,
        num_rbf=8,
        cutoff=4.0,
        use_warp=False,
        rbf_type=rbf_type,
        # SphericalBessel checkpoints (e.g. TensorNet-MatPES-r2SCAN) use the
        # smooth basis; the non-smooth path relies on sympy-lambdified funcs
        # that don't survive torch.jit.save.
        use_smooth=rbf_type == "SphericalBessel",
    )
    refs = torch.tensor([-1.5, -2.25], dtype=matgl.float_th)
    pot = Potential(
        model=model,
        data_mean=0.0,
        data_std=1.0,
        element_refs=refs,
        calc_forces=True,
        calc_stresses=True,
    )
    pot.eval()
    return pot, element_types


def _build_m3gnet_potential() -> tuple[Potential, tuple[str, ...]]:
    torch.manual_seed(0)
    element_types = ("Mo", "S")
    model = M3GNet(
        element_types=element_types,
        is_intensive=False,
        cutoff=4.0,
        threebody_cutoff=3.0,
        dim_node_embedding=16,
        dim_edge_embedding=16,
        n_blocks=1,
        max_n=3,
        max_l=3,
        units=16,
        rbf_type="SphericalBessel",
        use_smooth=True,
    )
    refs = torch.tensor([-1.5, -2.25], dtype=matgl.float_th)
    pot = Potential(
        model=model,
        data_mean=0.0,
        data_std=1.0,
        element_refs=refs,
        calc_forces=True,
        calc_stresses=True,
    )
    pot.eval()
    return pot, element_types


@pytest.fixture(params=["tensornet_gaussian", "tensornet_sb", "m3gnet"])
def tiny_potential(request):
    """Tiny deterministic Potential, parametrized over supported architectures.

    ``tensornet_sb`` exercises the SphericalBessel-smooth bond expansion path,
    which matches the pretrained ``TensorNet-MatPES-r2SCAN`` checkpoint and
    requires the LAMMPS kernel's pure-tensor SBF adapter.
    """
    if request.param == "tensornet_gaussian":
        return _build_tensornet_potential(rbf_type="Gaussian")
    if request.param == "tensornet_sb":
        return _build_tensornet_potential(rbf_type="SphericalBessel")
    return _build_m3gnet_potential()


@pytest.fixture
def mo_s_supercell():
    return Structure(
        Lattice.cubic(4.5),
        ["Mo", "S", "Mo", "S"],
        [
            [0.00, 0.00, 0.00],
            [0.50, 0.50, 0.50],
            [0.50, 0.00, 0.25],
            [0.00, 0.50, 0.75],
        ],
    )


def _potential_reference(potential, structure, element_types, cutoff, dtype):
    """Run ``Potential.forward`` on a structure via Structure2Graph, in eval()."""
    s2g = Structure2Graph(element_types=element_types, cutoff=cutoff)
    g, lat, _ = s2g.get_graph(structure)
    g.frac_coords = g.frac_coords.to(dtype)
    lat = lat.to(dtype)
    energy, forces, stresses, _ = potential(g, lat, None)
    return energy, forces, stresses


def test_eager_parity_against_potential(tiny_potential, mo_s_supercell):
    """Wrapper's energy/forces match Potential.forward for a periodic crystal."""
    potential, element_types = tiny_potential
    structure = mo_s_supercell
    cutoff = 4.0
    dtype = matgl.float_th

    # Reference path.
    e_ref, f_ref, _s_ref = _potential_reference(potential, structure, element_types, cutoff, dtype)

    # Wrapper path.
    wrapper = LAMMPSMatGLModel(potential=potential, dtype=dtype)
    wrapper.eval()

    pos, eidx, ushifts, cell, z, local = _build_lammps_inputs(structure, element_types, cutoff, dtype)
    out = wrapper(pos, eidx, ushifts, cell, z, local, compute_virials=True)

    e_wrap = out["total_energy_local"]
    f_wrap = out["forces"]

    # Energy parity (scalar).
    assert torch.allclose(e_wrap.detach(), e_ref.detach().reshape_as(e_wrap), atol=1e-5, rtol=1e-5), (
        f"energy mismatch: wrapper={e_wrap.item()} ref={e_ref.item()}"
    )

    # Force parity (per-atom).
    assert torch.allclose(f_wrap.detach(), f_ref.detach(), atol=1e-4, rtol=1e-4), (
        f"max force diff = {(f_wrap.detach() - f_ref.detach()).abs().max().item()}"
    )


def test_ghost_mask_partitions_energy(tiny_potential, mo_s_supercell):
    """Splitting atoms into 'owned' vs 'ghost' must sum back to the full energy."""
    potential, element_types = tiny_potential
    structure = mo_s_supercell
    cutoff = 4.0
    dtype = matgl.float_th

    wrapper = LAMMPSMatGLModel(potential=potential, dtype=dtype)
    wrapper.eval()

    pos, eidx, ushifts, cell, z, _ = _build_lammps_inputs(structure, element_types, cutoff, dtype)

    n = pos.shape[0]
    half = n // 2

    mask_a = torch.zeros(n, dtype=torch.bool)
    mask_a[:half] = True
    mask_b = ~mask_a

    out_a = wrapper(pos, eidx, ushifts, cell, z, mask_a, compute_virials=False)
    out_b = wrapper(pos, eidx, ushifts, cell, z, mask_b, compute_virials=False)
    out_full = wrapper(pos, eidx, ushifts, cell, z, torch.ones(n, dtype=torch.bool), compute_virials=False)

    e_split = out_a["total_energy_local"] + out_b["total_energy_local"]
    # data_mean is added once per call, so the split version adds it twice. We
    # constructed tiny_potential with data_mean=0, so this is consistent.
    assert torch.allclose(e_split.detach(), out_full["total_energy_local"].detach(), atol=1e-5, rtol=1e-5)


def test_torchscript_round_trip(tiny_potential, mo_s_supercell, tmp_path):
    """Scripted module saves, reloads, and matches eager outputs to fp precision."""
    potential, element_types = tiny_potential
    structure = mo_s_supercell
    cutoff = 4.0
    dtype = torch.float32

    wrapper = LAMMPSMatGLModel(potential=potential, dtype=dtype)
    wrapper.eval()

    scripted = torch.jit.script(wrapper)
    artifact = tmp_path / "model.pt"
    scripted.save(str(artifact))
    reloaded = torch.jit.load(str(artifact))

    pos, eidx, ushifts, cell, z, local = _build_lammps_inputs(structure, element_types, cutoff, dtype)

    out_eager = wrapper(pos, eidx, ushifts, cell, z, local, True)
    out_script = reloaded(pos, eidx, ushifts, cell, z, local, True)

    for key in ("total_energy_local", "node_energy", "forces", "virials"):
        diff = (out_eager[key].detach() - out_script[key].detach()).abs().max().item()
        assert diff < 1e-5, f"{key} max abs diff = {diff}"


def test_virials_match_stress_volume(tiny_potential, mo_s_supercell):
    """Wrapper virials = Potential stresses * volume / unit_factor.

    ``Potential`` returns stresses in GPa = (1/V) * eV/A^3 * 160.21766208.
    The wrapper returns the raw strain-grad tensor (no /V, no unit factor).
    So ``virial_wrapper == -stress_potential * V / 160.21766208`` (signs from
    LAMMPS sign convention).
    """
    potential, element_types = tiny_potential
    structure = mo_s_supercell
    cutoff = 4.0
    dtype = matgl.float_th

    _e_ref, _f_ref, s_ref = _potential_reference(potential, structure, element_types, cutoff, dtype)

    wrapper = LAMMPSMatGLModel(potential=potential, dtype=dtype)
    wrapper.eval()

    pos, eidx, ushifts, cell, z, local = _build_lammps_inputs(structure, element_types, cutoff, dtype)
    out = wrapper(pos, eidx, ushifts, cell, z, local, compute_virials=True)

    volume = float(np.linalg.det(structure.lattice.matrix))
    expected_virial = s_ref.detach() * volume / 160.21766208

    diff = (out["virials"].detach() - expected_virial).abs().max().item()
    assert diff < 1e-3, f"virial mismatch (max abs diff = {diff})"
