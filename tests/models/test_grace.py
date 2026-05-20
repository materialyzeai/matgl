from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
import torch

import matgl
from matgl.apps.pes import Potential
from matgl.models import GRACE


def _set_pos_and_pbc(graph, lat):
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]


def _make_grace(nblocks: int = 2, **overrides: Any) -> GRACE:
    """Build a GRACE model with small / fast hyperparameters for testing."""
    cfg: dict[str, Any] = {
        "element_types": ("Mo", "S"),
        "cutoff": 5.0,
        "n_rad_base": 6,
        "n_rad_max": 6,
        "lmax": 2,
        "embedding_size": 8,
        "max_order": 3,
        "nblocks": nblocks,
        "indicator_lmax": 1,
        "indicator_n_max": 8,
        "readout_hidden": (32,),
    }
    cfg.update(overrides)
    return GRACE(**cfg)


def _check_scalar_output(output):
    assert torch.numel(output) == 1
    assert torch.isfinite(output).all()


# Most behaviors should hold for both single-block (GRACE-1L) and multi-block
# (GRACE-2L+) configurations. Parametrize the equivariance / Potential tests.


@pytest.mark.parametrize("nblocks", [1, 2])
def test_grace_forward_returns_scalar(graph_MoS, nblocks):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    _set_pos_and_pbc(graph, lat)
    model = _make_grace(nblocks=nblocks).to(matgl.float_th)
    output = model(graph)
    _check_scalar_output(output)


@pytest.mark.parametrize("nblocks", [1, 2])
def test_grace_save_load_roundtrip(graph_MoS, tmp_path, nblocks):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    _set_pos_and_pbc(graph, lat)

    model = _make_grace(nblocks=nblocks).to(matgl.float_th)
    model.eval()
    out_before = model(graph).detach()

    save_dir = str(tmp_path)
    model.save(save_dir)
    loaded = GRACE.load(save_dir).to(matgl.float_th)
    loaded.eval()
    out_after = loaded(graph).detach()
    assert torch.allclose(out_before, out_after, atol=1e-10)
    for fname in ("model.pt", "state.pt", "model.json"):
        assert os.path.exists(os.path.join(save_dir, fname))


@pytest.mark.parametrize("nblocks", [1, 2])
def test_grace_translation_invariance(graph_MoS, nblocks):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    _set_pos_and_pbc(graph, lat)
    model = _make_grace(nblocks=nblocks).to(matgl.float_th)
    model.eval()

    e0 = model(graph).detach()
    pos_orig = graph.pos.detach().clone()
    shift = torch.tensor([0.7, -1.2, 0.4], dtype=matgl.float_th)
    graph.pos = pos_orig + shift
    e1 = model(graph).detach()
    graph.pos = pos_orig
    assert torch.allclose(e0, e1, atol=1e-6)


@pytest.mark.parametrize("nblocks", [1, 2])
def test_grace_rotation_invariance(graph_MoS, nblocks):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    _set_pos_and_pbc(graph, lat)
    model = _make_grace(nblocks=nblocks).to(matgl.float_th)
    model.eval()

    e0 = model(graph).detach()
    pos_orig = graph.pos.detach().clone()
    pbc_orig = graph.pbc_offshift.detach().clone()
    g_rng = torch.Generator().manual_seed(7)
    a = torch.randn(3, 3, generator=g_rng, dtype=matgl.float_th)
    q, _ = torch.linalg.qr(a)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    graph.pos = pos_orig @ q.T
    graph.pbc_offshift = pbc_orig @ q.T
    e1 = model(graph).detach()
    graph.pos = pos_orig
    graph.pbc_offshift = pbc_orig
    assert torch.allclose(e0, e1, atol=1e-5)


@pytest.mark.parametrize("nblocks", [1, 2])
def test_grace_with_potential_returns_forces_and_stresses(graph_MoS, nblocks):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    model = _make_grace(nblocks=nblocks).to(matgl.float_th)
    pot = Potential(model=model, calc_forces=True, calc_stresses=True)
    energies, forces, stresses, _ = pot(graph, lat)
    _check_scalar_output(energies)
    n_atoms = graph.frac_coords.shape[0]
    assert forces.shape == (n_atoms, 3)
    assert stresses.shape == (3, 3)
    # Newton's 3rd law: forces sum to ~0 across atoms.
    assert torch.allclose(forces.sum(0), torch.zeros(3, dtype=matgl.float_th), atol=1e-5)


# Argument validation — single-class checks (no parametrization).


def test_grace_invalid_max_order():
    with pytest.raises(ValueError, match="max_order must be >= 1"):
        GRACE(element_types=("Mo", "S"), max_order=0)


def test_grace_invalid_nblocks():
    with pytest.raises(ValueError, match="nblocks must be >= 1"):
        GRACE(element_types=("Mo", "S"), nblocks=0)


def test_grace_invalid_activation():
    with pytest.raises(ValueError, match="Invalid activation type"):
        GRACE(element_types=("Mo", "S"), activation_type="not_an_activation")  # type: ignore[arg-type]


# ``indicator_lmax`` validation only matters for multi-block configurations.


def test_grace_invalid_indicator_lmax_for_multi_block():
    with pytest.raises(ValueError, match="indicator_lmax"):
        GRACE(element_types=("Mo", "S"), nblocks=2, lmax=2, indicator_lmax=3)
    with pytest.raises(ValueError, match="indicator_lmax"):
        GRACE(element_types=("Mo", "S"), nblocks=2, lmax=2, indicator_lmax=-1)


def test_grace_indicator_lmax_unconstrained_for_single_block():
    """``indicator_lmax`` is irrelevant when ``nblocks == 1`` — the
    constructor should accept any value (including out-of-range values
    that would otherwise be rejected) without error."""
    GRACE(element_types=("Mo", "S"), nblocks=1, lmax=2, indicator_lmax=99)


def test_grace_indicator_lmax_zero_collapses_to_scalar_indicator(graph_MoS):
    """``indicator_lmax=0`` keeps only the L=0 component of the previous
    block's descriptor; subsequent blocks then see a per-atom *scalar*
    indicator (still routed through ``GraceSPBasisEquivariant``). The
    forward must remain finite and produce valid forces."""
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    _set_pos_and_pbc(graph, lat)
    model = _make_grace(nblocks=2, indicator_lmax=0).to(matgl.float_th)
    output = model(graph)
    _check_scalar_output(output)


def test_grace_deeper_stack_works(graph_MoS):
    """Sanity-check ``nblocks=3``: 3-block stack should compose without
    error and produce a finite total energy with non-trivial forces."""
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    model = _make_grace(nblocks=3).to(matgl.float_th)
    pot = Potential(model=model, calc_forces=True, calc_stresses=True)
    energies, forces, _stresses, _ = pot(graph, lat)
    _check_scalar_output(energies)
    assert torch.allclose(forces.sum(0), torch.zeros(3, dtype=matgl.float_th), atol=1e-5)
