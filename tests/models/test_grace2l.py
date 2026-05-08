from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("GRACE2L is PYG-only", allow_module_level=True)


from matgl.apps.pes import Potential
from matgl.layers._grace import GraceSPBasisEquivariant, pad_lm_axis
from matgl.models import GRACE2L


def _set_pos_and_pbc(graph, lat):
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]


def _make_grace2l(**overrides):
    cfg = {
        "element_types": ("Mo", "S"),
        "cutoff": 5.0,
        "n_rad_base": 6,
        "n_rad_max": 6,
        "lmax": 2,
        "embedding_size": 8,
        "max_order": 3,
        "indicator_lmax": 1,
        "indicator_n_max": 8,
        "readout_hidden": (32,),
    }
    cfg.update(overrides)
    return GRACE2L(**cfg)


def _check_scalar_output(output):
    assert torch.numel(output) == 1
    assert torch.isfinite(output).all()


# ---------------------------------------------------------------------------
# Layer-level tests for the equivariant SP basis + pad helper
# ---------------------------------------------------------------------------


def test_pad_lm_axis_grow_and_truncate():
    x = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)  # lm dim has 4 = (1+1)^2
    grown = pad_lm_axis(x, current_lmax=1, target_lmax=2)
    assert grown.shape == (2, 9, 3)
    # Original entries preserved at indices 0..3.
    assert torch.allclose(grown[:, :4, :], x)
    # New slots are exactly zero.
    assert torch.all(grown[:, 4:, :] == 0)

    same = pad_lm_axis(x, current_lmax=1, target_lmax=1)
    assert same is x  # no-op short-circuit

    truncated = pad_lm_axis(x, current_lmax=1, target_lmax=0)
    assert truncated.shape == (2, 1, 3)
    assert torch.allclose(truncated, x[:, :1, :])


def test_grace_sp_basis_equivariant_rejects_indicator_lmax_too_large():
    with pytest.raises(ValueError, match="indicator_lmax"):
        GraceSPBasisEquivariant(lmax=2, n_rad_max=4, indicator_lmax=3, indicator_n_max=4)


def test_grace_sp_basis_equivariant_shape():
    lmax = 2
    indicator_lmax = 1
    n_rad_max = 4
    indicator_n_max = 5
    n_atoms = 5
    n_edges = 8

    sp = GraceSPBasisEquivariant(
        lmax=lmax,
        n_rad_max=n_rad_max,
        indicator_lmax=indicator_lmax,
        indicator_n_max=indicator_n_max,
    )
    radial_nl = torch.randn(n_edges, (lmax + 1) ** 2, n_rad_max)
    spherical_lm = torch.randn(n_edges, (lmax + 1) ** 2)
    indicator = torch.randn(n_atoms, (indicator_lmax + 1) ** 2, indicator_n_max)
    edge_index = torch.randint(0, n_atoms, (2, n_edges))
    out = sp(
        radial_nl=radial_nl,
        spherical_lm=spherical_lm,
        indicator=indicator,
        edge_index=edge_index,
        num_nodes=n_atoms,
    )
    assert out.shape == (n_atoms, (lmax + 1) ** 2, n_rad_max)


# ---------------------------------------------------------------------------
# Model-level tests
# ---------------------------------------------------------------------------


def test_grace2l_forward_returns_scalar(graph_MoS):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    _set_pos_and_pbc(graph, lat)
    model = _make_grace2l().to(matgl.float_th)
    output = model(graph)
    _check_scalar_output(output)


def test_grace2l_save_load_roundtrip(graph_MoS, tmp_path):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    _set_pos_and_pbc(graph, lat)

    model = _make_grace2l().to(matgl.float_th)
    model.eval()
    out_before = model(graph).detach()

    save_dir = str(tmp_path)
    model.save(save_dir)
    loaded = GRACE2L.load(save_dir).to(matgl.float_th)
    loaded.eval()
    out_after = loaded(graph).detach()
    assert torch.allclose(out_before, out_after, atol=1e-10)
    for fname in ("model.pt", "state.pt", "model.json"):
        assert os.path.exists(os.path.join(save_dir, fname))


def test_grace2l_translation_invariance(graph_MoS):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    _set_pos_and_pbc(graph, lat)
    model = _make_grace2l().to(matgl.float_th)
    model.eval()

    e0 = model(graph).detach()
    pos_orig = graph.pos.detach().clone()
    shift = torch.tensor([0.7, -1.2, 0.4], dtype=matgl.float_th)
    graph.pos = pos_orig + shift
    e1 = model(graph).detach()
    graph.pos = pos_orig
    assert torch.allclose(e0, e1, atol=1e-6)


def test_grace2l_rotation_invariance(graph_MoS):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    _set_pos_and_pbc(graph, lat)
    model = _make_grace2l().to(matgl.float_th)
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


def test_grace2l_with_potential_returns_forces_and_stresses(graph_MoS):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    model = _make_grace2l().to(matgl.float_th)
    pot = Potential(model=model, calc_forces=True, calc_stresses=True)
    energies, forces, stresses, _ = pot(graph, lat)
    _check_scalar_output(energies)
    n_atoms = graph.frac_coords.shape[0]
    assert forces.shape == (n_atoms, 3)
    assert stresses.shape == (3, 3)
    assert torch.allclose(forces.sum(0), torch.zeros(3, dtype=matgl.float_th), atol=1e-5)


def test_grace2l_invalid_max_order():
    with pytest.raises(ValueError, match="max_order must be >= 1"):
        GRACE2L(element_types=("Mo", "S"), max_order=0)


def test_grace2l_invalid_indicator_lmax():
    with pytest.raises(ValueError, match="indicator_lmax"):
        GRACE2L(element_types=("Mo", "S"), lmax=2, indicator_lmax=3)
    with pytest.raises(ValueError, match="indicator_lmax"):
        GRACE2L(element_types=("Mo", "S"), lmax=2, indicator_lmax=-1)


def test_grace2l_indicator_lmax_zero_collapses_to_scalar_indicator(graph_MoS):
    """``indicator_lmax=0`` keeps only the L=0 component of layer 1's
    descriptor; layer 2 then sees a per-atom *scalar* indicator (still routed
    through ``GraceSPBasisEquivariant``). This must remain finite and produce
    valid forces.
    """
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    _set_pos_and_pbc(graph, lat)
    model = _make_grace2l(indicator_lmax=0).to(matgl.float_th)
    output = model(graph)
    _check_scalar_output(output)
