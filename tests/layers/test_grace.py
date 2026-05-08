from __future__ import annotations

import pytest
import torch

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("GRACE is PYG-only", allow_module_level=True)


from matgl.layers._grace import (
    ChebyshevRadialBasis,
    GraceACEStack,
    GraceSPBasis,
    GraceSPBasisEquivariant,
    LinearRadialFunction,
    collect_invariants,
    pad_lm_axis,
)


def test_chebyshev_radial_basis_shape_and_cutoff():
    basis = ChebyshevRadialBasis(nfunc=8, cutoff=5.0)
    r = torch.linspace(0.1, 5.5, 30)
    out = basis(r)
    assert out.shape == (30, 8)
    # Polynomial cutoff envelope strictly zero past rcut.
    assert torch.all(out[r >= 5.0] == 0)


def test_chebyshev_basis_accepts_1d_or_2d_input():
    basis = ChebyshevRadialBasis(nfunc=4, cutoff=4.0)
    r = torch.tensor([0.5, 1.0, 2.5])
    assert torch.allclose(basis(r), basis(r.unsqueeze(-1)))


def test_linear_radial_function_lm_layout():
    nfunc, n_rad_max, lmax = 6, 4, 2
    rad_fn = LinearRadialFunction(nfunc=nfunc, n_rad_max=n_rad_max, lmax=lmax)
    basis_values = torch.randn(7, nfunc)
    out = rad_fn(basis_values)
    assert out.shape == (7, (lmax + 1) ** 2, n_rad_max)
    # The 2l+1 m-values within a given l share the same R_{nl}(r) — verify
    # that the slice for l=1 (indices 1, 2, 3) is constant along the lm axis.
    l1_block = out[:, 1:4, :]
    assert torch.allclose(l1_block[:, 0, :], l1_block[:, 1, :])
    assert torch.allclose(l1_block[:, 0, :], l1_block[:, 2, :])


def test_grace_sp_basis_shape():
    lmax, n_rad_max = 2, 4
    n = 5
    e = 8
    sp = GraceSPBasis(lmax=lmax, n_rad_max=n_rad_max, n_elements=3, embedding_size=4, avg_n_neigh=2.0)
    radial_nl = torch.randn(e, (lmax + 1) ** 2, n_rad_max)
    spherical_lm = torch.randn(e, (lmax + 1) ** 2)
    edge_index = torch.randint(0, n, (2, e))
    node_type = torch.randint(0, 3, (n,))
    out = sp(
        radial_nl=radial_nl,
        spherical_lm=spherical_lm,
        edge_index=edge_index,
        node_type=node_type,
        num_nodes=n,
    )
    assert out.shape == (n, (lmax + 1) ** 2, n_rad_max)


def test_grace_ace_stack_chain_length():
    lmax, n_rad_max = 2, 3
    stack = GraceACEStack(lmax=lmax, max_order=4)
    a = torch.randn(6, (lmax + 1) ** 2, n_rad_max)
    out = stack(a)
    assert len(out) == 4
    for tensor in out:
        assert tensor.shape == a.shape


def test_grace_ace_stack_rejects_zero_order():
    with pytest.raises(ValueError, match="max_order must be >= 1"):
        GraceACEStack(lmax=2, max_order=0)


def test_collect_invariants_shape():
    n, n_rad_max = 4, 5
    tensors = [torch.randn(n, 9, n_rad_max) for _ in range(3)]
    out = collect_invariants(tensors)
    assert out.shape == (n, 3 * n_rad_max)
    # Verify it slices the L=0 row, i.e. column index 0.
    assert torch.allclose(out[:, :n_rad_max], tensors[0][:, 0, :])


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
