from __future__ import annotations

import torch

import matgl
from matgl.layers._so3 import (
    RealSphericalHarmonics,
    SO3Convolution,
    SO3GatedNonlinearity,
    SO3ParametricGatedNonlinearity,
    SO3TensorProduct,
    scalar2rsh,
)


def test_real_spherical_harmonics():
    lmax = 2
    rsh = RealSphericalHarmonics(lmax)

    vec = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=matgl.float_th)
    output = rsh(vec)

    assert output.shape, (vec.shape[0], (lmax + 1) ** 2)


def test_generatre_Ylm_coefficients():
    lmax = 2
    rsh = RealSphericalHarmonics(lmax)

    powers, zpow, cAm, cBm, cPi = rsh._generate_Ylm_coefficients(lmax)

    assert powers.shape, (lmax + 1, lmax + 1, 2)
    assert zpow.shape, (lmax + 1, lmax + 1)
    assert cAm.shape, (lmax, lmax + 1)
    assert cBm.shape, (lmax, lmax + 1)
    assert cPi.shape, (lmax + 1, lmax + 1, lmax // 2 + 1)


def test_scalar2rsh():
    x = torch.rand((4, 1, 16), dtype=matgl.float_th)
    lmax = 2
    result = scalar2rsh(x, lmax)

    expected_shape = (x.shape[0], (lmax + 1) ** 2, x.shape[2])

    assert result.shape == expected_shape
    assert torch.allclose(result[:, : x.shape[1], :], x)
    assert torch.allclose(result[:, x.shape[1] :, :], torch.zeros_like(result[:, x.shape[1] :, :]))


def test_so3_tensor_product():
    n_atoms = 3
    n_features = 4
    lmax = 2

    x1 = torch.rand((n_atoms, (lmax + 1) ** 2, n_features))
    x2 = torch.rand((n_atoms, (lmax + 1) ** 2, n_features))

    tp = SO3TensorProduct(lmax=lmax)

    results = tp(x1, x2)

    expected_shape = (n_atoms, (lmax + 1) ** 2, n_features)

    assert results.shape == expected_shape


def test_so3_convolution():
    n_atoms = 3
    n_neighbors = 4
    lmax = 2
    n_atom_basis = 4
    n_radial = 5

    x = torch.rand((n_atoms, (lmax + 1) ** 2, n_atom_basis))
    radial_ij = torch.rand((n_neighbors, n_radial))
    dir_ij = torch.rand((n_neighbors, (lmax + 1) ** 2))
    cutoff_ij = torch.rand((n_neighbors, 1))
    idx_i = torch.tensor([0, 0, 1, 2])
    idx_j = torch.tensor([1, 2, 0, 0])

    conv = SO3Convolution(lmax=lmax, n_atom_basis=n_atom_basis, n_radial=n_radial)
    result = conv(x, radial_ij, dir_ij, cutoff_ij, idx_i, idx_j)

    expected_shape = (n_atoms, (lmax + 1) ** 2, n_atom_basis)
    assert result.shape == expected_shape


def test_so3_parametric_gated_nonlinearity():
    n_atoms = 3
    n_in = 3
    lmax = 2

    x = torch.rand((n_atoms, (lmax + 1) ** 2, n_in))

    gated_nonlinearity = SO3ParametricGatedNonlinearity(n_in=n_in, lmax=lmax)
    result = gated_nonlinearity(x)

    expected_shape = (n_atoms, (lmax + 1) ** 2, n_in)
    assert result.shape == expected_shape


def test_so3_gated_nonlinearity():
    n_atoms = 3
    lmax = 2

    x = torch.rand((n_atoms, (lmax + 1) ** 2, 3))

    gated_nonlinearity = SO3GatedNonlinearity(lmax=lmax)
    result = gated_nonlinearity(x)

    expected_shape = (n_atoms, (lmax + 1) ** 2, 3)
    assert result.shape == expected_shape
