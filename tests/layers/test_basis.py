from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.testing import assert_close

from matgl.layers._basis import (
    ExpNormalFunction,
    FourierExpansion,
    GaussianExpansion,
    RadialBesselFunction,
    SphericalBesselFunction,
    SphericalHarmonicsFunction,
    spherical_bessel_smooth,
)
from matgl.layers._three_body import combine_sbf_shf
from matgl.utils.maths import _get_lambda_func


def test_gaussian():
    r = torch.linspace(1.0, 5.0, 11)
    rbf_gaussian = GaussianExpansion(initial=0.0, final=5.0, num_centers=10, width=0.5)
    rbf = rbf_gaussian(r)
    assert [rbf.size(dim=0), rbf.size(dim=1)] == [11, 10]

    rbf_gaussian = GaussianExpansion()
    r = torch.tensor([1.0])
    rbf = rbf_gaussian(r)
    # check the shape of a vector
    assert np.allclose([rbf.size(dim=0), rbf.size(dim=1)], [1, 20])
    # check the first value of expanded distance
    assert np.allclose(rbf[0][0], np.exp(-0.5 * np.power(1.0 - 0.0, 2.0)))
    # check the last value of expanded distance
    assert np.allclose(rbf[0][-1], np.exp(-0.5 * np.power(1.0 - 4.0, 2.0)))

    rbf_gaussian = GaussianExpansion(width=None)
    r = torch.tensor([1.0])
    rbf = rbf_gaussian(r)
    # check the shape of a vector
    assert np.allclose([rbf.size(dim=0), rbf.size(dim=1)], [1, 20])
    # check the first value of expanded distance
    assert rbf[0][0].numpy() == pytest.approx(0.00865169521421194)
    rbf_gaussian.reset_parameters()


def test_spherical_bessel_function():
    r = torch.linspace(1.0, 5.0, 11)
    rbf_sb = SphericalBesselFunction(max_n=3, max_l=3, cutoff=5.0, smooth=False)
    rbf = rbf_sb(r)
    assert [rbf.size(dim=0), rbf.size(dim=1)] == [11, 9]

    rbf_sb = SphericalBesselFunction(max_n=3, max_l=3, cutoff=5.0, smooth=True)
    rbf = rbf_sb(r)
    assert [rbf.size(dim=0), rbf.size(dim=1)] == [11, 3]


def test_exp_normal_function():
    r = torch.linspace(1.0, 5.0, 11)
    rbf = ExpNormalFunction(cutoff=5.0, num_rbf=3, learnable=False)
    res = rbf(r)
    assert [res.size(dim=0), res.size(dim=1)] == [11, 3]

    rbf = ExpNormalFunction(cutoff=5.0, num_rbf=3, learnable=True)
    res = rbf(r)
    assert [res.size(dim=0), res.size(dim=1)] == [11, 3]


def test_spherical_harmonic_function():
    theta = torch.linspace(-1, 1, 10)
    phi = torch.linspace(0, 2 * np.pi, 10)
    abf_sb = SphericalHarmonicsFunction(max_l=3, use_phi=True)
    abf = abf_sb(theta, phi)
    assert [abf.size(dim=0), abf.size(dim=1)] == [10, 9]


def test_spherical_bessel_harmonics_function():
    r = torch.empty(10).normal_()
    sbf = SphericalBesselFunction(max_l=3, cutoff=5.0, max_n=3, smooth=False)
    res = sbf(r)

    shf = SphericalHarmonicsFunction(max_l=3, use_phi=True)
    res_shf = shf(cos_theta=torch.linspace(-1, 1, 10), phi=torch.linspace(0, 2 * np.pi, 10))

    assert res_shf.numpy().shape == (10, 9)
    combined = combine_sbf_shf(res, res_shf, max_n=3, max_l=3, use_phi=True)

    assert combined.shape == (10, 27)

    res_shf2 = SphericalHarmonicsFunction(max_l=3, use_phi=False)(
        cos_theta=torch.linspace(-1, 1, 10), phi=torch.linspace(0, 2 * np.pi, 10)
    )
    combined = combine_sbf_shf(res, res_shf2, max_n=3, max_l=3, use_phi=False)

    assert combined.shape == (10, 9)
    rdf = spherical_bessel_smooth(r, cutoff=5.0, max_n=3)
    assert rdf.numpy().shape == (10, 3)


@pytest.mark.parametrize("learnable", [True, False])
def test_radial_bessel_function(learnable):
    max_n = 3
    r = torch.empty(10).normal_()
    rbf = RadialBesselFunction(max_n=max_n, cutoff=5.0, learnable=learnable)
    res = rbf(r)
    assert res.shape == (10, max_n)

    # compare with spherical bessel function
    sbf = SphericalBesselFunction(max_l=1, max_n=max_n, cutoff=5.0, smooth=False)
    res1 = sbf(r)
    res2 = sbf.rbf_j0(r, cutoff=5.0, max_n=max_n)

    assert_close(res, res1.float())
    assert_close(res, res2.float())

    if learnable:
        assert rbf.frequencies.requires_grad
    else:
        assert not rbf.frequencies.requires_grad


@pytest.mark.parametrize("learnable", [True, False])
def test_fourier_expansion(learnable):
    max_f = 5
    fe = FourierExpansion(max_f=max_f, learnable=learnable)
    x = torch.randn(10)
    res = fe(x)

    assert res.shape == (x.shape[0], 1 + max_f * 2)

    cosines = torch.cos(torch.outer(x, torch.arange(0, max_f + 1))) / torch.pi
    assert_close(res[:, ::2], cosines)

    sines = torch.sin(torch.outer(x, torch.arange(1, max_f + 1))) / np.pi
    assert_close(res[:, 1::2], sines)

    interval = 2.0
    fe = FourierExpansion(max_f=max_f, interval=interval, learnable=learnable)
    res = fe(x)

    cosines = torch.cos(torch.outer(x, torch.arange(0, max_f + 1)) * np.pi / interval) / interval
    assert_close(res[:, ::2], cosines)

    sines = torch.sin(torch.outer(x, torch.arange(1, max_f + 1)) * np.pi / interval) / interval
    assert_close(res[:, 1::2], sines)

    if learnable:
        assert fe.frequencies.requires_grad
    else:
        assert not fe.frequencies.requires_grad


@pytest.fixture
def restore_dtype():
    old = torch.get_default_dtype()
    yield
    torch.set_default_dtype(old)


def test_smooth_sbf_matches_closed_form(restore_dtype):
    """The n=0 basis function is sqrt(2)(2 sin(pi r/5) + sin(2 pi r/5))/(5 r)."""
    torch.set_default_dtype(torch.float32)
    sbf = SphericalBesselFunction(max_l=3, max_n=3, cutoff=5.0, smooth=True)
    r = torch.linspace(0.3, 5.0, 257, dtype=torch.float64)
    want = math.sqrt(2.0) * (2 * torch.sin(math.pi * r / 5) + torch.sin(2 * math.pi * r / 5)) / (5 * r)
    got = sbf(r)[:, 0]
    # `want` has an exact zero at r == cutoff, so the tolerance is normalised to
    # the amplitude of the basis function rather than applied pointwise.
    scale = want.abs().max()
    assert (got - want).abs().max() <= 1e-12 * scale


def test_smooth_sbf_independent_of_default_dtype(restore_dtype):
    """The basis must not change with the ambient default dtype."""
    r = torch.linspace(0.3, 5.0, 257, dtype=torch.float64)
    out = {}
    for dtype in (torch.float32, torch.float64):
        torch.set_default_dtype(dtype)
        out[dtype] = SphericalBesselFunction(3, 3, 5.0, smooth=True)(r)
    assert torch.equal(out[torch.float32], out[torch.float64])


def test_smooth_sbf_lambda_cache_is_reused(restore_dtype):
    """Identical modules must share the cached symbolic functions."""
    torch.set_default_dtype(torch.float32)
    _get_lambda_func.cache_clear()
    for _ in range(4):
        SphericalBesselFunction(3, 3, 5.0, smooth=True)
    info = _get_lambda_func.cache_info()
    assert info.currsize == 1, f"cache did not coalesce: {info}"
    assert info.hits == 3, f"cache never hit: {info}"
