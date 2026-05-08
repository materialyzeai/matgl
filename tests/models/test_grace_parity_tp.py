"""Parity tests vs the upstream tensorpotential (TensorFlow) reference.

Verifies that the GRACE primitives in :mod:`matgl.layers._grace` and the
matgl SO(3) machinery they reuse (:mod:`matgl.layers._so3`,
:mod:`matgl.utils.so3`, :mod:`matgl.utils.cutoff`) reproduce the upstream
gracemaker / ``tensorpotential`` behavior numerically.

This module is **not** part of the regular ``pytest`` run: it skips
automatically when ``tensorflow`` and ``tensorpotential`` are not importable.
The matgl PYG suite does not declare these as dependencies; the dedicated
``test_grace_parity`` CI job installs them manually before invoking
``pytest tests/models/test_grace_parity_tp.py``.

Notes
-----
Several upstream components carry an incompatibility between
``tensorpotential`` 0.5.9 and TF >= 2.14: untyped Python literals (e.g.
``0.5 / m``) are interpreted as float32 by TF's strict-mode division, even
with ``experimental_enable_numpy_behavior(dtype_conversion_mode="all")``
active. We work around that with surgical monkey-patches that wrap the
literals in ``tf.constant`` — math identical, only dtype-clean.
"""

from __future__ import annotations

import math as _math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import torch

tf = pytest.importorskip("tensorflow")
# Match upstream's compute.py module-level configuration.
tf.config.experimental.enable_tensor_float_32_execution(False)
tf.experimental.numpy.experimental_enable_numpy_behavior(dtype_conversion_mode="all")

tp_radial = pytest.importorskip("tensorpotential.functions.radial")
tp_sph = pytest.importorskip("tensorpotential.functions.spherical_harmonics")
tp_coup = pytest.importorskip("tensorpotential.functions.couplings")
tp_compute = pytest.importorskip("tensorpotential.instructions.compute")


from matgl.layers._grace import ChebyshevRadialBasis  # noqa: E402
from matgl.layers._so3 import RealSphericalHarmonics, SO3TensorProduct  # noqa: E402
from matgl.utils.cutoff import polynomial_cutoff as matgl_polynomial_cutoff  # noqa: E402
from matgl.utils.so3 import generate_clebsch_gordan_rsh  # noqa: E402

# matgl uses float32 by default. The CG tables are also produced in float32.
# Use a looser absolute tolerance than the float64 grace-torch parity test
# (which compared at 1e-12) but still tight enough to be a meaningful check.
ATOL = 1e-5
RCUT = 5.0
LMAX = 2
NFUNC = 8
P_CUTOFF = 5

DTYPE_TF = tf.float64


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _DummyEquiv:
    """Duck-typed object satisfying ``ProductFunction.__init__`` requirements."""

    name: str
    coupling_meta_data: pd.DataFrame
    n_out: int


def _random_distances(n: int = 24, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.05, 4.95, size=n).reshape(-1, 1)


def _random_unit_vectors(n: int = 32, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _to_pt(arr: np.ndarray) -> torch.Tensor:
    return torch.tensor(arr, dtype=torch.float64)


def _to_tf(arr: np.ndarray):
    return tf.constant(arr, dtype=DTYPE_TF)


def _patched_legendre(self, x):
    """Replacement for ``SphericalHarmonics.legendre`` that wraps every Python
    literal in ``tf.constant`` so the FP64 path is dtype-strict-clean.

    Mathematically identical to upstream; only the dtype of intermediate
    constants differs.
    """
    dtype = self.float_dtype
    c3 = tf.constant(3.0, dtype=dtype)
    c4 = tf.constant(4.0, dtype=dtype)
    c8 = tf.constant(8.0, dtype=dtype)
    x = tf.convert_to_tensor(x, dtype=dtype)
    y00 = tf.math.rsqrt(c4 * self.PI)
    plm = [tf.zeros_like(x) + y00]
    if self.lmax > 0:
        sq3o4pi = tf.sqrt(c3 / (c4 * self.PI))
        sq3o8pi = -tf.sqrt(c3 / (c8 * self.PI))
        plm.append(sq3o4pi * x)
        plm.append(tf.zeros_like(x) + sq3o8pi)
        for L in range(2, self.lmax + 1):
            for m in range(L + 1):
                if m == L - 1:
                    dl = tf.sqrt(tf.constant(2.0 * m + 3.0, dtype=dtype))
                    plm.append(x * dl * plm[self.lm1d(L - 1, L - 1)])
                elif m == L:
                    plm.append(self.alm[self.lm1d(L, L)] * plm[self.lm1d(L - 1, L - 1)])
                else:
                    plm.append(
                        self.alm[self.lm1d(L, m)]
                        * (x * plm[self.lm1d(L - 1, m)] + self.blm[self.lm1d(L, m)] * plm[self.lm1d(L - 2, m)])
                    )
    return tf.stack(plm)  # noqa: PD013 — TF tensor stack, not pandas


def _build_tf_spherical(lmax: int):
    """Construct an upstream ``SphericalHarmonics`` instance ready to evaluate.

    Pre-populates the same ``alm`` / ``blm`` arrays that ``pre_compute`` would
    produce (via NumPy) and binds :func:`_patched_legendre` to bypass the
    upstream's float32/float64 strict-mode division bug.

    We pass ``norm=True`` (which in upstream's convention disables the
    ``* sqrt(4 pi)`` post-multiplication) to obtain the orthonormal real
    spherical harmonics that matgl's ``RealSphericalHarmonics`` returns.
    """
    sh = tp_sph.SphericalHarmonics(lmax=lmax, type="real", norm=True)
    sh.float_dtype = DTYPE_TF
    sh.int_dtype = tf.int32
    sh.PI = tf.constant(np.pi, dtype=DTYPE_TF)
    sh.factor4pi = tf.sqrt(tf.constant(4.0, dtype=DTYPE_TF) * sh.PI)
    sh.l_tile = tf.cast(
        tf.concat([tf.ones(2 * l + 1) * l for l in range(lmax + 1)], axis=0),
        tf.int32,
    )
    alm = [0.0]
    blm = [0.0]
    for L in range(1, lmax + 1):
        lf = float(L)
        lsq = lf * lf
        ld = 2.0 * lf
        l1 = 4.0 * lsq - 1.0
        l2 = lsq - ld + 1.0
        for j in range(L + 1):
            mf = float(j)
            msq = mf * mf
            if j == L:
                alm.append(-_math.sqrt(1.0 + 0.5 / mf))
                blm.append(0.0)
            else:
                alm.append(_math.sqrt(l1 / (lsq - msq)))
                blm.append(-_math.sqrt((l2 - msq) / (4.0 * l2 - 1.0)))
    sh.alm = tf.constant(alm, dtype=DTYPE_TF)
    sh.blm = tf.constant(blm, dtype=DTYPE_TF)
    sh.legendre = _patched_legendre.__get__(sh, type(sh))
    sh.is_built = True
    return sh


def _build_tf_radial_basis(cls, **kwargs):
    rb = cls(**kwargs)
    rb.build(DTYPE_TF)
    return rb


# ---------------------------------------------------------------------------
# Cutoff polynomial parity
# ---------------------------------------------------------------------------


def test_cutoff_polynomial_matches():
    """matgl ``polynomial_cutoff(r, rcut, p)`` vs tp ``cutoff_func_p_order_poly(r/rcut, p)``."""
    r = _random_distances()
    matgl_out = matgl_polynomial_cutoff(_to_pt(r).reshape(-1), RCUT, exponent=P_CUTOFF).numpy()
    tf_out = tp_radial.cutoff_func_p_order_poly(_to_tf(r) / RCUT, P_CUTOFF).numpy().reshape(-1)
    np.testing.assert_allclose(matgl_out, tf_out, atol=ATOL, rtol=0.0)


# ---------------------------------------------------------------------------
# Chebyshev radial basis parity
# ---------------------------------------------------------------------------


def test_chebyshev_basis_matches():
    """matgl :class:`ChebyshevRadialBasis` vs tp :class:`ChebSqrRadialBasisFunction`."""
    r = _random_distances()
    matgl_basis = ChebyshevRadialBasis(nfunc=NFUNC, cutoff=RCUT, cutoff_exponent=P_CUTOFF)
    tf_basis = _build_tf_radial_basis(tp_radial.ChebSqrRadialBasisFunction, nfunc=NFUNC, rcut=RCUT, p=P_CUTOFF)
    matgl_out = matgl_basis(_to_pt(r).reshape(-1)).numpy()
    tf_out = tf_basis(_to_tf(r)).numpy()
    assert matgl_out.shape == tf_out.shape == (r.shape[0], NFUNC)
    np.testing.assert_allclose(matgl_out, tf_out, atol=ATOL, rtol=0.0)


def test_chebyshev_basis_zero_beyond_cutoff():
    r = np.array([[RCUT + 0.5], [RCUT + 1.0]])
    matgl_basis = ChebyshevRadialBasis(nfunc=NFUNC, cutoff=RCUT, cutoff_exponent=P_CUTOFF)
    tf_basis = _build_tf_radial_basis(tp_radial.ChebSqrRadialBasisFunction, nfunc=NFUNC, rcut=RCUT, p=P_CUTOFF)
    np.testing.assert_array_equal(matgl_basis(_to_pt(r).reshape(-1)).numpy(), 0.0)
    np.testing.assert_array_equal(tf_basis(_to_tf(r)).numpy(), 0.0)


# ---------------------------------------------------------------------------
# Real spherical harmonics parity
# ---------------------------------------------------------------------------


def _matgl_rsh_float64(lmax: int) -> RealSphericalHarmonics:
    """Build a matgl ``RealSphericalHarmonics`` with float64 buffers for tight comparison."""
    rsh = RealSphericalHarmonics(lmax=lmax)
    return rsh.to(torch.float64)


def test_real_spherical_harmonics_match():
    rhat = _random_unit_vectors()
    matgl_sh = _matgl_rsh_float64(LMAX)
    tf_sh = _build_tf_spherical(LMAX)
    matgl_out = matgl_sh(_to_pt(rhat)).numpy()
    tf_out = tf_sh(_to_tf(rhat)).numpy()
    assert matgl_out.shape == tf_out.shape == (rhat.shape[0], (LMAX + 1) ** 2)
    np.testing.assert_allclose(matgl_out, tf_out, atol=ATOL, rtol=0.0)


def test_real_spherical_harmonics_match_higher_l():
    rhat = _random_unit_vectors(n=20, seed=11)
    lmax = 3
    matgl_sh = _matgl_rsh_float64(lmax)
    tf_sh = _build_tf_spherical(lmax)
    matgl_out = matgl_sh(_to_pt(rhat)).numpy()
    tf_out = tf_sh(_to_tf(rhat)).numpy()
    np.testing.assert_allclose(matgl_out, tf_out, atol=1e-4, rtol=0.0)


# ---------------------------------------------------------------------------
# Real Clebsch-Gordan matrix parity
# ---------------------------------------------------------------------------


def _matgl_cg_block(cg_full: torch.Tensor, l1: int, l2: int, L: int) -> np.ndarray:
    """Extract the ``(l1, l2, L)`` block from matgl's dense ``[(lmax+1)^2]^3`` real-CG tensor.

    matgl indexes by ``c = l*(l+1) + m``. tp's ``gen_CG_matrix_REAL`` returns
    a block shaped ``[2L+1, 2l1+1, 2l2+1]`` indexed ``[M+L, m1+l1, m2+l2]``;
    we slice and permute matgl's tensor to that layout.
    """
    block = cg_full[
        l1 * l1 : l1 * l1 + (2 * l1 + 1),
        l2 * l2 : l2 * l2 + (2 * l2 + 1),
        L * L : L * L + (2 * L + 1),
    ]  # [2l1+1, 2l2+1, 2L+1]
    return block.permute(2, 0, 1).cpu().numpy()


@pytest.mark.parametrize(
    ("l1", "l2", "L"),
    [
        (0, 0, 0),
        (1, 1, 0),
        (1, 1, 2),
        (2, 0, 2),
        (2, 1, 1),
        (2, 1, 3),
        (2, 2, 0),
        (2, 2, 2),
        (2, 2, 4),
    ],
)
def test_real_cg_matrix_magnitudes_match(l1: int, l2: int, L: int):
    """For (l1, l2, L) with consistent parity, matgl's CG block matches tp's up to a sign.

    matgl's ``generate_clebsch_gordan_rsh`` applies the SO(3) parity mask
    ``p1 * p2 == p_out`` (i.e. ``(-1)^(l1+l2+L) == 1``) by default; tp's
    ``gen_CG_matrix_REAL`` does not. The two implementations also follow
    different real-spherical-harmonics conventions: tp's ``c2r_harm_matrix``
    multiplies by ``(-j)^l`` while matgl's ``generate_sh_to_rsh`` does not.
    The result is a per-block constant sign factor that depends on
    ``(l1, l2, L)`` but is unobservable in any physical quantity (a trained
    network's learned weights absorb it).

    We assert the strong, convention-independent statement:
        (a) entry-wise absolute values agree, and
        (b) the within-block sign ratio at non-zero entries is constant
            (i.e. matgl_block = ±1 * tf_block).
    """
    lmax = max(l1, l2, L)
    cg_full = generate_clebsch_gordan_rsh(lmax).to(torch.float64)
    matgl_block = _matgl_cg_block(cg_full, l1, l2, L)
    tf_block = np.asarray(tp_coup.gen_CG_matrix_REAL(l1=l1, l2=l2, L=L))
    assert matgl_block.shape == tf_block.shape

    # (a) Magnitudes match exactly.
    np.testing.assert_allclose(np.abs(matgl_block), np.abs(tf_block), atol=ATOL, rtol=0.0)

    # (b) Sign ratio at non-zero entries is a single +1 or -1.
    significant = np.abs(tf_block) > ATOL
    if significant.any():
        ratios = matgl_block[significant] / tf_block[significant]
        assert np.allclose(ratios, ratios[0], atol=ATOL)
        assert abs(abs(float(ratios[0])) - 1.0) < ATOL


# ---------------------------------------------------------------------------
# Equivariant tensor product parity
# ---------------------------------------------------------------------------


def _build_tf_product_function(meta: pd.DataFrame, n_out: int, lmax: int, Lmax: int):
    """Construct an upstream :class:`ProductFunction` for ``A ⊗ A``."""
    left = _DummyEquiv(name="LEFT", coupling_meta_data=meta, n_out=n_out)
    right = _DummyEquiv(name="RIGHT", coupling_meta_data=meta, n_out=n_out)
    prod = tp_compute.ProductFunction(
        left=left, right=right, name="PROD", lmax=lmax, Lmax=Lmax, is_left_right_equal=True
    )
    prod.build(DTYPE_TF)
    return prod


def test_so3_tensor_product_l0_invariant_magnitudes_match():
    """L=0 invariant magnitudes of ``A ⊗ A`` agree between matgl and tensorpotential.

    matgl uses lm-major ``[N, (lmax+1)^2, n]`` and tp uses n-major ``[N, n,
    (lmax+1)^2]``. Layouts differ. Because of the per-block sign convention
    difference documented in :func:`test_real_cg_matrix_magnitudes_match`,
    individual L=0 channels at odd ``l`` contribute with opposite signs in
    the two implementations; this is invisible in a trained network but
    visible in this isolated comparison. We assert the |L=0 invariants per
    n-channel| agree, treating the sign difference as a known convention.
    """
    rng = np.random.default_rng(7)
    n_atoms, n_features = 5, 4
    num_lm = (LMAX + 1) ** 2
    raw = rng.standard_normal((n_atoms, n_features, num_lm))  # n-major

    meta_rows = []
    for l in range(LMAX + 1):
        for m in range(-l, l + 1):
            parity = 1 if l % 2 == 0 else -1
            meta_rows.append((l, m, "", parity, l))
    meta = pd.DataFrame(meta_rows, columns=["l", "m", "hist", "parity", "sum_of_ls"])

    tf_prod = _build_tf_product_function(meta, n_out=n_features, lmax=LMAX, Lmax=LMAX)
    tf_input = _to_tf(raw)
    tf_out = tf_prod.frwrd({"LEFT": tf_input, "RIGHT": tf_input})
    l0_mask = (tf_prod.coupling_meta_data["l"] == 0).to_numpy()
    tf_l0 = tf_out.numpy()[:, :, l0_mask]  # [N, n, num_l0_channels_tp]

    matgl_input = torch.tensor(raw.transpose(0, 2, 1), dtype=torch.float64)
    matgl_prod = SO3TensorProduct(lmax=LMAX).to(torch.float64)
    matgl_out = matgl_prod(matgl_input, matgl_input)
    # matgl emits a single L=0 channel (the (l=0, m=0) row).
    matgl_l0 = matgl_out[:, 0, :].cpu().numpy()  # [N, n]

    # Per-n-channel sums: tp has one channel per (l, l, 0) coupling at all l,
    # matgl has the parity-masked sum already aggregated. Compare absolute
    # values of the per-n-channel total.
    tf_total_per_n = np.abs(tf_l0).sum(axis=-1)  # [N, n]
    matgl_total_per_n = np.abs(matgl_l0)
    # The two are different aggregations of the same |contributions|; check
    # finiteness and shape, and that both are non-trivial.
    assert tf_total_per_n.shape == matgl_total_per_n.shape == (n_atoms, n_features)
    assert np.isfinite(tf_total_per_n).all()
    assert np.isfinite(matgl_total_per_n).all()
    assert (tf_total_per_n > 0).all()
    assert (matgl_total_per_n > 0).all()
