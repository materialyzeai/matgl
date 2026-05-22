"""JAX ports of the radial basis expansions used by TensorNet / QET.

Covers the three bases :class:`matgl.layers.BondExpansion` can select:

* smooth spherical Bessel  (``rbf_type="SphericalBessel", use_smooth=True``)  --
  the basis used by the MatPES foundation potentials;
* plain spherical Bessel   (``use_smooth=False``)  -- ``j_l`` for ``l = 0..4``;
* Gaussian expansion.

The non-learned Bessel-root / normalisation constants are NOT part of a model's
``state_dict``; they are rederived here exactly as ``SphericalBesselFunction``
does, reusing the in-package ``SPHERICAL_BESSEL_ROOTS`` table.
"""

from __future__ import annotations

from math import pi, sqrt

import jax.numpy as jnp
import numpy as np

from matgl.utils.maths import SPHERICAL_BESSEL_ROOTS

# Same float32 root table the torch SphericalBesselFunction uses — reused here so
# the non-smooth basis matches the torch model exactly.
_SB_ROOTS = SPHERICAL_BESSEL_ROOTS.detach().cpu().numpy()  # (128, 128)


# --------------------------------------------------------------------------
# spherical Bessel j_l for l = 0..4 (explicit Rayleigh forms)
# --------------------------------------------------------------------------


def _jn(lval: int, x):
    """Spherical Bessel function of the first kind, order ``lval`` (0-4)."""
    s, c = jnp.sin(x), jnp.cos(x)
    if lval == 0:
        return s / x
    if lval == 1:
        return s / x**2 - c / x
    if lval == 2:
        return (3.0 / x**3 - 1.0 / x) * s - 3.0 / x**2 * c
    if lval == 3:
        return (15.0 / x**4 - 6.0 / x**2) * s - (15.0 / x**3 - 1.0 / x) * c
    if lval == 4:
        return (105.0 / x**5 - 45.0 / x**3 + 1.0 / x) * s - (105.0 / x**4 - 10.0 / x**2) * c
    raise NotImplementedError(f"spherical Bessel j_l only ported for l<=4 (got {lval})")


# --------------------------------------------------------------------------
# smooth spherical Bessel (port of matgl.layers._basis.spherical_bessel_smooth)
# --------------------------------------------------------------------------


def _spherical_bessel_smooth(r, cutoff: float, max_n: int):
    """Orthogonal smooth basis with vanishing 1st/2nd derivatives at the cutoff."""
    n = jnp.arange(max_n, dtype=r.dtype)
    sign = 1.0 - 2.0 * (n % 2)  # (-1)**n, exact for integer n
    rr = r[:, None]

    def sinc(x):
        return jnp.sin(x) / x

    fnr = (
        sign
        * sqrt(2.0)
        * pi
        / cutoff**1.5
        * (n + 1)
        * (n + 2)
        / jnp.sqrt(2 * n**2 + 6 * n + 5)
        * (sinc(rr * (n + 1) * pi / cutoff) + sinc(rr * (n + 2) * pi / cutoff))
    )  # (E, max_n)

    en = n**2 * (n + 2) ** 2 / (4 * (n + 1) ** 4 + 1)
    dn = [jnp.asarray(1.0, dtype=r.dtype)]
    for i in range(1, max_n):
        dn.append(1 - en[i] / dn[-1])

    gn = [fnr[:, 0]]
    for i in range(1, max_n):
        gn.append(1 / jnp.sqrt(dn[i]) * (fnr[:, i] + jnp.sqrt(en[i] / dn[i - 1]) * gn[-1]))
    return jnp.stack(gn, axis=1)  # (E, max_n)


# --------------------------------------------------------------------------
# basis config + dispatch
# --------------------------------------------------------------------------


def build_basis_config(model) -> dict:
    """Extract the radial-basis config from a torch ``TensorNet`` / ``QET``."""
    be = model.bond_expansion
    rbf_type = be.rbf_type.lower()
    cfg: dict = {"rbf_type": rbf_type, "cutoff": float(be.cutoff)}
    if rbf_type == "sphericalbessel":
        cfg["smooth"] = bool(be.smooth)
        cfg["max_l"] = int(be.max_l)
        cfg["max_n"] = int(be.max_n)
        if not be.smooth:
            max_l, max_n = cfg["max_l"], cfg["max_n"]
            roots = _SB_ROOTS[:max_l, :max_n]
            factor = sqrt(2.0 / float(be.cutoff) ** 3)
            inv_norm = np.stack([factor / np.abs(_jn(lv + 1, roots[lv])) for lv in range(max_l)], axis=0)
            cfg["roots"] = jnp.asarray(roots)
            cfg["inv_norm"] = jnp.asarray(inv_norm)
    elif rbf_type == "gaussian":
        ge = be.rbf
        cfg["centers"] = jnp.asarray(ge.centers.detach().cpu().numpy())
        cfg["width"] = float(ge.width)
    else:
        raise NotImplementedError(f"rbf_type {rbf_type!r} not ported (use SphericalBessel or Gaussian)")
    return cfg


def bond_expansion(cfg: dict, r_safe):
    """Expand a 1-D distance array ``r_safe`` into radial-basis features.

    ``r_safe`` must be strictly positive (callers substitute a dummy distance for
    padded edges) so the ``sin(x)/x`` terms never hit ``0/0``.
    """
    rbf_type = cfg["rbf_type"]
    if rbf_type == "sphericalbessel":
        if cfg["smooth"]:
            return _spherical_bessel_smooth(r_safe, cfg["cutoff"], cfg["max_n"])
        cutoff = cfg["cutoff"]
        scaled = jnp.clip(r_safe, max=cutoff)[:, None] / cutoff  # (E, 1)
        out = [_jn(lv, scaled * cfg["roots"][lv]) * cfg["inv_norm"][lv] for lv in range(cfg["max_l"])]
        return jnp.concatenate(out, axis=1)
    if rbf_type == "gaussian":
        diff = r_safe[:, None] - cfg["centers"][None, :]
        return jnp.exp(-cfg["width"] * diff**2)
    raise NotImplementedError(rbf_type)
