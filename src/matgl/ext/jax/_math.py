"""JAX ports of the tensor-math and cutoff primitives used by TensorNet / QET.

Every function here is a 1:1 numerical port of its PyTorch counterpart in
``matgl.utils.maths`` / ``matgl.utils.cutoff``. They are pure functions so they
compose cleanly under ``jax.jit`` and ``jax.grad``.
"""

from __future__ import annotations

from math import pi

import jax
import jax.numpy as jnp

# --------------------------------------------------------------------------
# activations
# --------------------------------------------------------------------------


def _softplus2(x):
    """``softplus(x) - log(2)`` — matgl's SoftPlus2."""
    return jax.nn.softplus(x) - jnp.log(2.0)


ACTIVATIONS = {
    "swish": jax.nn.silu,  # matgl ActivationFunction.swish == nn.SiLU
    "silu": jax.nn.silu,
    "tanh": jnp.tanh,
    "sigmoid": jax.nn.sigmoid,
    "softplus": jax.nn.softplus,
    "softplus2": _softplus2,
}


# --------------------------------------------------------------------------
# layer primitives (weights follow the JAX convention: Linear stores W as
# (in, out) and computes ``x @ W + b`` — see _convert.py)
# --------------------------------------------------------------------------


def linear(params: dict, x):
    """Dense layer. ``params`` has key ``w`` (in, out) and optionally ``b``."""
    out = x @ params["w"]
    if "b" in params:
        out = out + params["b"]
    return out


def layer_norm(params: dict, x, eps: float = 1e-5):
    """LayerNorm over the last axis. Matches ``torch.nn.LayerNorm`` (biased var)."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * params["w"] + params["b"]


def tensor_linear(w, t):
    """Apply a bias-free Linear over the ``units`` axis of a (N, units, 3, 3) tensor.

    Mirrors ``linears_tensor[i](t.permute(0,2,3,1)).permute(0,3,1,2)``.
    """
    return jnp.einsum("nuij,uv->nvij", t, w)


# --------------------------------------------------------------------------
# scatter
# --------------------------------------------------------------------------


def scatter_add(x, idx, dim_size: int):
    """Sum rows of ``x`` into ``dim_size`` segments keyed by ``idx`` (axis 0)."""
    return jax.ops.segment_sum(x, idx, num_segments=dim_size, indices_are_sorted=False)


# --------------------------------------------------------------------------
# cutoffs
# --------------------------------------------------------------------------


def cosine_cutoff(r, cutoff: float):
    """Cosine cutoff envelope (matgl.utils.cutoff.cosine_cutoff)."""
    return jnp.where(r <= cutoff, 0.5 * (jnp.cos(pi * r / cutoff) + 1.0), 0.0)


def polynomial_cutoff(r, cutoff: float, exponent: int = 3):
    """Polynomial cutoff envelope (matgl.utils.cutoff.polynomial_cutoff)."""
    coef1 = -(exponent + 1) * (exponent + 2) / 2
    coef2 = exponent * (exponent + 2)
    coef3 = -exponent * (exponent + 1) / 2
    ratio = r / cutoff
    poly = 1 + coef1 * ratio**exponent + coef2 * ratio ** (exponent + 1) + coef3 * ratio ** (exponent + 2)
    return jnp.where(r <= cutoff, poly, 0.0)


# --------------------------------------------------------------------------
# Cartesian-tensor algebra (ports of matgl.utils.maths)
# --------------------------------------------------------------------------


def vector_to_skewtensor(vector):
    """(..., 3) vector -> (..., 3, 3) skew-symmetric tensor."""
    zero = jnp.zeros(vector.shape[:-1], dtype=vector.dtype)
    vx, vy, vz = vector[..., 0], vector[..., 1], vector[..., 2]
    rows = jnp.stack([zero, -vz, vy, vz, zero, -vx, -vy, vx, zero], axis=-1)
    return rows.reshape(*vector.shape[:-1], 3, 3)


def vector_to_symtensor(vector):
    """(..., 3) vector -> (..., 3, 3) symmetric traceless tensor."""
    tensor = vector[..., :, None] * vector[..., None, :]
    eye = jnp.eye(3, dtype=vector.dtype)
    scalars = jnp.mean(jnp.diagonal(tensor, axis1=-2, axis2=-1), axis=-1)[..., None, None] * eye
    sym = 0.5 * (tensor + jnp.swapaxes(tensor, -2, -1))
    return sym - scalars


def decompose_tensor(tensor):
    """(..., 3, 3) -> (scalar-part, skew-part, symmetric-traceless-part)."""
    eye = jnp.eye(3, dtype=tensor.dtype)
    scalars = jnp.mean(jnp.diagonal(tensor, axis1=-2, axis2=-1), axis=-1)[..., None, None] * eye
    skew = 0.5 * (tensor - jnp.swapaxes(tensor, -2, -1))
    traceless = 0.5 * (tensor + jnp.swapaxes(tensor, -2, -1)) - scalars
    return scalars, skew, traceless


def new_radial_tensor(scalars, skew, traceless, f_i, f_a, f_s):
    """Multiply per-edge invariant features into the irreducible tensor components."""
    return (
        f_i[..., None, None] * scalars,
        f_a[..., None, None] * skew,
        f_s[..., None, None] * traceless,
    )


def tensor_norm(tensor):
    """Frobenius norm-squared over the trailing (3, 3) axes."""
    return jnp.sum(tensor**2, axis=(-2, -1))
