"""Edge padding / bucketing for static-shape XLA compilation.

A neighbour list has a variable edge count ``E`` that changes every MD step,
while the atom count ``N`` is fixed for a trajectory. Padding ``E`` up to a
bucket capacity keeps the jitted program shape-stable, so it is compiled once
and reused (at most ~log2 recompilations as ``E`` crosses bucket boundaries).
"""

from __future__ import annotations

import jax.numpy as jnp

# Coarse capacity ladder — keeps the number of distinct compilations small.
_BUCKETS = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]


def next_bucket(n_edges: int) -> int:
    """Smallest bucket capacity >= ``n_edges``."""
    for b in _BUCKETS:
        if b >= n_edges:
            return b
    b = _BUCKETS[-1]
    while b < n_edges:
        b *= 2
    return b


def pad_graph(edge_index, pbc_offset, e_cap: int):
    """Pad ``edge_index`` / ``pbc_offset`` to capacity ``e_cap``.

    Padded edges are sentinel self-loops on atom 0 with PBC image ``[1, 0, 0]``;
    the returned boolean ``edge_mask`` is ``1`` for real edges and ``0`` for
    padding. Callers must multiply per-edge contributions by ``edge_mask`` before
    any scatter so padded edges contribute exactly zero.

    The non-zero PBC image is deliberate: it gives padded edges a non-zero bond
    vector (``= lattice[0]``), so ``grad(||bond_vec||)`` stays finite. A zero
    bond vector would make the norm gradient ``0/0`` and the masked-out ``0 * NaN``
    would poison atom 0's force.
    """
    e = edge_index.shape[1]
    if e_cap < e:
        raise ValueError(f"e_cap ({e_cap}) smaller than edge count ({e})")
    pad = e_cap - e
    edge_index = jnp.concatenate([edge_index, jnp.zeros((2, pad), edge_index.dtype)], axis=1)
    pad_offset = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 0.0], dtype=pbc_offset.dtype), (pad, 3))
    pbc_offset = jnp.concatenate([pbc_offset, pad_offset], axis=0)
    edge_mask = jnp.concatenate([jnp.ones(e), jnp.zeros(pad)])
    return edge_index, pbc_offset, edge_mask
