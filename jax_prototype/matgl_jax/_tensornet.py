"""Functional JAX port of the PyG TensorNet forward pass (inference path).

This reproduces ``matgl.models.TensorNet.forward`` (the ``use_warp=False`` branch)
plus the embedding / interaction layers and the extensive / intensive readouts.
Everything is a pure function of ``(params, geometry)`` so it composes under
``jax.jit`` and ``jax.grad``; ``cfg`` carries the architecture (Python scalars +
non-learned constant arrays) and is closed over at fn-build time.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._basis import bond_expansion
from ._math import (
    ACTIVATIONS,
    cosine_cutoff,
    decompose_tensor,
    layer_norm,
    linear,
    new_radial_tensor,
    scatter_add,
    tensor_linear,
    tensor_norm,
    vector_to_skewtensor,
    vector_to_symtensor,
)

# --------------------------------------------------------------------------
# generic MLP primitives
# --------------------------------------------------------------------------


def mlp(layers: list, x, activation, activate_last: bool):
    """Plain MLP — Linear layers with ``activation`` between them."""
    for i, lp in enumerate(layers):
        x = linear(lp, x)
        if i < len(layers) - 1 or activate_last:
            x = activation(x)
    return x


def gated_mlp(params: dict, x, activate_last: bool = False):
    """Gated MLP — ``value(x) * sigmoid_gate(x)``. Internal activation is SiLU."""
    silu = jax.nn.silu
    v = x
    vlayers = params["value"]
    for i, lp in enumerate(vlayers):
        v = linear(lp, v)
        if i < len(vlayers) - 1 or activate_last:
            v = silu(v)
    g = x
    glayers = params["gate"]
    for i, lp in enumerate(glayers):
        g = linear(lp, g)
        g = silu(g) if i < len(glayers) - 1 else jax.nn.sigmoid(g)
    return v * g


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def pair_vector_and_distance(pos, edge_index, pbc_offshift):
    """Port of ``compute_pair_vector_and_distance``."""
    src, dst = edge_index[0], edge_index[1]
    bond_vec = pos[dst] + pbc_offshift - pos[src]
    bond_dist = jnp.linalg.norm(bond_vec, axis=1)
    return bond_vec, bond_dist


# --------------------------------------------------------------------------
# TensorEmbedding
# --------------------------------------------------------------------------


def tensor_embedding(params, cfg, z, edge_index, edge_attr, edge_weight, edge_vec, edge_mask):
    """Port of ``matgl.layers._embedding_pyg.TensorEmbedding.forward``."""
    act = ACTIVATIONS[cfg["activation"]]
    p = params["tensor_embedding"]
    n_nodes = z.shape[0]
    src, dst = edge_index[0], edge_index[1]

    x = p["emb"][z]  # (N, units)

    cut = cosine_cutoff(edge_weight, cfg["cutoff"])[:, None]
    w1 = linear(p["distance_proj1"], edge_attr) * cut
    w2 = linear(p["distance_proj2"], edge_attr) * cut
    w3 = linear(p["distance_proj3"], edge_attr) * cut

    norm = jnp.clip(jnp.linalg.norm(edge_vec, axis=1, keepdims=True), min=1e-6)
    edge_vec = edge_vec / norm

    eye = jnp.eye(3, dtype=edge_vec.dtype)[None, None]
    iij, aij, sij = new_radial_tensor(
        eye,
        vector_to_skewtensor(edge_vec)[:, None],
        vector_to_symtensor(edge_vec)[:, None],
        w1,
        w2,
        w3,
    )  # each (E, units, 3, 3)

    zij = jnp.concatenate([x[src], x[dst]], axis=-1)
    zij = linear(p["emb2"], zij)[..., None, None]  # (E, units, 1, 1)
    m = edge_mask[:, None, None, None]
    scalars = scatter_add(zij * iij * m, src, n_nodes)
    skew = scatter_add(zij * aij * m, src, n_nodes)
    traceless = scatter_add(zij * sij * m, src, n_nodes)

    norm = tensor_norm(scalars + skew + traceless)
    norm = layer_norm(p["init_norm"], norm)
    scalars = tensor_linear(p["linears_tensor"][0]["w"], scalars)
    skew = tensor_linear(p["linears_tensor"][1]["w"], skew)
    traceless = tensor_linear(p["linears_tensor"][2]["w"], traceless)
    for lp in p["linears_scalar"]:
        norm = act(linear(lp, norm))
    norm = norm.reshape(norm.shape[0], cfg["units"], 3)
    scalars, skew, traceless = new_radial_tensor(scalars, skew, traceless, norm[..., 0], norm[..., 1], norm[..., 2])
    return scalars + skew + traceless  # (N, units, 3, 3)


# --------------------------------------------------------------------------
# TensorNetInteraction
# --------------------------------------------------------------------------


def interaction(params, cfg, edge_index, edge_weight, edge_attr, x, edge_mask):
    """Port of ``matgl.layers._graph_convolution_pyg.TensorNetInteraction.forward``."""
    act = ACTIVATIONS[cfg["activation"]]
    units = cfg["units"]
    n_nodes = x.shape[0]
    src, dst = edge_index[0], edge_index[1]

    cut = cosine_cutoff(edge_weight, cfg["cutoff"])[:, None]
    for lp in params["linears_scalar"]:
        edge_attr = act(linear(lp, edge_attr))
    edge_attr = (edge_attr * cut).reshape(edge_attr.shape[0], units, 3)

    x = x / (tensor_norm(x) + 1)[..., None, None]
    scalars, skew, traceless = decompose_tensor(x)
    scalars = tensor_linear(params["linears_tensor"][0]["w"], scalars)
    skew = tensor_linear(params["linears_tensor"][1]["w"], skew)
    traceless = tensor_linear(params["linears_tensor"][2]["w"], traceless)
    y = scalars + skew + traceless

    # message: gather destination tensors, scale by per-edge radial features
    mi, ma, ms = new_radial_tensor(
        scalars[dst], skew[dst], traceless[dst], edge_attr[..., 0], edge_attr[..., 1], edge_attr[..., 2]
    )
    m = edge_mask[:, None, None, None]
    msg = scatter_add(mi * m, src, n_nodes) + scatter_add(ma * m, src, n_nodes) + scatter_add(ms * m, src, n_nodes)

    if cfg["group"] == "O(3)":
        a = jnp.matmul(msg, y)
        b = jnp.matmul(y, msg)
        scalars, skew, traceless = decompose_tensor(a + b)
    elif cfg["group"] == "SO(3)":
        b = jnp.matmul(y, msg)
        scalars, skew, traceless = decompose_tensor(2 * b)
    else:
        raise ValueError("group must be 'O(3)' or 'SO(3)'")

    normp1 = (tensor_norm(scalars + skew + traceless) + 1)[..., None, None]
    scalars = tensor_linear(params["linears_tensor"][3]["w"], scalars / normp1)
    skew = tensor_linear(params["linears_tensor"][4]["w"], skew / normp1)
    traceless = tensor_linear(params["linears_tensor"][5]["w"], traceless / normp1)

    dx = scalars + skew + traceless
    return x + dx + jnp.matmul(dx, dx)


# --------------------------------------------------------------------------
# full forward
# --------------------------------------------------------------------------


def forward_features(params, cfg, z, pos, edge_index, pbc_offshift, edge_mask):
    """Run TensorNet feature extraction up to the per-atom ``readout`` features."""
    bond_vec, bond_dist = pair_vector_and_distance(pos, edge_index, pbc_offshift)
    # Padded edges have bond_dist == 0; substitute a dummy distance so the
    # sin(x)/x radial terms never hit 0/0. Their contribution is killed by the
    # edge_mask multiply before every scatter, so the dummy value is irrelevant.
    r_safe = jnp.where(edge_mask > 0, bond_dist, 1.0)
    edge_attr = bond_expansion(cfg["basis"], r_safe)

    x = tensor_embedding(params, cfg, z, edge_index, edge_attr, r_safe, bond_vec, edge_mask)
    for layer_params in params["layers"]:
        x = interaction(layer_params, cfg, edge_index, r_safe, edge_attr, x, edge_mask)

    scalars, skew, traceless = decompose_tensor(x)
    x = jnp.concatenate([tensor_norm(scalars), tensor_norm(skew), tensor_norm(traceless)], axis=-1)
    x = layer_norm(params["out_norm"], x)
    return linear(params["linear"], x)  # (N, units)


def tensornet_energy(params, cfg, z, pos, edge_index, pbc_offshift, batch, num_graphs, edge_mask):
    """Per-graph TensorNet energy (raw model output, before Potential denorm)."""
    x = forward_features(params, cfg, z, pos, edge_index, pbc_offshift, edge_mask)

    if not cfg["is_intensive"]:
        atomic = gated_mlp(params["final_layer"], x, activate_last=False).reshape(-1)
        if num_graphs == 1:
            return jnp.sum(atomic)[None]
        return scatter_add(atomic, batch, num_graphs)

    # intensive: weighted-atom readout then a plain MLP head
    act = ACTIVATIONS[cfg["activation"]]
    rp = params["readout"]
    updated = mlp(rp["mlp"], x, act, activate_last=True)
    weights = jax.nn.sigmoid(mlp(rp["weight"], x, act, activate_last=False))
    wsum = scatter_add(weights, batch, num_graphs)
    factor = weights / jnp.clip(wsum[batch], min=1e-8)
    node_vec = scatter_add(factor * updated, batch, num_graphs)
    return mlp(params["final_layer"], node_vec, act, activate_last=False).reshape(-1)
