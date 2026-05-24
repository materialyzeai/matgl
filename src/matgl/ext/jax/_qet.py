"""Functional JAX port of the PyG QET model.

QET = TensorNet feature extractor + a charge-equilibration head:

* per-atom electronegativity ``chi`` / hardness / Gaussian width ``sigma``;
* a closed-form charge-equilibration solve (``LinearQeq`` -- O(N), no linear
  system, just segment sums);
* a Gaussian-smeared Coulomb electrostatic potential;
* a LayerNorm + gated readout over ``[node_feat, charge, elec_pot, magmom?]``.

The TensorNet ``forward_features`` is reused verbatim.
"""

from __future__ import annotations

from math import sqrt

import jax
import jax.numpy as jnp

from matgl.config import COULOMB_CONSTANT

from ._math import layer_norm, polynomial_cutoff, scatter_add
from ._tensornet import forward_features, gated_mlp, mlp, pair_vector_and_distance

_COULOMB = float(COULOMB_CONSTANT)
_QEQ_DENOM_EPS = 1e-12
_INV_SQRT2 = 1.0 / sqrt(2.0)


def linear_qeq(chi, hardness, batch, num_graphs: int, total_charge: float):
    """Closed-form charge equilibration (port of ``LinearQeq.forward``).

    ``q_i = -chi_i/eta_i + (1/eta_i) (Q + sum_j chi_j/eta_j) / (sum_j 1/eta_j)``.
    """
    hardness_inv = 1.0 / hardness
    chi_hardness_inv = chi * hardness_inv
    if num_graphs == 1:
        sum_hi = jnp.maximum(jnp.sum(hardness_inv), _QEQ_DENOM_EPS)
        sum_chi_hi = jnp.sum(chi_hardness_inv)
        return -chi * hardness_inv + hardness_inv * (total_charge + sum_chi_hi) / sum_hi
    sum_hi = jnp.maximum(scatter_add(hardness_inv, batch, num_graphs)[batch], _QEQ_DENOM_EPS)
    sum_chi_hi = scatter_add(chi_hardness_inv, batch, num_graphs)[batch]
    sum_q = jnp.full((num_graphs,), total_charge, dtype=chi.dtype)[batch]
    return -chi * hardness_inv + hardness_inv * (sum_q + sum_chi_hi) / sum_hi


def electrostatic_potential(charge, sigma, pos, edge_index, pbc_offshift, cutoff: float, edge_mask):
    """Gaussian-smeared Coulomb potential (port of ``ElectrostaticPotential``)."""
    src, dst = edge_index[0], edge_index[1]
    _, bond_dist = pair_vector_and_distance(pos, edge_index, pbc_offshift)
    r_safe = jnp.where(edge_mask > 0, bond_dist, 1.0)
    gamma = jnp.sqrt(sigma[src] ** 2 + sigma[dst] ** 2)
    edge_msg = (
        charge[dst] * jax.scipy.special.erf(r_safe * _INV_SQRT2 / gamma) * polynomial_cutoff(r_safe, cutoff) / r_safe
    )
    return scatter_add(edge_msg * _COULOMB * edge_mask, src, charge.shape[0])


def qet_energy(params, cfg, z, pos, edge_index, pbc_offshift, batch, num_graphs, edge_mask):
    """Per-graph QET energy (raw model output, before Potential denorm)."""
    x = forward_features(params, cfg, z, pos, edge_index, pbc_offshift, edge_mask)

    # chi / hardness readouts use fixed activations (SiLU / Softplus), independent
    # of the model's activation_type — matching QET.__init__.
    chi = mlp(params["chi_readout"], x, jax.nn.silu, activate_last=True).reshape(-1)
    if cfg["is_hardness_envs"]:
        hardness = mlp(params["hardness_readout"], x, jax.nn.softplus, activate_last=True).reshape(-1)
    else:
        hardness = params["hardness_readout"][z]
    sigma = params["sigma"][z]

    charge = linear_qeq(chi, hardness, batch, num_graphs, cfg.get("total_charge", 0.0))
    elec_pot = electrostatic_potential(charge, sigma, pos, edge_index, pbc_offshift, cfg["cutoff"], edge_mask)

    feats = [x, charge[:, None], elec_pot[:, None]]
    if cfg["include_magmom"]:
        magmom = mlp(params["magmom_readout"], x, jax.nn.silu, activate_last=False).reshape(-1)
        feats.append(magmom[:, None])
    node_feat = layer_norm(params["norm"], jnp.concatenate(feats, axis=1))
    atomic = gated_mlp(params["final_layer"], node_feat, activate_last=False).reshape(-1)

    if num_graphs == 1:
        return jnp.sum(atomic)[None]
    return scatter_add(atomic, batch, num_graphs)
