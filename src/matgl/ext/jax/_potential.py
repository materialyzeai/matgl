"""JAX port of ``matgl.apps.pes.Potential`` (inference path: E, forces, stress).

Reproduces the strain-based stress derivation of ``matgl.apps._pes.Potential``:
a symbolic strain ``eps`` is introduced, the lattice becomes ``lat @ (I + eps)``,
and ``stress = (dE/d eps) / V``. Energy and gradients are produced by a single
``jax.value_and_grad`` under one ``jax.jit``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._qet import qet_energy, qet_energy_and_charges
from ._tensornet import scatter_add, tensornet_energy

# 1 eV/A^3 = 160.21766208 GPa (matches matgl.apps._pes.EV_PER_ANG3_TO_GPA).
EV_PER_ANG3_TO_GPA = 160.21766208


def make_potential_fn(params, cfg, extras, num_graphs: int = 1, energy_model=None, with_charges: bool = False):
    """Build a jitted ``(E, forces, stress)`` function for a converted model.

    The returned callable takes ``(pos, strain, frac_coords, lat3, pbc_offset, z,
    edge_index, batch, edge_mask)`` and returns energy (eV), forces (eV/A) and the
    3x3 stress (GPa). ``cfg`` / ``num_graphs`` / denorm constants are closed over.

    With ``with_charges=True`` (converted QET models only) the callable takes one
    extra trailing argument, a per-batch ``total_charge`` scalar, and additionally
    returns the QEq per-atom charges: ``(E, forces, stress, charges)``.
    ``total_charge`` is traced, so varying it does not trigger recompilation.

    Forces and stress correspond exactly to ``Potential.forward``'s two autograd
    leaves: ``pos`` carries the force gradient, while ``strain`` deforms *both*
    the PBC offshift and the atomic positions (via ``frac_coords``), so the stress
    captures the full position-deformation term.
    """
    is_qet = energy_model is None and cfg.get("model_type") == "QET"
    if with_charges and not is_qet:
        raise ValueError("with_charges=True requires a converted QET model (charges come from its QEq solve).")
    if energy_model is None:
        energy_model = qet_energy if is_qet else tensornet_energy
    data_mean = extras["data_mean"]
    data_std = extras["data_std"]
    element_refs = extras.get("element_refs")

    def total_energy(pos, strain, frac_coords, lat3, pbc_offset, z, edge_index, batch, edge_mask, total_charge):
        eye = jnp.eye(3, dtype=pos.dtype)
        lattice = lat3 @ (eye + strain)
        pbc_offshift = pbc_offset @ lattice
        # pos == frac_coords @ lat3 (strain == 0); the correction term makes
        # `strain` deform positions too, matching torch's downstream g.pos node.
        pos_corr = pos + frac_coords @ (lattice - lat3)
        if is_qet:
            e_model, charges = qet_energy_and_charges(
                params, cfg, z, pos_corr, edge_index, pbc_offshift, batch, num_graphs, edge_mask, total_charge
            )
        else:
            e_model = energy_model(params, cfg, z, pos_corr, edge_index, pbc_offshift, batch, num_graphs, edge_mask)
            charges = None
        e = data_std * e_model + data_mean
        if element_refs is not None:
            ref = element_refs[z]
            e = e + (jnp.sum(ref) if num_graphs == 1 else scatter_add(ref, batch, num_graphs))
        return e, charges

    def energy_sum(pos, strain, frac_coords, lat3, pbc_offset, z, edge_index, batch, edge_mask, total_charge):
        e, charges = total_energy(
            pos, strain, frac_coords, lat3, pbc_offset, z, edge_index, batch, edge_mask, total_charge
        )
        return jnp.sum(e), charges

    @jax.jit
    def run_aux(pos, strain, frac_coords, lat3, pbc_offset, z, edge_index, batch, edge_mask, total_charge):
        ((e, charges), (d_pos, d_strain)) = jax.value_and_grad(energy_sum, argnums=(0, 1), has_aux=True)(
            pos, strain, frac_coords, lat3, pbc_offset, z, edge_index, batch, edge_mask, total_charge
        )
        lattice = lat3 @ (jnp.eye(3, dtype=pos.dtype) + strain)
        volume = jnp.abs(jnp.linalg.det(lattice))
        forces = -d_pos
        stress = d_strain * (EV_PER_ANG3_TO_GPA / volume)
        return e, forces, stress, charges

    if with_charges:
        return run_aux

    def run(pos, strain, frac_coords, lat3, pbc_offset, z, edge_index, batch, edge_mask):
        total_charge = jnp.asarray(cfg.get("total_charge", 0.0), dtype=pos.dtype)
        e, forces, stress, _ = run_aux(
            pos, strain, frac_coords, lat3, pbc_offset, z, edge_index, batch, edge_mask, total_charge
        )
        return e, forces, stress

    return run
