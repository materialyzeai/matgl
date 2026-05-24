"""ASE calculator backed by the JAX TensorNet/QET inference path.

``JAXPESCalculator`` is a near drop-in twin of ``matgl.ext.ase.PESCalculator``:
it reuses matgl's existing (numpy/CPU) neighbour-list build, pads the edge list
to a bucket capacity for shape-stable XLA compilation, and evaluates a single
jitted ``(E, forces, stress)`` function. It plugs straight into matgl's
``MolecularDynamics`` / ``Relaxer``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from matgl.ext._ase import Atoms2Graph

from ._convert import convert_potential
from ._pad import next_bucket, pad_graph
from ._potential import make_potential_fn


class JAXPESCalculator(Calculator):
    """ASE calculator that runs a converted matgl ``Potential`` under JAX/XLA."""

    implemented_properties = ("energy", "free_energy", "forces", "stress")

    def __init__(
        self,
        potential,
        *,
        stress_unit: str = "GPa",
        stress_weight: float = 1.0,
        use_voigt: bool = False,
        dtype: str = "float32",
        pad_edges: bool = True,
        **kwargs,
    ):
        """Initialize from a (converted-on-the-fly) matgl ``Potential``.

        Args:
            potential: a ``matgl.apps.pes.Potential`` wrapping TensorNet / QET.
            stress_unit: ``"GPa"`` or ``"eV/A3"`` (use the latter for ASE MD/relax).
            stress_weight: extra multiplier applied to the stress.
            use_voigt: emit stress as a Voigt 6-vector instead of a 3x3 matrix.
            dtype: ``"float32"`` (default) or ``"float64"``.
            pad_edges: pad the edge list to a bucket capacity so the jitted
                program is shape-stable across MD steps.
            **kwargs: forwarded to ``ase.calculators.calculator.Calculator``.
        """
        super().__init__(**kwargs)
        params, cfg, extras = convert_potential(potential)
        self._fn = make_potential_fn(params, cfg, extras, num_graphs=1)
        self.element_types = tuple(potential.model.element_types)
        self.cutoff = float(potential.model.cutoff)
        self._a2g = Atoms2Graph(self.element_types, self.cutoff)
        self._pad_edges = pad_edges
        self._dtype = jnp.float64 if dtype == "float64" else jnp.float32

        if stress_unit == "eV/A3":
            cf = units.GPa / (units.eV / units.Angstrom**3)
        elif stress_unit == "GPa":
            cf = 1.0
        else:
            raise ValueError(f"stress_unit must be 'GPa' or 'eV/A3', got {stress_unit!r}")
        self.conversion_factor = cf * stress_weight
        self.use_voigt = use_voigt

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """Build the graph, run the jitted JAX potential, store ASE results."""
        properties = properties or ["energy"]
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        g, lat, _ = self._a2g.get_graph(atoms)
        dt = self._dtype
        lat3 = jnp.asarray(np.asarray(lat[0]), dtype=dt)
        frac = jnp.asarray(g.frac_coords.numpy(), dtype=dt)
        pos = frac @ lat3
        pbc_offset = jnp.asarray(g.pbc_offset.numpy(), dtype=dt)
        z = jnp.asarray(g.node_type.numpy(), dtype=jnp.int32)
        edge_index = jnp.asarray(g.edge_index.numpy(), dtype=jnp.int32)
        n_atoms = z.shape[0]
        n_edges = edge_index.shape[1]
        batch = jnp.zeros(n_atoms, dtype=jnp.int32)
        strain = jnp.zeros((3, 3), dtype=dt)

        if self._pad_edges:
            edge_index, pbc_offset, edge_mask = pad_graph(edge_index, pbc_offset, next_bucket(n_edges))
        else:
            edge_mask = jnp.ones(n_edges, dtype=dt)

        e, f, s = self._fn(pos, strain, frac, lat3, pbc_offset, z, edge_index, batch, edge_mask)

        energy = float(np.asarray(e).reshape(-1)[0])
        self.results.update(energy=energy, free_energy=energy, forces=np.asarray(f, dtype=np.float64))
        stress = np.asarray(s, dtype=np.float64)
        if self.use_voigt:
            stress = full_3x3_to_voigt_6_stress(stress)
        self.results.update(stress=stress * self.conversion_factor)
