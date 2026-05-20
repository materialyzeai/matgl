"""Interatomic potential wrapper for matgl graph models.

This module exposes a single class -- :class:`Potential` -- which turns an
energy-predicting graph model into a full interatomic potential. From a
scalar per-graph energy, :class:`Potential` derives forces and stress via
PyTorch autograd (and optionally the Hessian, partial charges, and magnetic
moments if the wrapped model produces them).

Units and conventions (matching the README and the rest of matgl):

* energies in eV (per structure, not per atom),
* forces in eV/Å,
* stresses in **GPa with the compressive-negative convention** (VASP's
  kbar output multiplied by ``-0.1``),
* coordinates in Å (after multiplication by the supplied lattice).

See :class:`Potential` for the full set of options and the variable return
tuple shape, which depends on which ``calc_*`` flags are enabled.
"""

from __future__ import annotations

from ._pes import Potential

__all__ = ["Potential"]
