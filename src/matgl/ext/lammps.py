"""LAMMPS interface for MatGL.

Exports a TorchScript-friendly wrapper that LAMMPS pair styles
(``pair_matgl``, ``pair_matgl/kokkos``) load via ``torch::jit::load``.
"""

from __future__ import annotations

from ._lammps import LAMMPSMatGLModel, export_lammps_model

__all__ = ["LAMMPSMatGLModel", "export_lammps_model"]
