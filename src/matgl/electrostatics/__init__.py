"""Electrostatics module for MatGL.

Re-exports the active backend's :class:`LinearQeq` charge-equilibration solver
and :class:`ElectrostaticPotential` aggregator, controlled by
``matgl.config.BACKEND`` (``"DGL"`` or ``"PYG"``).
"""

from __future__ import annotations

from matgl.config import BACKEND

if BACKEND == "DGL":
    from ._elec_pot_dgl import ElectrostaticPotential
    from ._fast_qeq_dgl import LinearQeq
else:
    from ._elec_pot_pyg import ElectrostaticPotential  # type: ignore[assignment]
    from ._fast_qeq_pyg import LinearQeq  # type: ignore[assignment]

__all__ = ["ElectrostaticPotential", "LinearQeq"]
