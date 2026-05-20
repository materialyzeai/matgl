"""Electrostatics module for MatGL.

Exposes :class:`LinearQeq` charge-equilibration solver and
:class:`ElectrostaticPotential` aggregator used by the QET model.
"""

from __future__ import annotations

from ._elec_pot import ElectrostaticPotential
from ._fast_qeq import LinearQeq

__all__ = ["ElectrostaticPotential", "LinearQeq"]
