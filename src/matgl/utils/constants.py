"""Centralized physical and numerical constants for matgl.

This module is the single source of truth for the scalar physical constants and
unit conversions used across matgl. It deliberately avoids importing torch or any
matgl submodule so that it can be imported anywhere without circular imports.
"""

from __future__ import annotations

# Coulomb constant in eV·Å/e² (used for electrostatic energy in QET / charge
# equilibration).
COULOMB_CONSTANT = 14.399645478425668

# 1 eV/Å³ = 160.21766208 GPa. Used to convert stress (autograd of energy in eV
# w.r.t. strain, divided by volume in Å³) into GPa.
EV_PER_ANG3_TO_GPA = 160.21766208
