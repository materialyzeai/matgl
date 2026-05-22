"""Test structures spanning ~2 to ~1000 atoms for the JAX-vs-PyTorch benchmark."""

from __future__ import annotations

from pymatgen.core import Lattice, Structure

_SI_CONV = Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0.0, 0.0, 0.0]])  # 8 atoms


def make_structures(include_xlarge: bool = False) -> dict[str, Structure]:
    """Return ``name -> Structure`` covering a range of system sizes."""
    out: dict[str, Structure] = {
        "tiny-2": Structure(Lattice.cubic(3.0), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    }
    for label, mult in [("small-64", 2), ("medium-216", 3), ("large-512", 4)]:
        s = _SI_CONV.copy()
        s.make_supercell([mult, mult, mult])
        out[label] = s
    if include_xlarge:
        s = _SI_CONV.copy()
        s.make_supercell([5, 5, 5])
        out["xlarge-1000"] = s
    return out
