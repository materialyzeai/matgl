"""Reference energies/forces/stresses for `in.matgl_si`.

Run this *after* exporting the model with `mgl create-lammps-model` and
*before* running LAMMPS so you have a gold standard to diff against.

    cd lammps/tests
    uv run mgl create-lammps-model -m <model> -o model.pt --dtype float32
    uv run python python_reference.py
    <lammps>/build/lmp -in in.matgl_si

Both runs use the same 4-atom Mo-S supercell at fixed atomic positions, so
results should match within ~1e-5 eV (energy) / 1e-4 eV/Å (forces) /
1e-3 GPa (stress).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from pymatgen.core import Lattice, Structure

import matgl
from matgl.ext.ase import PESCalculator


def _build_structure() -> Structure:
    return Structure(
        Lattice.cubic(4.5),
        ["Mo", "S", "Mo", "S"],
        [
            [0.00, 0.00, 0.00],
            [0.50, 0.50, 0.50],
            [0.50, 0.00, 0.25],
            [0.00, 0.50, 0.75],
        ],
    )


def main() -> int:
    """Print reference energies/forces/stresses for the in.matgl_si test deck."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="MatGL model identifier (HF Hub repo id or local path) — must "
        "be the same one passed to mgl create-lammps-model.",
    )
    args = parser.parse_args()

    structure = _build_structure()
    pot = matgl.load_model(args.model)
    pot.eval()

    calc = PESCalculator(potential=pot, stress_unit="GPa", use_voigt=False)
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor().get_atoms(structure)
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()  # 6-vector in Voigt order (xx,yy,zz,yz,xz,xy)

    print(f"# python reference for `in.matgl_si` (model = {args.model})")
    print(f"energy_eV = {energy:.10e}")
    print("forces_eV_per_A =")
    for row in forces:
        print(f"  {row[0]:.10e}  {row[1]:.10e}  {row[2]:.10e}")
    if isinstance(stress, np.ndarray) and stress.size == 6:
        print("stress_GPa (Voigt: xx yy zz yz xz xy) =")
        print("  " + "  ".join(f"{s:.6e}" for s in stress))
    return 0


if __name__ == "__main__":
    sys.exit(main())
