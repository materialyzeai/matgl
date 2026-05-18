"""Generate the NaCl training-set artifact for GRACE training-parity tests.

Builds 10 perturbed copies of rocksalt NaCl (the conventional cubic cell
matching MP ``mp-22862``: ``a = 5.6402 Å``, 4 Na + 4 Cl) and labels each
with the ``TensorNet-PES-MatPES-r2SCAN-2025.2`` foundation potential
(energies in eV, forces in eV/Å, stresses in eV/Å³).

The artifact is consumed by

- :data:`tests.conftest.nacl_training_set`
- :mod:`tests.models.test_grace_training_parity`

so that the parity test does not pay the TensorNet inference cost on
every CI run, and so the labels are stable across pytest invocations.

Usage::

    uv run python tests/parity_data/gen_nacl_training_set.py

Re-run this script (and commit the regenerated ``nacl_training_set.json.gz``)
whenever the labelling potential or the perturbation recipe changes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from monty.serialization import dumpfn
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

import matgl
from matgl.ext.ase import PESCalculator

LABELLER_NAME = "TensorNet-PES-MatPES-r2SCAN-2025.2"

# Rocksalt NaCl conventional cubic cell — matches MP ``mp-22862``.
A_NACL = 5.6402
SPECIES = ["Na", "Cl", "Na", "Cl", "Na", "Cl", "Na", "Cl"]
COORDS = [
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.0],
    [0.0, 0.0, 0.5],
    [0.5, 0.0, 0.5],
    [0.0, 0.5, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.0],
]

# Perturbation recipe (deterministic via ``np.random.default_rng(0)``).
N_PERTURB = 9  # +1 unperturbed → 10 configurations total
STRAIN_AMP = 0.02  # ±2% symmetric lattice strain
DISP_AMP = 0.05  # ±0.05 Å per-atom Cartesian displacement
SEED = 0

OUT_PATH = Path(__file__).parent / "nacl_training_set.json.gz"


def _perturbed_structures() -> list[Structure]:
    base = Structure(Lattice.cubic(A_NACL), SPECIES, COORDS)
    rng = np.random.default_rng(SEED)
    structures = [base.copy()]
    for _ in range(N_PERTURB):
        s = base.copy()
        eps = rng.uniform(-STRAIN_AMP, STRAIN_AMP, size=(3, 3))
        eps = 0.5 * (eps + eps.T)
        s.lattice = Lattice((np.eye(3) + eps) @ s.lattice.matrix)
        for site_i in range(len(s)):
            disp = rng.uniform(-DISP_AMP, DISP_AMP, size=3)
            s.translate_sites(site_i, disp, frac_coords=False, to_unit_cell=False)
        structures.append(s)
    return structures


def main() -> None:
    pot = matgl.load_model(LABELLER_NAME)
    calc = PESCalculator(pot, stress_unit="eV/A3")
    structures = _perturbed_structures()

    samples: list[dict] = []
    for s in structures:
        atoms = AseAtomsAdaptor.get_atoms(s)
        atoms.calc = calc
        samples.append(
            {
                "structure": s.as_dict(),
                "energy": float(atoms.get_potential_energy()),
                "forces": atoms.get_forces().astype("float64").tolist(),
                "stress": atoms.get_stress(voigt=False).astype("float64").tolist(),
            }
        )

    payload = {
        "labeller": LABELLER_NAME,
        "n_structures": len(samples),
        "strain_amp": STRAIN_AMP,
        "disp_amp": DISP_AMP,
        "seed": SEED,
        "samples": samples,
    }
    dumpfn(payload, str(OUT_PATH))
    print(f"wrote {OUT_PATH} with {len(samples)} samples")


if __name__ == "__main__":
    main()
