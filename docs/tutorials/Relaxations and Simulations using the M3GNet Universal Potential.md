---
layout: default
title: Relaxations and Simulations using the M3GNet Universal Potential.md
nav_exclude: true
---

# Introduction

This notebook demonstrates the use of the pre-trained universal potentials to perform structural relaxations, molecular dynamics simulations and single-point calculations.

Author: Tsz Wai Ko (Kenko)
Email: t1ko@ucsd.edu


```python
from __future__ import annotations

import warnings

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

import matgl
from matgl.ext.ase import MolecularDynamics, PESCalculator, Relaxer

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
```

# Loading the pre-trained M3GNet PES model

We will first load an M3GNet PES model trained on the MatPES-PBE dataset. This can be done with a single line of code. Here we only use M3GNet for demonstration and users can choose other available models.


```python
# Load a pre-trained universal M3GNet potential from Hugging Face.
# To see available models, use matgl.get_available_pretrained_models().
pot = matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2")
```

# Structure Relaxation

To perform structure relaxation, we use the Relaxer class. Here, we demonstrate the relaxation of a simple CsCl structure.


```python
relaxer = Relaxer(potential=pot)
struct = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.5), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
relax_results = relaxer.relax(struct, fmax=0.01)
# extract results
final_structure = relax_results["final_structure"]
final_energy = relax_results["trajectory"].energies[-1]
# print out the final relaxed structure and energy

print(final_structure)
print(f"The final energy is {float(final_energy):.3f} eV.")
```

# Molecular Dynamics

MD simulations are performed with the ASE interface.


```python
ase_adaptor = AseAtomsAdaptor()
# Create ase atom object
atoms = ase_adaptor.get_atoms(final_structure)
# Initialize the velocity according to Maxwell Boltzamnn distribution
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
# Create the MD class
driver = MolecularDynamics(atoms, potential=pot, temperature=300, logfile="md_trial.log")
# Run
driver.run(100)
print(f"The potential energy of CsCl at 300 K after 100 steps is {float(atoms.get_potential_energy()):.3f} eV.")
```

# Single point energy calculation

Perform a single-point calculation for final structure using PESCalculator.


```python
# define the M3GNet calculator
calc = PESCalculator(pot)
# set up the calculator for atoms object
atoms.set_calculator(calc)
print(f"The calculated potential energy is {atoms.get_potential_energy():.3f} eV.")
```