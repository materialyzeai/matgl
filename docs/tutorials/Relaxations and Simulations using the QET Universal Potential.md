---
layout: default
title: Relaxations and Simulations using the QET Universal Potential.md
nav_exclude: true
---

# Introduction

This notebook demonstrates the use of the charge-aware pre-trained universal potentials to perform structural relaxations, molecular dynamics simulations and single-point calculations.

Author: Tsz Wai Ko (Kenko)
Email: t1ko@ucsd.edu


```python
from __future__ import annotations

import warnings

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

import matgl
from matgl.ext._ase_dgl import MolecularDynamics, PESCalculator, Relaxer

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
```

# Loading the pre-trained QET PES model

We will first load the QET PES model, which is trained on the MatQ dataset. This can be done with a single line of code. Here we only use QET for demonstration and users can choose other available models.


```python
# You can load any pretrained potentials stored in the 'pretrained_models' directory
# To see available models, use get_available_pretrained_models()
pot = matgl.load_model("../pretrained_models/QET-MatQ-PES")
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

    Full Formula (Cs1 Cl1)
    Reduced Formula: CsCl
    abc   :   4.207706   4.207706   4.207706
    angles:  90.000000  90.000000  90.000000
    pbc   :       True       True       True
    Sites (2)
      #  SP      a    b    c    final_charge
    ---  ----  ---  ---  ---  --------------
      0  Cs    0    0    0          0.819602
      1  Cl    0.5  0.5  0.5       -0.819602
    The final energy is -6.532 eV.


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

    The potential energy of CsCl at 300 K after 100 steps is -6.407 eV.


# Single point energy calculation

Perform a single-point calculation for final structure using PESCalculator.


```python
# define the M3GNet calculator
calc = PESCalculator(pot)
# set up the calculator for atoms object
atoms.set_calculator(calc)
print(f"The calculated potential energy is {atoms.get_potential_energy():.3f} eV.")
# you can also calculate charges as well
print(f"The calculated charges is {atoms.get_charges()}.")
```

    The calculated potential energy is -6.407 eV.
    The calculated charges is [ 0.8030549  -0.80305505].
