# Description

This model is a TensorNet universal potential for 89 elements of the periodic table. It has broad applications in the
dynamic simulations of materials.

# Training dataset

MatPESv2025.1-PBE: 300K MD simulations and Materials Project ground state calculations.
- Training set size: 391240
- Validation set size: 21735
- Test set size: 21737

# Performance metrics

## Training and validation errors

MAEs of energies, forces and stresses, respectively
- Training: 41 meV/atom, 116 meV/A, 0.526 GPa
- Validation: 42 meV/atom, 131 meV/A, 0.629 GPa
- Test: 42 meV/atom, 146 meV/A, 0.627 GPa

# References

```txt
Aaron Kaplan, Runze Liu, Ji Qi, Tsz Wai Ko, Bowen Deng, Gerbrand Ceder, Kristin A. Persson, Shyue Ping Ong.
A foundational potential energy surface dataset for materials. Submitted.
```
