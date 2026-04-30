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
- Training: 33 meV/atom, 121 meV/A, 0.602 GPa
- Validation: 36 meV/atom, 138 meV/A, 0.695 GPa
- Test: 36 meV/atom, 148 meV/A, 0.700 GPa

# References

```txt
Aaron Kaplan, Runze Liu, Ji Qi, Tsz Wai Ko, Bowen Deng, Gerbrand Ceder, Kristin A. Persson, Shyue Ping Ong.
A foundational potential energy surface dataset for materials. Submitted.
```
