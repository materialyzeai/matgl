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
- Training: 40 meV/atom, 155 meV/A, 0.734 GPa
- Validation: 45 meV/atom, 177 meV/A, 0.898 GPa
- Test: 45 meV/atom, 181 meV/A, 0.888 GPa

# References

```txt
Aaron Kaplan, Runze Liu, Ji Qi, Tsz Wai Ko, Bowen Deng, Gerbrand Ceder, Kristin A. Persson, Shyue Ping Ong.
A foundational potential energy surface dataset for materials. Submitted.
```
