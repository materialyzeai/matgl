## MatQ Dataset

The **MatQ dataset** comprises both **near-equilibrium** and **out-of-equilibrium** structures designed to provide broad coverage of the potential energy surface (PES) of crystalline materials.

### Structure Generation

The initial pool of **6,652,874 near-equilibrium structures** was generated from Materials Project crystals by applying strains of **±2%, ±4%, and ±6%** along all crystallographic directions.

The **out-of-equilibrium structures** were obtained from **1000 K and 3000 K NVT/NPT molecular dynamics (MD) trajectories** in the **OMat24 validation set**.

To reduce redundancy while maintaining coverage of the configuration space, the **DIRECT (Dimensionality-Reduced Encoded Clusters with sTratified sampling)** method was applied. This approach selects representative structures using dimensionality reduction and stratified sampling, enabling efficient exploration of the configuration space while minimizing overlap.

Using DIRECT sampling, **60,000 structures** were selected from both the near-equilibrium and out-of-equilibrium pools. This significantly reduces the computational cost of generating high-quality reference data while preserving the diversity of structural configurations.

### DFT Calculations

All structures were evaluated using **spin-polarized density functional theory (DFT)** calculations with the **Vienna Ab initio Simulation Package (VASP)**.

The calculations used the **Perdew–Burke–Ernzerhof (PBE)** generalized gradient approximation (GGA) to describe exchange–correlation interactions.

Input files were generated using the **`MatPESStaticSet`** workflow implemented in **pymatgen**, which has been carefully benchmarked to ensure well-converged potential energy surface properties.

The main computational parameters are:

- **Plane-wave energy cutoff:** 680 eV
- **k-point spacing:** 0.35 Å⁻¹
- **Electronic convergence criterion:** 1×10⁻⁵ eV

Atomic charges were computed using the **DDEC6 charge partitioning scheme** implemented in **Chargemol (version 09_26_2017)** from the DFT charge densities.

### Final Dataset

The final **MatQ dataset contains 114,445 structures**, after excluding:

- unconverged DFT calculations
- structures with extremely large force components (**|Fₓ, Fᵧ, F_z| > 50 eV/Å**)

The units of length, energy, force, stress, and charge are Å, eV, eV/Å, eV/Å³, and e, respectively.
