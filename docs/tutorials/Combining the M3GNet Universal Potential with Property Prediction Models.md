---
layout: default
title: Combining the M3GNet Universal Potential with Property Prediction Models.md
nav_exclude: true
---

# Introduction

There may be instances where you do not have access to a DFT relaxed structure. For instance, you may have a generated hypothetical structure or a structure obtained from an experimental source. In this notebook, we demonstrate how you can use the M3GNet universal potential to relax a crystal prior to property predictions.

This provides a pathway to "DFT-free" property predictions using ML models. It should be cautioned that this is not a substitute for DFT and errors can be expected. But it is sufficiently useful in some cases as a pre-screening tool for massive scale exploration of materials.


```python
from __future__ import annotations

import warnings

import torch
from pymatgen.core import Lattice, Structure
from pymatgen.ext.matproj import MPRester

import matgl
from matgl.ext.ase import Relaxer

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
```

For the purposes of demonstration, we will use the perovskite SrTiO3 (STO). We will create a STO with an arbitrary lattice parameter of 4.5 A.


```python
sto = Structure.from_spacegroup(
    "Pm-3m", Lattice.cubic(4.5), ["Sr", "Ti", "O"], [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0]]
)
print(sto)
```

    Full Formula (Sr1 Ti1 O3)
    Reduced Formula: SrTiO3
    abc   :   4.500000   4.500000   4.500000
    angles:  90.000000  90.000000  90.000000
    pbc   :       True       True       True
    Sites (5)
      #  SP      a    b    c
    ---  ----  ---  ---  ---
      0  Sr    0    0    0
      1  Ti    0.5  0.5  0.5
      2  O     0.5  0    0.5
      3  O     0    0.5  0.5
      4  O     0.5  0.5  0


As a ground truth reference, we will also obtain the Materials Project DFT calculated SrTiO3 structure (mpid: mp-???) using pymatgen's interface to the Materials API.


```python
mpr = MPRester()
doc = mpr.summary.search(material_ids=["mp-5229"])[0]
sto_dft = doc["structure"]
sto_dft_bandgap = doc["band_gap"]
sto_dft_forme = doc["formation_energy_per_atom"]
```

# Relaxing the crystal


```python
pot = matgl.load_model("M3GNet-PES-MatPES-PBE-2025.2")
```


    model.pt:   0%|          | 0.00/3.35k [00:00<?, ?B/s]



    state.pt:   0%|          | 0.00/1.23M [00:00<?, ?B/s]



    model.json:   0%|          | 0.00/5.33k [00:00<?, ?B/s]



```python
relaxer = Relaxer(potential=pot)
relax_results = relaxer.relax(sto, fmax=0.01)
relaxed_sto = relax_results["final_structure"]
print(relaxed_sto)
```

    Full Formula (Sr1 Ti1 O3)
    Reduced Formula: SrTiO3
    abc   :   3.943746   3.943774   3.943905
    angles:  90.001964  89.998498  89.997854
    pbc   :       True       True       True
    Sites (5)
      #  SP            a          b          c
    ---  ----  ---------  ---------  ---------
      0  Sr     4.6e-05    0.0001     5.2e-05
      1  Ti     0.500172   0.500084   0.500085
      2  O      0.499999  -4.8e-05    0.499941
      3  O     -0.000183   0.499879   0.499924
      4  O      0.499967   0.499986  -1e-06


You can compare the lattice parameter with the DFT one from MP. Quite clearly, the M3GNet universal potential does a reasonably good job on relaxing STO.


```python
print(sto_dft)
```

    Full Formula (Sr1 Ti1 O3)
    Reduced Formula: SrTiO3
    abc   :   3.912701   3.912701   3.912701
    angles:  90.000000  90.000000  90.000000
    pbc   :       True       True       True
    Sites (5)
      #  SP       a     b     c    magmom
    ---  ----  ----  ----  ----  --------
      0  Sr    -0    -0    -0          -0
      1  Ti     0.5   0.5   0.5        -0
      2  O      0.5  -0     0.5         0
      3  O      0.5   0.5  -0           0
      4  O     -0     0.5   0.5         0


# Formation energy prediction

To demonstrate the difference between making predictions with a unrelaxed vs a relaxed crystal, we will load the M3GNet formation energy model.


```python
# Load the pre-trained MEGNet formation energy model.
model = matgl.load_model("M3GNet-Eform-MP-2018.6.1")
eform_sto = model.predict_structure(sto)
eform_relaxed_sto = model.predict_structure(relaxed_sto)

print(f"The predicted formation energy for the unrelaxed SrTiO3 is {float(eform_sto):.3f} eV/atom.")
print(f"The predicted formation energy for the relaxed SrTiO3 is {float(eform_relaxed_sto):.3f} eV/atom.")
print(f"The Materials Project formation energy for DFT-relaxed SrTiO3 is {sto_dft_forme:.3f} eV/atom.")
```


    model.pt:   0%|          | 0.00/2.53k [00:00<?, ?B/s]



    state.pt:   0%|          | 0.00/1.61M [00:00<?, ?B/s]



    model.json:   0%|          | 0.00/3.63k [00:00<?, ?B/s]


    The predicted formation energy for the unrelaxed SrTiO3 is -1.288 eV/atom.
    The predicted formation energy for the relaxed SrTiO3 is -1.991 eV/atom.
    The Materials Project formation energy for DFT-relaxed SrTiO3 is -3.551 eV/atom.


The predicted formation energy from the M3GNet relaxed STO is in fairly good agreement with the DFT value.

# Band gap prediction

We will repeat the above exericse but for the band gap.


```python
model = matgl.load_model("MEGNet-BandGap-mfi-MP-2019.4.1")

# For multi-fidelity models, we need to define graph label ("0": PBE, "1": GLLB-SC, "2": HSE, "3": SCAN)
for i, method in ((0, "PBE"), (1, "GLLB-SC"), (2, "HSE"), (3, "SCAN")):
    graph_attrs = torch.tensor([i])
    bandgap_sto = model.predict_structure(structure=sto, state_attr=graph_attrs)
    bandgap_relaxed_sto = model.predict_structure(structure=relaxed_sto, state_attr=graph_attrs)

    print(f"{method} band gap")
    print(f"\tUnrelaxed STO = {float(bandgap_sto):.2f} eV.")
    print(f"\tRelaxed STO = {float(bandgap_relaxed_sto):.2f} eV.")
print(f"The PBE band gap for STO from Materials Project is {sto_dft_bandgap:.2f} eV.")
```


    model.pt:   0%|          | 0.00/1.13k [00:00<?, ?B/s]



    state.pt:   0%|          | 0.00/801k [00:00<?, ?B/s]



    model.json:   0%|          | 0.00/1.39k [00:00<?, ?B/s]


    PBE band gap
    	Unrelaxed STO = -0.01 eV.
    	Relaxed STO = -0.00 eV.
    GLLB-SC band gap
    	Unrelaxed STO = 0.46 eV.
    	Relaxed STO = 2.68 eV.
    HSE band gap
    	Unrelaxed STO = -0.00 eV.
    	Relaxed STO = 0.08 eV.
    SCAN band gap
    	Unrelaxed STO = 0.01 eV.
    	Relaxed STO = 0.02 eV.
    The PBE band gap for STO from Materials Project is 1.77 eV.


Again, you can see that using the unrelaxed SrTiO3 leads to large errors, predicting SrTiO3 to have very small band agps. Using the relaxed STO leads to predictions that are much closer to expectations. In particular, the predicted PBE band gap is quite close to the Materials Project PBE value. The experimental band gap is around 3.2 eV, which is reproduced very well by the GLLB-SC prediction!