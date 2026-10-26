---
layout: default
title: Property Predictions using MEGNet or M3GNet Models.md
nav_exclude: true
---

# Introduction

This notebook demonstrates the use of a pre-trained M3GNet model to predict formation energies.

Author: Tsz Wai Ko (Kenko)
Email: t1ko@ucsd.edu



```python
from __future__ import annotations

import warnings

from pymatgen.core import Lattice, Structure

import matgl

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
```

# MP Formation energy

The pre-trained M3GNet formation energy model is based on the Materials Project mp.2019.4.1 dataset.


We create the structure first. This is based on the relaxed structure obtained from the Materials Project. Alternatively, one can use the Materials Project API to obtain the structure.


```python
struct = Structure.from_spacegroup("Pm-3m", Lattice.cubic(4.1437), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
```

## Using the M3GNet-Eform-MP-2019.4.1 model


```python
# Load the pre-trained M3GNet formation energy model
model = matgl.load_model("M3GNet-Eform-MP-2019.4.1")
eform = model.predict_structure(struct)
print(f"The predicted formation energy for CsCl is {float(eform):.3f} eV/atom.")
```

    The predicted formation energy for CsCl is -2.218 eV/atom.



```python
# Intermediate outputs of the last prediction are stored in model.feature_dict.
# The structure-wise readout has dimension node_features * 2 from the set2set layer.
print(model.feature_dict["readout"].shape)
```

    torch.Size([1, 128])

