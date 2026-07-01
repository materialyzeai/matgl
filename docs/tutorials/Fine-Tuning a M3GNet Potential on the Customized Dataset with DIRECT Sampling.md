---
layout: default
title: Fine-Tuning a M3GNet Potential on the Customized Dataset with DIRECT Sampling.md
nav_exclude: true
---

# Introduction

This notebook demonstrates how to fine-tune a M3GNet potential combined with DIRECT Sampling in MatGL.


```python
from __future__ import annotations

import contextlib
import os
import shutil
import warnings
from functools import partial

import lightning as L
import numpy as np
from lightning.pytorch.loggers import CSVLogger
from pymatgen.ext.matproj import MPRester

import matgl
from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_pes, split_dataset
from matgl.utils.training import PotentialLightningModule

try:
    from maml.sampling.direct import BirchClustering, DIRECTSampler, SelectKFromClusters
except ImportError:
    print("MAML is not installed or the import failed.")
    print("Please install it by running:")
    print("pip install maml")

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
```

For the purposes of demonstration, we will download all Si-O compounds in the Materials Project via the MPRester. The forces and stresses are set to zero, though in a real context, these would be non-zero and obtained from DFT calculations.


```python
# Obtain your API key here: https://next-gen.materialsproject.org/api
mpr = MPRester(api_key="YOUR_API_KEY")
entries = mpr.get_entries_in_chemsys(["Si", "O"])
structures = [e.structure for e in entries]
energies = [e.energy for e in entries]
forces = [np.zeros((len(s), 3)).tolist() for s in structures]
stresses = [np.zeros((3, 3)).tolist() for s in structures]


print(f"{len(structures)} downloaded from MP.")
```

We will set up the DIRECTSampler to select structures with high diversity. Since the number of structures here is relatively small, the number of clusters, n, is set to 20. This parameter, along with the number of structures selected per cluster, k, can be adjusted based on your dataset.


```python
# Initialize DIRECT sampler
DIRECT_sampler = DIRECTSampler(
    clustering=BirchClustering(n=20, threshold_init=0.05), select_k_from_clusters=SelectKFromClusters(k=1)
)
# Fit the DIRECT sampler
DIRECT_selection = DIRECT_sampler.fit_transform(structures)
```
We can now select the structures obtained through DIRECT sampling.

```python
# Select structures from DIRECT sampling
selected_indexes = DIRECT_selection["selected_indexes"]
selected_structures = structures[selected_indexes]
selected_labels = {}
selected_labels["energies"] = energies[selected_indexes]
selected_labels["forces"] = forces[selected_indexes]
selected_labels["stresses"] = stresses[selected_indexes]

print(f"{len(selected_structures)} structures selected from DIRECT")
```

We can setup the MGLDataset and MGLDataLoader for the selected structures.


```python
# Using DEFAULT_ELEMENTS for element_types to adapt the pretrained models
element_types = DEFAULT_ELEMENTS
# Setup the graph converter for periodic systems
converter = Structure2Graph(element_types=element_types, cutoff=5.0)
# The three-body (line) graph is built on the fly inside M3GNet.forward using the
# model's own threebody_cutoff, so the dataset does not need a threebody_cutoff.
dataset = MGLDataset(
    structures=selected_structures,
    converter=converter,
    labels=selected_labels,
)
train_data, val_data, test_data = split_dataset(
    dataset,
    frac_list=[0.8, 0.1, 0.1],
    shuffle=True,
    random_state=42,
)
# if you are not intended to use stress for training, switch include_stress=False!
my_collate_fn = partial(collate_fn_pes, include_stress=True)
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=my_collate_fn,
    batch_size=2,
    num_workers=0,
)
```

## Finetuning a pre-trained M3GNet
In the following cells, we demonstrate the fine-tuning of our pretrained model on the customized dataset.


```python
# download a pre-trained M3GNet
m3gnet_nnp = matgl.load_model("M3GNet-MP-2021.2.8-PES")
model_pretrained = m3gnet_nnp.model
# obtain element energy offset
property_offset = m3gnet_nnp.element_refs.property_offset
# you should test whether including the original property_offset helps improve training and validation accuracy
lit_module_finetune = PotentialLightningModule(model=model_pretrained, element_refs=property_offset, lr=1e-4)
```


```python
# If you wish to disable GPU or MPS (M1 mac) training, use the accelerator="cpu" kwarg.
logger = CSVLogger("logs", name="M3GNet_finetuning")
trainer = L.Trainer(max_epochs=10, accelerator="cpu", logger=logger, inference_mode=False)
trainer.fit(model=lit_module_finetune, train_dataloaders=train_loader, val_dataloaders=val_loader)
```


```python
# save trained model
model_save_path = "./finetuned_model/"
lit_module_finetune.model.save(model_save_path)
# load trained model
trained_model = matgl.load_model(path=model_save_path)
```


```python
# This code just performs cleanup for this notebook.

for fn in ("pyg_graph.pt", "lattice.pt", "pyg_line_graph.pt", "state_attr.pt", "labels.json"):
    with contextlib.suppress(FileNotFoundError):
        os.remove(fn)

shutil.rmtree("logs")
shutil.rmtree("finetuned_model")
```