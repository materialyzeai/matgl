---
layout: default
title: Training a Multi-Fidelity MEGNet BandGap Energy Model with PyTorch Lightning.md
nav_exclude: true
---

# Introduction

This notebook demonstrates how to train a Multi-Fidelity MEGNet Band Gap model from scratch using PyTorch Lightning with MatGL.


```python
from __future__ import annotations

import contextlib
import json
import os
import shutil
import warnings
from copy import deepcopy

import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import hf_hub_download
from lightning.pytorch.loggers import CSVLogger
from pymatgen.core import Structure

from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_graph, split_dataset
from matgl.models import MEGNet
from matgl.utils.training import ModelLightningModule

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
```

# Dataset Preparation

We will download the original dataset used in the training of the Multi-Fidelity Band Gap model (MP.2019.4.1) from the Hugging Face Hub: the structures from [`materialyze/mp.eform.2019.4.1`](https://huggingface.co/datasets/materialyze/mp.eform.2019.4.1) and the multi-fidelity band gaps from [`materialyze/bandgap.mfi.mp.2019.04.01`](https://huggingface.co/datasets/materialyze/bandgap.mfi.mp.2019.04.01). `hf_hub_download` caches the files locally, so subsequent runs reuse the cached copies.


```python
# Download (and cache) the structures and multi-fidelity band gaps from the Hugging Face Hub.
structure_path = hf_hub_download(
    repo_id="materialyze/mp.eform.2019.4.1",
    filename="mp.eform.2019.4.1.json",
    repo_type="dataset",
)
bandgap_path = hf_hub_download(
    repo_id="materialyze/bandgap.mfi.mp.2019.04.01",
    filename="band_gap.json",
    repo_type="dataset",
)

ALL_FIDELITIES = ["pbe", "gllb-sc", "hse", "scan"]

## Load the structures
with open(structure_path) as f:
    structure_data = {i["material_id"]: i["structure"] for i in json.load(f)}
print(f"Structure dataset contains {len(structure_data)} structures")


##  Band gap data
with open(bandgap_path) as f:
    bandgap_data = json.load(f)

useful_ids = set.union(*[set(bandgap_data[i].keys()) for i in ALL_FIDELITIES])  # mp ids that are used in training
print(f"Only {len(useful_ids)} structures are used")
print("Calculating the graphs for all structures... this may take minutes.")
structure_data = {i: structure_data[i] for i in useful_ids}
structure_data = {i: Structure.from_str(j, fmt="cif") for i, j in structure_data.items()}
```

In this section, we generate graphs and labels corresponding to the fidelities.


```python
##  Generate graphs and Combined with Fidelity Labels
structures = []
material_ids = []
graph_attrs = []
targets = []
for fidelity_id, fidelity in enumerate(ALL_FIDELITIES):
    for mp_id in bandgap_data[fidelity]:
        structure = deepcopy(structure_data[mp_id])

        # The fidelity information is included here by changing the state attributes
        # PBE: 0, GLLB-SC: 1, HSE: 2, SCAN: 3
        graph_attrs.append(fidelity_id)
        structures.append(structure)
        targets.append(bandgap_data[fidelity][mp_id])
        # the new id is of the form mp-id_fidelity, e.g., mp-1234_pbe
        material_ids.append(f"{mp_id}_{fidelity}")
```

Here, we set up the dataset.


```python
# Define Graph Converter
element_types = DEFAULT_ELEMENTS
cry_graph = Structure2Graph(element_types, cutoff=5.0)
# Define labels for bandgap values
labels = {"bandgap": targets}
# Use a dedicated cache root so this dataset never collides with other MEGNet
# examples (e.g. the eform notebook) that default to root="MGLDataset".
dataset = MGLDataset(
    structures=structures,
    graph_labels=graph_attrs,
    labels=labels,
    converter=cry_graph,
    root="MGLDataset_bandgap",
)
```

We will then split the dataset into training, validation and test data.


```python
# here we set 0.1, 0.1 and 0.8 for demonstration purpose to shorten the training time
train_data, val_data, test_data = split_dataset(
    dataset,
    frac_list=[0.1, 0.1, 0.8],
    shuffle=True,
    random_state=42,
)
# Initialize MGLDataLoader
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=collate_fn_graph,
    batch_size=64,
)
```

# Model setup

In the next step, we setup the model and the ModelLightningModule. Here, we have initialized a MEGNet model from scratch. Alternatively, you can also load one of the pre-trained models for transfer learning, which may speed up the training.


```python
# setup the MEGNet model
model = MEGNet(
    element_types=element_types,
    cutoff=5.0,
    dim_state_embedding=64,
    ntypes_state=4,
    include_state=True,
)

# setup the MEGNetTrainer
lit_module = ModelLightningModule(model=model)
```

# Training

Finally, we will initialize the Pytorch Lightning trainer and run the fitting. Note that the max_epochs is set at 20 to demonstrate the fitting on a laptop. A real fitting should use max_epochs > 100 and be run in parallel on GPU resources. For the formation energy, it should be around 2000. The `accelerator="cpu"` was set just to ensure compatibility with M1 Macs. In a real world use case, please remove the kwarg or set it to cuda for GPU based training. You may also need to use `torch.set_default_device("cuda")` or `with torch.device("cuda")` to ensure all data are loaded onto the GPU for training.

We have also initialized the Pytorch Lightning Trainer with a `CSVLogger`, which provides a detailed log of the loss metrics at each epoch.


```python
logger = CSVLogger("logs", name="MEGNet_training")
trainer = L.Trainer(max_epochs=20, accelerator="cpu", logger=logger)
trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

# Visualizing the convergence

Finally, we can plot the convergence plot for the loss metrics. You can see that the MAE is already going down nicely with 20 epochs. Obviously, this is nowhere state of the art performance for the band gap, but a longer training time should lead to results consistent with what was reported in the original MEGNet work.


```python
metrics = pd.read_csv("logs/MEGNet_training/version_0/metrics.csv")
metrics["train_MAE"].dropna().plot()
metrics["val_MAE"].dropna().plot()

_ = plt.legend()
```


```python
# This code just performs cleanup for this notebook.

for fn in ("pyg_graph.pt", "lattice.pt", "pyg_line_graph.pt", "state_attr.pt", "labels.json"):
    with contextlib.suppress(FileNotFoundError):
        os.remove(fn)

shutil.rmtree("logs")
shutil.rmtree("MGLDataset_bandgap")
```