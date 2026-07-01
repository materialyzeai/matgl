---
layout: default
title: Training a Formation Energy Model with PyTorch Lightning.md
nav_exclude: true
---

# Introduction

This notebook demonstrates how to refit a **formation energy model** using PyTorch Lightning with MatGL. The same workflow trains either a **MEGNet** or an **M3GNet** model for an intensive property (formation energy per atom) - the only thing that changes is the model architecture. Choose the model with the `MODEL` switch below.


```python
from __future__ import annotations

import contextlib
import os
import shutil
import warnings

import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from lightning.pytorch.loggers import CSVLogger
from pymatgen.core import Structure
from tqdm import tqdm

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_graph, split_dataset
from matgl.layers import BondExpansion
from matgl.models import M3GNet, MEGNet
from matgl.utils.training import ModelLightningModule

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
```

# Model selection

Set `MODEL` to either `"MEGNet"` or `"M3GNet"`. Every subsequent cell - data loading, training, and plotting - is shared; only the model-construction cell branches on this choice.


```python
MODEL = "M3GNet"  # choose "M3GNet" or "MEGNet"
MAX_STRUCTURES = 100  # number of structures to use for this demo (set to None to use the full dataset)
```

# Dataset Preparation

We will download the original dataset used in the training of the MEGNet formation energy model (MP.2018.6.1) from the [`materialyze/mp.eform.2018.6.1`](https://huggingface.co/datasets/materialyze/mp.eform.2018.6.1) dataset on the Hugging Face Hub. `hf_hub_download` caches the file locally, so subsequent runs reuse the cached copy.


```python
# define a function to load the dataset from the Hugging Face Hub
def load_dataset() -> tuple[list[Structure], list[str], list[float]]:
    """Raw data loading function.

    Returns:
        tuple[list[Structure], list[str], list[float]]: structures, mp_id, Eform_per_atom
    """
    # Download (and cache) the dataset from the Hugging Face Hub.
    json_path = hf_hub_download(
        repo_id="materialyze/mp.eform.2018.6.1",
        filename="mp.eform.2018.6.1.json",
        repo_type="dataset",
    )
    data = pd.read_json(json_path)
    # For a quick demo we only use the first MAX_STRUCTURES entries. Slicing the
    # frame up front keeps structures and labels aligned and avoids parsing all
    # ~69k CIFs. Set MAX_STRUCTURES = None to train on the full dataset.
    if MAX_STRUCTURES is not None:
        data = data.head(MAX_STRUCTURES)
    structures = [Structure.from_str(s, fmt="cif") for s in tqdm(data["structure"])]
    return structures, data["material_id"].tolist(), data["formation_energy_per_atom"].tolist()


structures, mp_ids, eform_per_atom = load_dataset()
```

Here, we set up the dataset.


```python
# get element types in the dataset
elem_list = get_element_list(structures)
# setup a graph converter
converter = Structure2Graph(element_types=elem_list, cutoff=4.0)
# convert the raw dataset into a MGLDataset.
# For M3GNet, the three-body (line) graph is built on the fly inside M3GNet.forward using the
# model's own threebody_cutoff, so the dataset does not need a threebody_cutoff argument.
mp_dataset = MGLDataset(
    structures=structures,
    converter=converter,
    labels={"eform": eform_per_atom},
)
```

We will then split the dataset into training, validation and test data.


```python
train_data, val_data, test_data = split_dataset(
    mp_dataset,
    frac_list=[0.8, 0.1, 0.1],
    shuffle=True,
    random_state=42,
)
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=collate_fn_graph,
    batch_size=2,
    num_workers=0,
)
```

# Model setup

In the next step, we setup the model and the `ModelLightningModule`. The model is initialized from scratch according to the `MODEL` switch. Alternatively, you can also load one of the pre-trained models for transfer learning, which may speed up the training.


```python
if MODEL == "M3GNet":
    # setup the architecture of the M3GNet model
    model = M3GNet(
        element_types=elem_list,
        is_intensive=True,
        readout_type="set2set",
    )
elif MODEL == "MEGNet":
    # setup the embedding layer for node attributes
    node_embed = torch.nn.Embedding(len(elem_list), 16)
    # define the bond expansion
    bond_expansion = BondExpansion(rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.5)
    # setup the architecture of the MEGNet model
    model = MEGNet(
        dim_node_embedding=16,
        dim_edge_embedding=100,
        dim_state_embedding=2,
        nblocks=3,
        hidden_layer_sizes_input=(64, 32),
        hidden_layer_sizes_conv=(64, 64, 32),
        nlayers_set2set=1,
        niters_set2set=2,
        hidden_layer_sizes_output=(32, 16),
        is_classification=False,
        activation_type="softplus2",
        element_types=elem_list,
        bond_expansion=bond_expansion,
        cutoff=4.0,
        gauss_width=0.5,
    )
else:
    raise ValueError(f"Unknown MODEL: {MODEL!r}. Choose 'M3GNet' or 'MEGNet'.")

# setup the trainer
lit_module = ModelLightningModule(model=model)
```

# Training

Finally, we will initialize the Pytorch Lightning trainer and run the fitting. Note that the max_epochs is set at 20 to demonstrate the fitting on a laptop. A real fitting should use max_epochs > 100 and be run in parallel on GPU resources. For the formation energy, it should be around 2000. The `accelerator="cpu"` was set just to ensure compatibility with M1 Macs. In a real world use case, please remove the kwarg or set it to cuda for GPU based training. You may also need to use `torch.set_default_device("cuda")` or `with torch.device("cuda")` to ensure all data are loaded onto the GPU for training.

We have also initialized the Pytorch Lightning Trainer with a `CSVLogger`, which provides a detailed log of the loss metrics at each epoch.


```python
logger = CSVLogger("logs", name=f"{MODEL}_training")
trainer = L.Trainer(max_epochs=20, accelerator="cpu", logger=logger)
trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

# Visualizing the convergence

Finally, we can plot the convergence plot for the loss metrics. You can see that the MAE is already going down nicely with 20 epochs. Obviously, this is nowhere state of the art performance for the formation energies, but a longer training time should lead to results consistent with what was reported in the original work.


```python
metrics = pd.read_csv(f"logs/{MODEL}_training/version_0/metrics.csv")
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
shutil.rmtree("MGLDataset", ignore_errors=True)
```