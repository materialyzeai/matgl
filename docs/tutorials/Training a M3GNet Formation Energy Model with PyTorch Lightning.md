---
layout: default
title: Training a M3GNet Formation Energy Model with PyTorch Lightning.md
nav_exclude: true
---

# Introduction

This notebook demonstrates how to refit a MEGNet formation energy model using PyTorch Lightning with MatGL.


```python
from __future__ import annotations

import os
import shutil
import warnings
import zipfile
from functools import partial

import lightning as L
import matplotlib.pyplot as plt
import pandas as pd
import requests
from lightning.pytorch.loggers import CSVLogger
from pymatgen.core import Structure
from tqdm import tqdm

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_graph, split_dataset
from matgl.models import M3GNet
from matgl.utils.training import ModelLightningModule

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
```

# Dataset Preparation

We will download the original dataset used in the training of the MEGNet formation energy model (MP.2018.6.1) from figshare. To make it easier, we will also cache the data.


```python
# define a function to download the dataset
def download_file(url: str, filename: str):
    """Downloads a file from a URL and saves it locally."""
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error if the request fails

    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


# define a function to load the dataset
def load_dataset() -> tuple[list[Structure], list[str], list[float]]:
    """Raw data loading function.

    Returns:
        tuple[list[Structure], list[str], list[float]]: structures, mp_id, Eform_per_atom
    """
    json_filename = "mp.2018.6.1.json"
    zip_filename = "mp.2018.6.1.json.zip"

    url = "https://figshare.com/ndownloader/files/15087992"

    # Download and extract the dataset if it does not exist
    if not os.path.exists(json_filename):
        print(f"Downloading dataset from {url}...")
        download_file(url, zip_filename)
        with zipfile.ZipFile(zip_filename, "r") as zf:
            zf.extractall(".")
        os.remove(zip_filename)  # Clean up the zip file

        # Load the data
    data = pd.read_json(json_filename)
    structures = []
    mp_ids = []

    for mid, structure_str in tqdm(zip(data["material_id"], data["structure"], strict=False)):
        struct = Structure.from_str(structure_str, fmt="cif")
        structures.append(struct)
        mp_ids.append(mid)

    return structures, mp_ids, data["formation_energy_per_atom"].tolist()


structures, mp_ids, eform_per_atom = load_dataset()
```

    Downloading dataset from https://figshare.com/ndownloader/files/15087992...



    ---------------------------------------------------------------------------

    BadZipFile                                Traceback (most recent call last)

    Cell In[2], line 45
         41
         42     return structures, mp_ids, data["formation_energy_per_atom"].tolist()
         43
         44
    ---> 45 structures, mp_ids, eform_per_atom = load_dataset()


    Cell In[2], line 28, in load_dataset()
         24     # Download and extract the dataset if it does not exist
         25     if not os.path.exists(json_filename):
         26         print(f"Downloading dataset from {url}...")
         27         download_file(url, zip_filename)
    ---> 28         with zipfile.ZipFile(zip_filename, "r") as zf:
         29             zf.extractall(".")
         30         os.remove(zip_filename)  # Clean up the zip file
         31


    File ~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/zipfile/__init__.py:1370, in ZipFile.__init__(self, file, mode, compression, allowZip64, compresslevel, strict_timestamps, metadata_encoding)
       1368 try:
       1369     if mode == 'r':
    -> 1370         self._RealGetContents()
       1371     elif mode in ('w', 'x'):
       1372         # set the modified flag so central directory gets written
       1373         # even if no files are added to the archive
       1374         self._didModify = True


    File ~/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/lib/python3.12/zipfile/__init__.py:1437, in ZipFile._RealGetContents(self)
       1435     raise BadZipFile("File is not a zip file")
       1436 if not endrec:
    -> 1437     raise BadZipFile("File is not a zip file")
       1438 if self.debug > 1:
       1439     print(endrec)


    BadZipFile: File is not a zip file


For demonstration purposes, we are only going to select 100 structures from the entire set of structures to shorten the training time.


```python
structures = structures[:100]
eform_per_atom = eform_per_atom[:100]
```

Here, we set up the dataset.


```python
# get element types in the dataset
elem_list = get_element_list(structures)
# setup a graph converter
converter = Structure2Graph(element_types=elem_list, cutoff=4.0)
# convert the raw dataset into M3GNetDataset
mp_dataset = MGLDataset(
    threebody_cutoff=4.0,
    structures=structures,
    converter=converter,
    labels={"eform": eform_per_atom},
    include_line_graph=True,
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
my_collate_fn = partial(collate_fn_graph, include_line_graph=True)
train_loader, val_loader, test_loader = MGLDataLoader(
    train_data=train_data,
    val_data=val_data,
    test_data=test_data,
    collate_fn=my_collate_fn,
    batch_size=2,
    num_workers=1,
)
```

# Model setup

In the next step, we setup the model and the ModelLightningModule. Here, we have initialized a M3GNet model from scratch. Alternatively, you can also load one of the pre-trained models for transfer learning, which may speed up the training.


```python
# setup the architecture of M3GNet model
model = M3GNet(
    element_types=elem_list,
    is_intensive=True,
    readout_type="set2set",
)
# setup the M3GNetTrainer
lit_module = ModelLightningModule(model=model, include_line_graph=True)
```

# Training

Finally, we will initialize the Pytorch Lightning trainer and run the fitting. Note that the max_epochs is set at 20 to demonstrate the fitting on a laptop. A real fitting should use max_epochs > 100 and be run in parallel on GPU resources. For the formation energy, it should be around 2000. The `accelerator="cpu"` was set just to ensure compatibility with M1 Macs. In a real world use case, please remove the kwarg or set it to cuda for GPU based training. You may also need to use `torch.set_default_device("cuda")` or `with torch.device("cuda")` to ensure all data are loaded onto the GPU for training.

We have also initialized the Pytorch Lightning Trainer with a `CSVLogger`, which provides a detailed log of the loss metrics at each epoch.


```python
logger = CSVLogger("logs", name="M3GNet_training")
trainer = L.Trainer(max_epochs=20, accelerator="cpu", logger=logger)
trainer.fit(model=lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

# Visualizing the convergence

Finally, we can plot the convergence plot for the loss metrics. You can see that the MAE is already going down nicely with 20 epochs. Obviously, this is nowhere state of the art performance for the formation energies, but a longer training time should lead to results consistent with what was reported in the original M3GNet work.


```python
metrics = pd.read_csv("logs/M3GNet_training/version_0/metrics.csv")
metrics["train_MAE"].dropna().plot()
metrics["val_MAE"].dropna().plot()

_ = plt.legend()
```


```python
# This code just performs cleanup for this notebook.

for fn in ("pyg_graph.pt", "lattice.pt", "pyg_line_graph.pt", "state_attr.pt", "labels.json"):
    try:
        os.remove(fn)
    except FileNotFoundError:
        pass

shutil.rmtree("logs")
```