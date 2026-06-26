---
layout: default
title: CHGNet-PyG Foundation Potential.md
nav_exclude: true
---

# CHGNet-PyG Foundation Potential

This notebook demonstrates CHGNet with the PyTorch Geometric (PyG) backend. CHGNet is a
charge-informed graph neural network potential that jointly predicts energy, forces, stresses,
and site-wise **magnetic moments**.

We cover:
1. **Static prediction** – energy / forces / stresses / magnetic moments for a single structure
2. **Structure relaxation** – ionic + cell relaxation via the ASE `Relaxer`
3. **Molecular dynamics** – NVT trajectory via the ASE `MolecularDynamics` driver
4. **Training from scratch** – fit a small CHGNet on a custom dataset
5. **Fine-tuning** – continue training from the pre-trained MatPES checkpoint

**Reference:**
Deng, B. et al. *CHGNet as a pretrained universal neural network potential for charge-informed
atomistic modelling.* Nat. Mach. Intell. (2023) doi:10.1038/s42256-023-00716-3

Author: Bowen Deng


```python
from __future__ import annotations

import os
import warnings

import numpy as np
import torch
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

import matgl
from matgl.ext.ase import MolecularDynamics, PESCalculator, Relaxer

warnings.filterwarnings("ignore")
```

## 1. Load the pre-trained CHGNet-PyG model

Two MatPES checkpoints are available:

| Model name | Training functional |
|---|---|
| `BowenD-UCB/CHGNet-PyG-MatPES-r2SCAN-2025.2.10` | r2SCAN |
| `BowenD-UCB/CHGNet-PyG-MatPES-PBE-2025.2.10` | PBE |

Weights are numerically identical to the DGL checkpoints; the only difference is the
message-passing backend. All predictions are in **eV** (energy), **eV/Å** (forces),
**GPa** (stresses), and **μB** (magnetic moments).


```python
pot = matgl.load_model("BowenD-UCB/CHGNet-PyG-MatPES-r2SCAN-2025.2.10")
print(pot)
```

    Potential(
      (model): CHGNet(
        (bond_expansion): RadialBesselFunction()
        (threebody_bond_expansion): RadialBesselFunction()
        (angle_expansion): FourierExpansion()
        (atom_embedding): Embedding(89, 128)
        (bond_embedding): _MLPNorm(
          (layers): ModuleList(
            (0): Linear(in_features=63, out_features=128, bias=False)
          )
          (activation): SiLU()
        )
        (angle_embedding): _MLPNorm(
          (layers): ModuleList(
            (0): Linear(in_features=65, out_features=128, bias=False)
          )
          (activation): SiLU()
        )
        (atom_bond_weights): Linear(in_features=63, out_features=128, bias=False)
        (bond_bond_weights): Linear(in_features=63, out_features=128, bias=False)
        (threebody_bond_weights): Linear(in_features=63, out_features=128, bias=False)
        (atom_graph_layers): ModuleList(
          (0-4): 5 x CHGNetAtomGraphBlock(
            (conv): CHGNetGraphConv(
              (node_update_func): _GatedMLPNorm(
                (value): _MLPNorm(
                  (layers): ModuleList(
                    (0): Linear(in_features=384, out_features=128, bias=True)
                    (1): Linear(in_features=128, out_features=128, bias=True)
                  )
                  (norms): ModuleList(
                    (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True, bias=True)
                  )
                  (activation): SiLU()
                )
                (gate): _MLPNorm(
                  (layers): ModuleList(
                    (0): Linear(in_features=384, out_features=128, bias=True)
                    (1): Linear(in_features=128, out_features=128, bias=True)
                  )
                  (norms): ModuleList(
                    (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True, bias=True)
                  )
                  (activation): SiLU()
                )
                (sigmoid): Sigmoid()
              )
              (node_out_func): Linear(in_features=128, out_features=128, bias=False)
              (edge_update_func): _GatedMLPNorm(
                (value): _MLPNorm(
                  (layers): ModuleList(
                    (0): Linear(in_features=384, out_features=128, bias=True)
                    (1): Linear(in_features=128, out_features=128, bias=True)
                  )
                  (norms): ModuleList(
                    (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True, bias=True)
                  )
                  (activation): SiLU()
                )
                (gate): _MLPNorm(
                  (layers): ModuleList(
                    (0): Linear(in_features=384, out_features=128, bias=True)
                    (1): Linear(in_features=128, out_features=128, bias=True)
                  )
                  (norms): ModuleList(
                    (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True, bias=True)
                  )
                  (activation): SiLU()
                )
                (sigmoid): Sigmoid()
              )
            )
            (atom_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True, bias=True)
            (bond_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True, bias=True)
            (dropout): Identity()
          )
        )
        (bond_graph_layers): ModuleList(
          (0-3): 4 x CHGNetBondGraphBlock(
            (conv): CHGNetLineGraphConv(
              (node_update_func): _GatedMLPNorm(
                (value): _MLPNorm(
                  (layers): ModuleList(
                    (0): Linear(in_features=512, out_features=128, bias=True)
                    (1): Linear(in_features=128, out_features=128, bias=True)
                  )
                  (norms): ModuleList(
                    (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True, bias=True)
                  )
                  (activation): SiLU()
                )
                (gate): _MLPNorm(
                  (layers): ModuleList(
                    (0): Linear(in_features=512, out_features=128, bias=True)
                    (1): Linear(in_features=128, out_features=128, bias=True)
                  )
                  (norms): ModuleList(
                    (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True, bias=True)
                  )
                  (activation): SiLU()
                )
                (sigmoid): Sigmoid()
              )
              (node_out_func): Linear(in_features=128, out_features=128, bias=False)
              (edge_update_func): _GatedMLPNorm(
                (value): _MLPNorm(
                  (layers): ModuleList(
                    (0): Linear(in_features=512, out_features=128, bias=True)
                  )
                  (norms): ModuleList(
                    (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True, bias=True)
                  )
                  (activation): SiLU()
                )
                (gate): _MLPNorm(
                  (layers): ModuleList(
                    (0): Linear(in_features=512, out_features=128, bias=True)
                  )
                  (norms): ModuleList(
                    (0): LayerNorm((128,), eps=1e-05, elementwise_affine=True, bias=True)
                  )
                  (activation): SiLU()
                )
                (sigmoid): Sigmoid()
              )
            )
            (bond_dropout): Identity()
            (angle_dropout): Identity()
          )
        )
        (sitewise_readout): Linear(in_features=128, out_features=1, bias=True)
        (final_layer): _MLPNorm(
          (layers): ModuleList(
            (0-1): 2 x Linear(in_features=128, out_features=128, bias=True)
            (2): Linear(in_features=128, out_features=1, bias=True)
          )
          (activation): SiLU()
        )
        (final_dropout): Identity()
      )
      (element_refs): AtomRef()
    )


## 2. Static prediction

Use `PESCalculator` (an ASE `Calculator` wrapper) to obtain energy, forces,
stresses, and magnetic moments for any pymatgen `Structure`.


```python
# BCC iron — a classic magnetic test case
fe_struct = Structure(
    Lattice.cubic(2.87),
    ["Fe", "Fe"],
    [[0, 0, 0], [0.5, 0.5, 0.5]],
)

adaptor = AseAtomsAdaptor()
fe_atoms = adaptor.get_atoms(fe_struct)

calc = PESCalculator(potential=pot)
fe_atoms.set_calculator(calc)

energy = fe_atoms.get_potential_energy()  # eV
forces = fe_atoms.get_forces()  # eV/Å
magmoms = fe_atoms.calc.results["magmoms"]  # μB per site

print(f"Energy          : {energy:.4f} eV  ({energy / len(fe_struct):.4f} eV/atom)")
print(f"Max |force|     : {np.abs(forces).max():.4e} eV/Å")
print(f"Magnetic moments: {magmoms.flatten().tolist()}  μB")
```

    The stress unit is now in GPa. Please set stress_unit='eV/A3' if you want to use PESCalculator for other ASE applications.


    Energy          : -28.8047 eV  (-14.4023 eV/atom)
    Max |force|     : 1.2666e-07 eV/Å
    Magnetic moments: [2.735947370529175, 2.7359464168548584]  μB


### 2a. Direct Potential call (without ASE)

You can also call the `Potential` directly with a PyG graph for finer control
(e.g. batching, or when you need raw tensors for downstream differentiable ops).


```python
from matgl.ext.pymatgen import Structure2Graph

conv = Structure2Graph(
    element_types=pot.model.element_types,
    cutoff=pot.model.cutoff,
)

g, lat, state = conv.get_graph(fe_struct)
# Attach Cartesian positions and PBC shift vectors required by Potential
g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
g.pos = g.frac_coords @ lat[0]

# out = (energy, forces, stresses, hessian[unused], magmom)
out = pot(g=g, lat=lat, state_attr=state)
energy_t, forces_t, stress_t, _, magmom_t = out

print(f"Energy/atom : {energy_t.item() / len(fe_struct):.4f} eV")
print(f"Forces (eV/Å):\n{forces_t.detach().numpy()}")
print(f"Stress (GPa):\n{stress_t.detach().numpy()}")
print(f"Magmom (μB) : {magmom_t.detach().flatten().tolist()}")
```

    Energy/atom : -14.4023 eV
    Forces (eV/Å):
    [[-5.9604645e-08 -0.0000000e+00 -0.0000000e+00]
     [ 6.7055225e-08  1.4901161e-08 -3.7252903e-08]]
    Stress (GPa):
    [[ 1.2303069e+00 -4.3476845e-07  2.1738423e-07]
     [-5.0722986e-07  1.2303065e+00  2.1738423e-07]
     [-2.1738423e-07  0.0000000e+00  1.2303057e+00]]
    Magmom (μB) : [2.735947370529175, 2.7359464168548584]


## 3. Structure relaxation

`Relaxer` wraps the ASE cell-filter + FIRE optimizer. By default it relaxes both
ionic positions *and* the cell (`relax_cell=True`).


```python
# Deliberately distorted CsCl structure
struct = Structure(
    Lattice.cubic(4.5),
    ["Cs", "Cl"],
    [[0, 0, 0], [0.48, 0.52, 0.50]],  # slightly off-centre
)

relaxer = Relaxer(potential=pot)
result = relaxer.relax(struct, fmax=0.01, steps=500)

final_struct = result["final_structure"]
traj = result["trajectory"]

print(f"Initial energy  : {traj.energies[0]:.4f} eV")
print(f"Final energy    : {traj.energies[-1]:.4f} eV")
print(f"Relaxation steps: {len(traj.energies)}")
print(f"Final structure :\n{final_struct}")
```

    Initial energy  : -35.3086 eV
    Final energy    : -35.4979 eV
    Relaxation steps: 44
    Final structure :
    Full Formula (Cs1 Cl1)
    Reduced Formula: CsCl
    abc   :   4.126269   4.126269   4.126277
    angles:  90.000000  90.000001  90.107806
    pbc   :       True       True       True
    Sites (2)
      #  SP            a         b    c  final_magmom
    ---  ----  ---------  --------  ---  --------------
      0  Cs    -0.010176  0.010176  0    [-0.00871307]
      1  Cl     0.490176  0.509824  0.5  [-0.01081812]



```python
# Trajectory data as a DataFrame
df = traj.as_pandas()
df[["energies", "forces"]].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>energies</th>
      <th>forces</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-35.308647</td>
      <td>[[0.007051613, -0.0070515927, -0.0], [-0.00705...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-35.322723</td>
      <td>[[0.003279036, -0.003278911, -0.0], [-0.003279...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-35.348724</td>
      <td>[[-0.004945389, 0.004944224, -0.0], [0.0049454...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-35.382706</td>
      <td>[[-0.014665502, 0.014665462, -0.0], [0.0146654...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-35.420616</td>
      <td>[[-0.02560742, 0.02560746, -0.0], [0.02560737,...</td>
    </tr>
  </tbody>
</table>
</div>



## 4. Molecular dynamics

Run a short NVT simulation (Nose-Hoover thermostat) on the relaxed structure.
Ensembles available: `"nve"`, `"nvt"`, `"nvt_langevin"`, `"npt"`, `"npt_nose_hoover"`, …


```python
# Convert the relaxed pymatgen structure to ASE Atoms
atoms = adaptor.get_atoms(final_struct)

# Initialise velocities from a Maxwell-Boltzmann distribution at 300 K
MaxwellBoltzmannDistribution(atoms, temperature_K=300)

driver = MolecularDynamics(
    atoms=atoms,
    potential=pot,
    ensemble="nvt",  # Nose-Hoover NVT
    temperature=300,  # K
    timestep=2.0,  # fs
    taut=100,  # thermostat time constant (fs)
    logfile="md.log",
    loginterval=10,
)

driver.run(steps=100)
print("MD finished. Final potential energy:", atoms.get_potential_energy(), "eV")
```

    MD finished. Final potential energy: -35.44999313354492 eV


## 5. Training a CHGNet model from scratch

We demonstrate training on a small synthetic dataset. In a real workflow, replace
`structures` / `energies` / `forces` / `stresses` with your own DFT data (or download
from the Materials Project via `mp_api`).

Set `magmom_weight > 0` and include `"magmoms"` in the label dict to train the
magnetic moment head.


```python
from matgl.ext.pymatgen import get_element_list
from matgl.graph.data import MGLDataset
from matgl.models import CHGNet
from matgl.utils.training import MGLPotentialTrainer, fit_element_refs
```


```python
# Small synthetic dataset for demonstration
# Replace with real DFT data or an mp_api download in production
train_structures = [
    Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.01, 0.0, 0.0], [0.51, 0.5, 0.5]]),
    Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.02, 0.0], [0.5, 0.52, 0.5]]),
    Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, -0.01], [0.5, 0.5, 0.49]]),
    Structure(Lattice.cubic(3.95), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    Structure(Lattice.cubic(4.05), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
]
train_energies = [-10.00, -9.98, -9.97, -9.99, -9.95, -9.96]
train_forces = [np.zeros((2, 3)).tolist() for _ in train_structures]
train_stresses = [np.zeros((3, 3)).tolist() for _ in train_structures]

element_types = get_element_list(train_structures)
print(f"Element types: {element_types}")
```

    Element types: ('S', 'Mo')



```python
# Fit per-element energy offsets so the model learns residuals only
atomrefs = fit_element_refs(train_structures, train_energies, element_types)
print(f"Element refs (eV): {dict(zip(element_types, atomrefs.tolist(), strict=True))}")

# Build PyG graphs with three-body (line-graph) support
converter = Structure2Graph(element_types=element_types, cutoff=6.0)
dataset = MGLDataset(
    structures=train_structures,
    converter=converter,
    labels={"energies": train_energies, "forces": train_forces, "stresses": train_stresses},
    include_line_graph=True,
    save_cache=False,
)
print(f"Dataset: {len(dataset)} structures")
```

    Processing...


    Element refs (eV): {'S': -4.987499999999999, 'Mo': -4.987500000000002}


    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 1649.68it/s]

    Dataset: 6 structures



    Done!



```python
# Small CHGNet for demonstration (2 blocks)
model = CHGNet(
    element_types=element_types,
    num_blocks=2,
    dim_atom_embedding=64,
    dim_bond_embedding=64,
    dim_angle_embedding=64,
)

trainer = MGLPotentialTrainer(
    model=model,
    energy_weight=1.0,
    force_weight=1.0,
    stress_weight=0.1,
    magmom_weight=0.0,  # set > 0 when magmom labels are available
    loss="huber_loss",
    lr=1e-3,
    max_epochs=5,  # small for demo; use 100+ for real training
    accelerator="cpu",  # change to "gpu" / "cuda" on a GPU machine
)

# Lightning collects CUDA RNG states even when accelerator="cpu" if a CUDA
# device is visible but its driver is incompatible. Hide all GPUs to avoid
# the crash; remove this line (or set to your device id) on a GPU machine.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

potential = trainer.fit(
    dataset=dataset,
    atomrefs=atomrefs,
    save_path="./trained_chgnet_pyg/",
)
print("Training complete.")
```

    Seed set to 42
    GPU available: True (mps), used: False
    TPU available: False, using: 0 TPU cores
    💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
    💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold">   </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Name  </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Type              </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Params </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Mode  </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> FLOPs </span>┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 0 </span>│ mae   │ MeanAbsoluteError │      0 │ train │     0 │
│<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 1 </span>│ rmse  │ MeanSquaredError  │      0 │ train │     0 │
│<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 2 </span>│ model │ Potential         │  163 K │ train │     0 │
└───┴───────┴───────────────────┴────────┴───────┴───────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Trainable params</span>: 163 K
<span style="font-weight: bold">Non-trainable params</span>: 0
<span style="font-weight: bold">Total params</span>: 163 K
<span style="font-weight: bold">Total estimated model params size (MB)</span>: 0
<span style="font-weight: bold">Modules in train mode</span>: 79
<span style="font-weight: bold">Modules in eval mode</span>: 0
<span style="font-weight: bold">Total FLOPs</span>: 0
</pre>




    Output()


    `Trainer.fit` stopped: `max_epochs=5` reached.



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">      test_Charge_MAE      </span>│<span style="color: #800080; text-decoration-color: #800080">            0.0            </span>│
│<span style="color: #008080; text-decoration-color: #008080">     test_Charge_RMSE      </span>│<span style="color: #800080; text-decoration-color: #800080">            0.0            </span>│
│<span style="color: #008080; text-decoration-color: #008080">      test_Energy_MAE      </span>│<span style="color: #800080; text-decoration-color: #800080">    0.0143890380859375     </span>│
│<span style="color: #008080; text-decoration-color: #008080">     test_Energy_RMSE      </span>│<span style="color: #800080; text-decoration-color: #800080">    0.0143890380859375     </span>│
│<span style="color: #008080; text-decoration-color: #008080">      test_Force_MAE       </span>│<span style="color: #800080; text-decoration-color: #800080">   2.637534635141492e-11   </span>│
│<span style="color: #008080; text-decoration-color: #008080">      test_Force_RMSE      </span>│<span style="color: #800080; text-decoration-color: #800080">   3.453685601395584e-11   </span>│
│<span style="color: #008080; text-decoration-color: #008080">      test_Magmom_MAE      </span>│<span style="color: #800080; text-decoration-color: #800080">            0.0            </span>│
│<span style="color: #008080; text-decoration-color: #008080">     test_Magmom_RMSE      </span>│<span style="color: #800080; text-decoration-color: #800080">            0.0            </span>│
│<span style="color: #008080; text-decoration-color: #008080">      test_Stress_MAE      </span>│<span style="color: #800080; text-decoration-color: #800080">   0.006952452939003706    </span>│
│<span style="color: #008080; text-decoration-color: #008080">     test_Stress_RMSE      </span>│<span style="color: #800080; text-decoration-color: #800080">   0.012042000889778137    </span>│
│<span style="color: #008080; text-decoration-color: #008080">      test_Total_Loss      </span>│<span style="color: #800080; text-decoration-color: #800080">  0.00011077270028181374   </span>│
└───────────────────────────┴───────────────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



    Training complete.



```python
# Load the saved model back
loaded_pot = matgl.load_model("./trained_chgnet_pyg/")
print(loaded_pot)
```

    Potential(
      (model): CHGNet(
        (bond_expansion): RadialBesselFunction()
        (threebody_bond_expansion): RadialBesselFunction()
        (angle_expansion): FourierExpansion()
        (atom_embedding): Embedding(2, 64)
        (bond_embedding): _MLPNorm(
          (layers): ModuleList(
            (0): Linear(in_features=9, out_features=64, bias=False)
          )
          (activation): SiLU()
        )
        (angle_embedding): _MLPNorm(
          (layers): ModuleList(
            (0): Linear(in_features=9, out_features=64, bias=False)
          )
          (activation): SiLU()
        )
        (atom_bond_weights): Linear(in_features=9, out_features=64, bias=False)
        (bond_bond_weights): Linear(in_features=9, out_features=64, bias=False)
        (threebody_bond_weights): Linear(in_features=9, out_features=64, bias=False)
        (atom_graph_layers): ModuleList(
          (0-1): 2 x CHGNetAtomGraphBlock(
            (conv): CHGNetGraphConv(
              (node_update_func): _GatedMLPNorm(
                (value): _MLPNorm(
                  (layers): ModuleList(
                    (0): Linear(in_features=192, out_features=64, bias=True)
                    (1): Linear(in_features=64, out_features=64, bias=True)
                  )
                  (activation): SiLU()
                )
                (gate): _MLPNorm(
                  (layers): ModuleList(
                    (0): Linear(in_features=192, out_features=64, bias=True)
                    (1): Linear(in_features=64, out_features=64, bias=True)
                  )
                  (activation): SiLU()
                )
                (sigmoid): Sigmoid()
              )
              (node_out_func): Linear(in_features=64, out_features=64, bias=False)
            )
            (dropout): Identity()
          )
        )
        (bond_graph_layers): ModuleList(
          (0): CHGNetBondGraphBlock(
            (conv): CHGNetLineGraphConv(
              (node_update_func): _GatedMLPNorm(
                (value): _MLPNorm(
                  (layers): ModuleList(
                    (0): Linear(in_features=256, out_features=64, bias=True)
                    (1): Linear(in_features=64, out_features=64, bias=True)
                  )
                  (activation): SiLU()
                )
                (gate): _MLPNorm(
                  (layers): ModuleList(
                    (0): Linear(in_features=256, out_features=64, bias=True)
                    (1): Linear(in_features=64, out_features=64, bias=True)
                  )
                  (activation): SiLU()
                )
                (sigmoid): Sigmoid()
              )
              (node_out_func): Linear(in_features=64, out_features=64, bias=False)
              (edge_update_func): _GatedMLPNorm(
                (value): _MLPNorm(
                  (layers): ModuleList(
                    (0): Linear(in_features=256, out_features=64, bias=True)
                  )
                  (activation): SiLU()
                )
                (gate): _MLPNorm(
                  (layers): ModuleList(
                    (0): Linear(in_features=256, out_features=64, bias=True)
                  )
                  (activation): SiLU()
                )
                (sigmoid): Sigmoid()
              )
            )
            (bond_dropout): Identity()
            (angle_dropout): Identity()
          )
        )
        (sitewise_readout): Linear(in_features=64, out_features=1, bias=True)
        (final_layer): _MLPNorm(
          (layers): ModuleList(
            (0-1): 2 x Linear(in_features=64, out_features=64, bias=True)
            (2): Linear(in_features=64, out_features=1, bias=True)
          )
          (activation): SiLU()
        )
        (final_dropout): Identity()
      )
      (element_refs): AtomRef()
    )


## 6. Fine-tuning the pre-trained CHGNet-PyG

Start from the MatPES r2SCAN checkpoint and continue training on your own dataset.
A lower learning rate (`lr=1e-4`) prevents the pre-trained weights from drifting too fast.

Since the pre-trained CHGNet was trained on the full periodic table, make sure
`element_types` here matches the pre-trained model's `element_types` (or is a subset of it).


```python
# Load the pre-trained potential
pretrained_pot = matgl.load_model("BowenD-UCB/CHGNet-PyG-MatPES-r2SCAN-2025.2.10")
pretrained_model = pretrained_pot.model

# Per-element energy references from the pre-trained checkpoint
property_offset = pretrained_pot.element_refs.property_offset.numpy()

# Build dataset using the same cutoffs as the pre-trained model
ft_converter = Structure2Graph(
    element_types=pretrained_model.element_types,
    cutoff=pretrained_model.cutoff,
)
ft_dataset = MGLDataset(
    structures=train_structures,
    converter=ft_converter,
    labels={"energies": train_energies, "forces": train_forces, "stresses": train_stresses},
    include_line_graph=True,
    save_cache=False,
)

ft_trainer = MGLPotentialTrainer(
    model=pretrained_model,
    energy_weight=1.0,
    force_weight=1.0,
    stress_weight=0.1,
    magmom_weight=0.0,  # set > 0 and add "magmoms" to labels to train the magmom head
    lr=1e-4,  # lower LR for fine-tuning
    max_epochs=5,
    accelerator="cpu",
)

ft_potential = ft_trainer.fit(
    dataset=ft_dataset,
    atomrefs=property_offset,
    save_path="./finetuned_chgnet_pyg/",
)
print("Fine-tuning complete.")
```

    Processing...
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:00<00:00, 2660.80it/s]
    Done!
    Seed set to 42
    GPU available: True (mps), used: False
    TPU available: False, using: 0 TPU cores
    💡 Tip: For seamless cloud logging and experiment tracking, try installing [litlogger](https://pypi.org/project/litlogger/) to enable LitLogger, which logs metrics and artifacts automatically to the Lightning Experiments platform.
    💡 Tip: For seamless cloud uploads and versioning, try installing [litmodels](https://pypi.org/project/litmodels/) to enable LitModelCheckpoint, which syncs automatically with the Lightning model registry.



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━┳━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━┓
┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold">   </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Name  </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Type              </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Params </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> Mode  </span>┃<span style="color: #800080; text-decoration-color: #800080; font-weight: bold"> FLOPs </span>┃
┡━━━╇━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━┩
│<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 0 </span>│ mae   │ MeanAbsoluteError │      0 │ train │     0 │
│<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 1 </span>│ rmse  │ MeanSquaredError  │      0 │ train │     0 │
│<span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 2 </span>│ model │ Potential         │  2.7 M │ train │     0 │
└───┴───────┴───────────────────┴────────┴───────┴───────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Trainable params</span>: 2.7 M
<span style="font-weight: bold">Non-trainable params</span>: 0
<span style="font-weight: bold">Total params</span>: 2.7 M
<span style="font-weight: bold">Total estimated model params size (MB)</span>: 10
<span style="font-weight: bold">Modules in train mode</span>: 322
<span style="font-weight: bold">Modules in eval mode</span>: 0
<span style="font-weight: bold">Total FLOPs</span>: 0
</pre>




    Output()


    `Trainer.fit` stopped: `max_epochs=5` reached.



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




    Output()



<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold">        Test metric        </span>┃<span style="font-weight: bold">       DataLoader 0        </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│<span style="color: #008080; text-decoration-color: #008080">      test_Charge_MAE      </span>│<span style="color: #800080; text-decoration-color: #800080">            0.0            </span>│
│<span style="color: #008080; text-decoration-color: #008080">     test_Charge_RMSE      </span>│<span style="color: #800080; text-decoration-color: #800080">            0.0            </span>│
│<span style="color: #008080; text-decoration-color: #008080">      test_Energy_MAE      </span>│<span style="color: #800080; text-decoration-color: #800080">    11.532892227172852     </span>│
│<span style="color: #008080; text-decoration-color: #008080">     test_Energy_RMSE      </span>│<span style="color: #800080; text-decoration-color: #800080">    11.532892227172852     </span>│
│<span style="color: #008080; text-decoration-color: #008080">      test_Force_MAE       </span>│<span style="color: #800080; text-decoration-color: #800080">   5.277494441457975e-09   </span>│
│<span style="color: #008080; text-decoration-color: #008080">      test_Force_RMSE      </span>│<span style="color: #800080; text-decoration-color: #800080">   7.411673941248864e-09   </span>│
│<span style="color: #008080; text-decoration-color: #008080">      test_Magmom_MAE      </span>│<span style="color: #800080; text-decoration-color: #800080">            0.0            </span>│
│<span style="color: #008080; text-decoration-color: #008080">     test_Magmom_RMSE      </span>│<span style="color: #800080; text-decoration-color: #800080">            0.0            </span>│
│<span style="color: #008080; text-decoration-color: #008080">      test_Stress_MAE      </span>│<span style="color: #800080; text-decoration-color: #800080">    0.9226765036582947     </span>│
│<span style="color: #008080; text-decoration-color: #008080">     test_Stress_RMSE      </span>│<span style="color: #800080; text-decoration-color: #800080">    1.5981227159500122     </span>│
│<span style="color: #008080; text-decoration-color: #008080">      test_Total_Loss      </span>│<span style="color: #800080; text-decoration-color: #800080">    11.108492851257324     </span>│
└───────────────────────────┴───────────────────────────┘
</pre>




<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>



    Fine-tuning complete.


## 7. Cleanup


```python
import os
import shutil

for path in ("md.log", "trained_chgnet_pyg", "finetuned_chgnet_pyg"):
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
```