---
layout: default
title: matgl.ext.torch_sim.md
nav_exclude: true
---

# TorchSim integration

`matgl.ext.torch_sim.MatGLModel` exposes a MatGL interatomic `Potential` through TorchSim's batched,
torch-native `ModelInterface`. Install the optional dependency on Python 3.12 or newer:

```sh
pip install 'matgl[torch-sim]'
```

The `torch-sim` and `alchmtk` extras currently cannot be installed together: TorchSim requires
`nvalchemi-toolkit-ops>=0.4`, while the current Alchemi Toolkit adapter is pinned below 0.4 for API compatibility.

The adapter builds graphs with TorchSim's neighbor list, accepts single or batched `SimState` objects, and returns
energy in eV, forces in eV/Angstrom, and stress in eV/Angstrom^3:

```python
import matgl
import torch_sim as ts
from ase.build import bulk
from matgl.ext.torch_sim import MatGLModel

potential = matgl.load_model("TensorNet-PES-MatPES-PBE-2025.2")
model = MatGLModel(potential)
state = ts.initialize_state([bulk("Si"), bulk("Si").repeat((2, 1, 1))], model.device, model.dtype)

properties = model(state)
relaxed = ts.optimize(system=state, model=model, optimizer=ts.Optimizer.fire)
```

Stress is reported as zero for nonperiodic systems, where bulk stress is undefined. For QET potentials, a
per-system `total_charge` TorchSim extra (or the same forward keyword) constrains QEq; predicted charges are returned
as `partial_charges`. Use `Potential(qet, calc_charge=True)` for charge-conditioned QET evaluation. The per-atom
`external_potential` extra (or `ext_pot` keyword) supplies an external potential; `state_attr` can be passed as a
system extra, constructor default, or forward keyword. Forward keywords take precedence over extras.

TorchSim's ASE initialization currently requires all systems in a batch to share PBC flags. Periodic, slab, and
molecular batches are supported. Set `model.device` and `model.dtype` to move the potential and its adapter buffers
together. The adapter configures the supplied potential for inference and returns detached tensors.
