# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""
NVT Molecular Dynamics with TensorNet and nvalchemi-toolkit
============================================================

This example runs NVT Langevin molecular dynamics on a periodic LiFePO4
crystal using a pretrained MatGL TensorNet potential wrapped for the
nvalchemi-toolkit dynamics engine.

The workflow:

1. Load a pretrained TensorNet potential from MatGL.
2. Wrap it with :class:`~matgl.ext.alchmtk.TensorNetWrapper`.
3. Build a periodic LiFePO4 structure using pymatgen.
4. Convert to :class:`~nvalchemi.data.AtomicData` via ``from_structure``.
5. Run 1000 NVT Langevin steps at 300 K.
6. Compute final temperature from kinetic energy.

Requirements::

    pip install matgl[alchmtk]

"""

from __future__ import annotations

import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks import LoggingHook
from pymatgen.util.testing import PymatgenTest

import matgl
from matgl.ext.alchmtk import TensorNetWrapper

# %%
# Load and wrap model
# --------------------
# Load a pretrained TensorNet PES potential from MatGL and wrap it
# with ``TensorNetWrapper`` for use in nvalchemi dynamics.

potential = matgl.load_model("TensorNet-MatPES-PBE-v2025.1-PES")
model = TensorNetWrapper.from_potential(potential)

print(f"Model: TensorNet (cutoff={model.model.cutoff} A, units={model.model.units})")
print(f"Outputs: {model.model_config.outputs}")
print(f"Active:  {model.model_config.active_outputs}")

# %%
# Build structure
# ----------------
# Load the LiFePO4 structure (28 atoms) from pymatgen's test database.
# ``from_structure`` handles periodic boundary conditions automatically.

structure = PymatgenTest.get_structure("LiFePO4")
n_atoms = len(structure)

# %%
# Build AtomicData with initial fields
# --------------------------------------
# The integrator requires ``forces``, ``energy``, and ``velocities``
# to be present on the batch before the first step.  We initialize
# forces and energy to zero (the model overwrites them at the first
# BEFORE_COMPUTE hook) and sample velocities from Maxwell-Boltzmann.

T_TARGET = 300.0  # K
KB_EV = 8.617333262e-5  # eV/K

data = AtomicData.from_structure(structure)
data.forces = torch.zeros(n_atoms, 3)
data.energy = torch.zeros(1, 1)

# Maxwell-Boltzmann velocities at T_TARGET
torch.manual_seed(42)
masses = data.atomic_masses  # amu
v_scale = (KB_EV * T_TARGET / masses).sqrt().unsqueeze(-1)
velocities = torch.randn(n_atoms, 3) * v_scale
velocities -= velocities.mean(dim=0, keepdim=True)  # zero COM velocity
data.add_node_property("velocities", velocities)

batch = Batch.from_data_list([data])
print(f"\nStructure: {structure.formula} ({n_atoms} atoms, cubic {structure.lattice.a:.1f} A)")

# %%
# NVTLangevin integrator and hooks
# ----------------------------------
# :class:`~nvalchemi.dynamics.NVTLangevin` implements the BAOAB Langevin
# splitting scheme.  Key arguments:
#
# * ``dt`` — timestep in fs
# * ``temperature`` — target temperature in K
# * ``friction`` — Langevin friction in 1/fs
# * ``random_seed`` — reproducible stochastic forces
#
# The neighbor list hook is registered via ``model.make_neighbor_hooks()``,
# which reads the model's ``NeighborConfig`` and creates the appropriate hook.

nvt = NVTLangevin(
    model=model,
    dt=1.0,
    temperature=T_TARGET,
    friction=0.01,
    random_seed=42,
    n_steps=1000,
)

for hook in model.make_neighbor_hooks():
    nvt.register_hook(hook, stage=DynamicsStage.BEFORE_COMPUTE)

with LoggingHook(backend="csv", log_path="md_log.csv", frequency=10) as log_hook:
    nvt.register_hook(log_hook)

    print(f"\nRunning {nvt.n_steps} NVT steps at T={T_TARGET} K ...")
    batch = nvt.run(batch)
    print(f"NVT completed {nvt.step_count} steps.")

# %%
# Inspecting temperature
# -----------------------
# The instantaneous kinetic temperature is computed from the
# equipartition theorem:
#
#   T = (2 · KE) / (3 · N · kB)

masses = batch.atomic_masses  # (N,) amu
vels = batch.velocities  # (N, 3)
ke_ev = 0.5 * (masses * (vels**2).sum(dim=-1)).sum().item()
T_final = (2.0 * ke_ev) / (3.0 * n_atoms * KB_EV)

print(f"\nFinal temperature: {T_final:.1f} K (target: {T_TARGET} K)")
print(f"Final energy: {batch.energy.item():.4f} eV")
print("MD trajectory saved to md_log.csv")
