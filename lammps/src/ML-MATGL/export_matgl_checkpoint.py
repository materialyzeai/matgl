#!/usr/bin/env python
"""Export a matgl checkpoint directory to a LAMMPS-ready TorchScript module.

Invoked automatically by pair_matgl / pair_matgl_kokkos's coeff() when
pair_coeff is given a directory (or a model.pt next to a model.json +
state.pt) holding a matgl IOMixIn checkpoint, rather than an
already-exported TorchScript module. Equivalent to
``mgl create-lammps-model -m <checkpoint_dir> -o <output.pt> --dtype float32``.

Usage: export_matgl_checkpoint.py <checkpoint_dir> <output.pt>
"""

from __future__ import annotations

import sys

import torch

import matgl
from matgl.ext.lammps import export_lammps_model

if len(sys.argv) != 3:
    sys.exit(__doc__)
checkpoint_dir, out_path = sys.argv[1], sys.argv[2]

potential = matgl.load_model(checkpoint_dir)
potential.eval()
wrapper = export_lammps_model(potential, out_path, dtype=torch.float32, script=True)
print(f"exported {checkpoint_dir} -> {out_path} (r_max={wrapper.r_max}, n_species={wrapper.n_species}, float32)")
