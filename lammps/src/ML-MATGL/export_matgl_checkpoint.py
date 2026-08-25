#!/usr/bin/env python
"""Export a matgl checkpoint directory to a LAMMPS-ready TorchScript module.

Invoked automatically by pair_matgl/pair_matgl_kokkos's coeff() when
pair_coeff is given a directory (or a model.pt next to a model.json +
state.pt) holding a matgl IOMixIn checkpoint (model.json + model.pt +
state.pt), rather than an already-exported TorchScript module.

Usage: export_matgl_checkpoint.py <checkpoint_dir> <output.pt>
"""

import sys

import torch

import matgl

# Module layout as of matgl 4.0.3: the helpers live in matgl.graph._compute
# and the exporter in matgl.ext.lammps. The previous names
# (matgl.graph._compute_pyg, matgl.ext._lammps) predate that layout; the first
# no longer exists, so the import raised ModuleNotFoundError and made the
# "point pair_coeff at a checkpoint directory" path unusable.
# _compute exports create_line_graph_torch directly,
# so the in-memory shim that used to sit here is no longer needed either.
from matgl.ext.lammps import export_lammps_model

checkpoint_dir, out_path = sys.argv[1], sys.argv[2]
potential = matgl.load_model(checkpoint_dir)
export_lammps_model(potential, out_path, dtype=torch.float32, script=True)
