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

# matgl/src/matgl/ext/_lammps.py imports create_line_graph_torch, which is
# only used by the M3GNet export path and is absent from the current
# _compute_pyg.py. Irrelevant to TensorNet exports; shim it in-memory only
# (no matgl repo file changes) so the import below succeeds.
import matgl.graph._compute_pyg as _cpyg

if not hasattr(_cpyg, "create_line_graph_torch"):
    _cpyg.create_line_graph_torch = _cpyg.create_line_graph

from matgl.ext._lammps import export_lammps_model

checkpoint_dir, out_path = sys.argv[1], sys.argv[2]
potential = matgl.load_model(checkpoint_dir)
export_lammps_model(potential, out_path, dtype=torch.float32, script=True)
