"""Generate the MEGNet parity reference artifact.

Run this script manually to (re)generate ``tests/parity_data/megnet_parity.pt``.
The consumer test in ``test_megnet.py`` loads this artifact and checks ``allclose``.

Usage::

    uv run python tests/parity_data/gen_megnet_parity.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

import matgl
from matgl.ext._pymatgen import Structure2Graph
from matgl.models import MEGNet

# Architecture / forward inputs MUST stay in lockstep with the consumer test.
INIT_ARGS: dict = {
    "element_types": ("Mo", "S"),
    "nblocks": 2,
    "dim_node_embedding": 8,
    "dim_edge_embedding": 16,
    "dim_state_embedding": 2,
    "hidden_layer_sizes_input": (16, 8),
    "hidden_layer_sizes_conv": (16, 8),
    "hidden_layer_sizes_output": (8, 4),
    "niters_set2set": 2,
    "nlayers_set2set": 1,
    "activation_type": "swish",
    "include_state": True,
    "is_classification": False,
    "ntypes_state": None,
    "cutoff": 5.0,
}

STRUCTURE_KW = {"lattice": Lattice.cubic(4.0), "species": ["Mo", "S"], "coords": [[0, 0, 0], [0.5, 0.5, 0.5]]}
STATE_ATTR = np.array([0.0, 0.0], dtype=np.float32)
SEED = 42


def build_graph(structure):
    """Build a PyG graph + position tensors."""
    conv = Structure2Graph(element_types=INIT_ARGS["element_types"], cutoff=INIT_ARGS["cutoff"])
    g, lat, _ = conv.get_graph(structure)
    g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
    g.pos = g.frac_coords @ lat[0]
    return g


def main():
    """Generate the parity artifact."""
    torch.manual_seed(SEED)
    structure = Structure(**STRUCTURE_KW)
    model = MEGNet(**INIT_ARGS)
    model.eval()
    g = build_graph(structure)
    state = torch.tensor(STATE_ATTR, dtype=matgl.float_th)
    with torch.no_grad():
        output = model(g=g, state_attr=state)
    artifact = {
        "init_args": INIT_ARGS,
        "state_dict": model.state_dict(),
        "expected_output": output.detach().clone(),
        "structure_kw": STRUCTURE_KW,
        "state_attr": STATE_ATTR,
        "seed": SEED,
    }
    out_path = Path(__file__).parent / "megnet_parity.pt"
    torch.save(artifact, out_path)
    print(f"Wrote {out_path}: output={output.item():.6f}")


if __name__ == "__main__":
    main()
