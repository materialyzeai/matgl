"""Generate the M3GNet DGL<->PyG parity reference artifact.

Run this script manually under either backend to (re)generate
``tests/parity_data/m3gnet_parity.pt``. Both DGL and PyG must produce the
same numeric output for parity to hold; the consumer test in
``test_m3gnet_parity.py`` loads this artifact and checks ``allclose``.

Usage::

    uv run python tests/parity_data/gen_m3gnet_parity.py             # PyG
    MATGL_BACKEND=DGL .venv_dgl/bin/python tests/parity_data/gen_m3gnet_parity.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

import matgl
from matgl.models import M3GNet

INIT_ARGS: dict = {
    "element_types": ("Mo", "S"),
    "dim_node_embedding": 16,
    "dim_edge_embedding": 16,
    "max_n": 3,
    "max_l": 3,
    "nblocks": 2,
    "rbf_type": "SphericalBessel",
    "is_intensive": False,
    "cutoff": 5.0,
    "threebody_cutoff": 4.0,
    "units": 16,
    "use_smooth": False,
    "use_phi": False,
    "include_state": False,
    "activation_type": "swish",
}

STRUCTURE_KW = {"lattice": Lattice.cubic(4.0), "species": ["Mo", "S"], "coords": [[0, 0, 0], [0.5, 0.5, 0.5]]}
STATE_ATTR = np.array([0.0, 0.0], dtype=np.float32)
SEED = 42


def build_graph(structure):
    """Build a backend-aware graph + position tensors."""
    if matgl.config.BACKEND == "DGL":
        from matgl.ext._pymatgen_dgl import Structure2Graph

        conv = Structure2Graph(element_types=INIT_ARGS["element_types"], cutoff=INIT_ARGS["cutoff"])
        g, lat, _ = conv.get_graph(structure)
        g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lat[0])
        g.ndata["pos"] = g.ndata["frac_coords"] @ lat[0]
        return g
    from matgl.ext._pymatgen_pyg import Structure2Graph

    conv = Structure2Graph(element_types=INIT_ARGS["element_types"], cutoff=INIT_ARGS["cutoff"])
    g, lat, _ = conv.get_graph(structure)
    g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
    g.pos = g.frac_coords @ lat[0]
    return g


def main():
    """Generate the parity artifact."""
    torch.manual_seed(SEED)
    structure = Structure(**STRUCTURE_KW)
    model = M3GNet(**INIT_ARGS)
    model.eval()
    g = build_graph(structure)
    with torch.no_grad():
        output = model(g=g)
    artifact = {
        "init_args": INIT_ARGS,
        "state_dict": model.state_dict(),
        "expected_output": output.detach().clone(),
        "structure_kw": STRUCTURE_KW,
        "state_attr": STATE_ATTR,
        "seed": SEED,
        "generated_under_backend": matgl.config.BACKEND,
    }
    out_path = Path(__file__).parent / "m3gnet_parity.pt"
    torch.save(artifact, out_path)
    print(f"Wrote {out_path}: backend={matgl.config.BACKEND}, output={output.item():.6f}")


if __name__ == "__main__":
    main()
