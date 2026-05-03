"""DGL<->PyG parity test for ``MEGNet``.

Loads a state-dict + golden output from ``tests/parity_data/megnet_parity.pt``
(generated under one backend) and verifies the model on the *current* backend
reproduces the reference output to floating-point tolerance. Run under both
backends in CI to establish parity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from pymatgen.core import Structure

import matgl
from matgl.models import MEGNet

ARTIFACT = Path(__file__).resolve().parents[1] / "parity_data" / "megnet_parity.pt"


@pytest.fixture(scope="module")
def parity_artifact():
    """Load the MEGNet parity artifact."""
    if not ARTIFACT.exists():
        pytest.skip(f"Parity artifact missing: {ARTIFACT}. Generate via tests/parity_data/gen_megnet_parity.py.")
    return torch.load(ARTIFACT, map_location="cpu", weights_only=False)


def _build_graph(structure, init_args):
    """Build a backend-aware graph + position tensors mirroring the generator."""
    if matgl.config.BACKEND == "DGL":
        from matgl.ext._pymatgen_dgl import Structure2Graph

        conv = Structure2Graph(element_types=init_args["element_types"], cutoff=init_args["cutoff"])
        g, lat, _ = conv.get_graph(structure)
        g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lat[0])
        g.ndata["pos"] = g.ndata["frac_coords"] @ lat[0]
        return g
    from matgl.ext._pymatgen_pyg import Structure2Graph

    conv = Structure2Graph(element_types=init_args["element_types"], cutoff=init_args["cutoff"])
    g, lat, _ = conv.get_graph(structure)
    g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
    g.pos = g.frac_coords @ lat[0]
    return g


def test_megnet_dgl_pyg_parity(parity_artifact):
    """The state-dict trained on one backend must reproduce its golden output on the other."""
    init_args = parity_artifact["init_args"]
    state_dict = parity_artifact["state_dict"]
    expected = parity_artifact["expected_output"]
    structure_kw = parity_artifact["structure_kw"]
    state_attr = parity_artifact["state_attr"]
    structure = Structure(**structure_kw)

    model = MEGNet(**init_args)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not unexpected, f"Unexpected keys when loading parity state_dict: {unexpected}"
    # Acceptable missing keys are bond_expansion buffers (computed at forward time).
    bad_missing = [k for k in missing if "bond_expansion" not in k]
    assert not bad_missing, f"Missing keys when loading parity state_dict: {bad_missing}"

    model.eval()
    g = _build_graph(structure, init_args)
    state = torch.tensor(np.asarray(state_attr), dtype=matgl.float_th)
    with torch.no_grad():
        output = model(g=g, state_attr=state)

    assert torch.allclose(output, expected, atol=1e-5, rtol=1e-5), (
        f"MEGNet parity broken on backend={matgl.config.BACKEND}: "
        f"got {output.item()}, expected {expected.item()} "
        f"(artifact generated under {parity_artifact['generated_under_backend']})"
    )
