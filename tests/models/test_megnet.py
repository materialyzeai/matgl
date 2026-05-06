from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

import matgl

BACKEND = matgl.config.BACKEND

if BACKEND == "DGL":
    from matgl.graph._compute_dgl import compute_pair_vector_and_distance
elif BACKEND == "PYG":
    from matgl.graph._compute_pyg import compute_pair_vector_and_distance  # type: ignore[assignment]
else:
    pytest.skip(f"Unsupported backend: {BACKEND}", allow_module_level=True)

from matgl.models import MEGNet  # noqa: E402

PARITY_ARTIFACT = Path(__file__).resolve().parents[1] / "parity_data" / "megnet_parity.pt"


def _device_of(graph):
    if BACKEND == "DGL":
        return graph.device
    return graph.pos.device


def _prep_graph(graph, structure):
    """Attach pos / pbc_offshift / bond_dist for the active backend."""
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=_device_of(graph))
    if BACKEND == "DGL":
        graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
        graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
        _, bond_dist = compute_pair_vector_and_distance(graph)
        graph.edata["bond_dist"] = bond_dist
    else:
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
        graph.pos = graph.frac_coords @ lat[0]
        _, bond_dist = compute_pair_vector_and_distance(graph.pos, graph.edge_index, graph.pbc_offshift)
        graph.bond_dist = bond_dist
    return graph


def _make_megnet(**overrides):
    base = {
        "dim_node_embedding": 16,
        "dim_edge_embedding": 100,
        "dim_state_embedding": 2,
        "nblocks": 3,
        "include_state": True,
        "hidden_layer_sizes_input": (64, 32),
        "hidden_layer_sizes_conv": (64, 64, 32),
        "activation_type": "swish",
        "nlayers_set2set": 4,
        "niters_set2set": 3,
        "hidden_layer_sizes_output": (32, 16),
        "is_classification": True,
    }
    base.update(overrides)
    return MEGNet(**base)


def test_megnet(graph_MoS):
    structure, graph, state = graph_MoS
    graph = _prep_graph(graph, structure)
    state_t = torch.tensor(np.array(state), dtype=matgl.float_th)
    output = None
    for act in ["tanh", "sigmoid", "softplus2", "softexp", "swish"]:
        model = _make_megnet(activation_type=act)
        output = model(g=graph, state_attr=state_t)
    with pytest.raises(ValueError, match="Invalid activation type"):
        _ = MEGNet(activation_type="whatever")
    assert torch.numel(output) == 1


def test_megnet_isolated_atom():
    structure = Structure(Lattice.cubic(10.0), ["Mo"], [[0.0, 0, 0]])
    model = _make_megnet(dropout=0.1)
    output = model.predict_structure(structure)
    assert torch.numel(output) == 1


def test_save_load(tmp_path):
    model = _make_megnet(dropout=0.1)
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        model.save(".", metadata={"description": "forme model"})
        MEGNet.load(".")
    finally:
        os.chdir(cwd)


# ---------------------------------------------------------------------------
# DGL <-> PyG parity (artifact-driven; runs on whichever backend is active)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parity_artifact():
    """Load the MEGNet parity artifact."""
    if not PARITY_ARTIFACT.exists():
        pytest.skip(f"Parity artifact missing: {PARITY_ARTIFACT}. Generate via tests/parity_data/gen_megnet_parity.py.")
    return torch.load(PARITY_ARTIFACT, map_location="cpu", weights_only=False)


def _build_parity_graph(structure, init_args):
    """Build a backend-aware graph + position tensors mirroring the artifact generator."""
    if BACKEND == "DGL":
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
    bad_missing = [k for k in missing if "bond_expansion" not in k]
    assert not bad_missing, f"Missing keys when loading parity state_dict: {bad_missing}"

    model.eval()
    g = _build_parity_graph(structure, init_args)
    state = torch.tensor(np.asarray(state_attr), dtype=matgl.float_th)
    with torch.no_grad():
        output = model(g=g, state_attr=state)

    assert torch.allclose(output, expected, atol=1e-5, rtol=1e-5), (
        f"MEGNet parity broken on backend={BACKEND}: "
        f"got {output.item()}, expected {expected.item()} "
        f"(artifact generated under {parity_artifact['generated_under_backend']})"
    )
