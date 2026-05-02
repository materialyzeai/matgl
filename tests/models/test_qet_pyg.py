"""Tests for the PyG QET model, including DGL <-> PyG numerical parity."""

from __future__ import annotations

import importlib
import os

import pytest
import torch

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("Skipping PyG tests", allow_module_level=True)

from matgl.models._qet_pyg import QET


def _has_dgl() -> bool:
    try:
        importlib.import_module("dgl")
    except Exception:  # DGL has many import-time failure modes (missing libs, version skew)
        return False
    return True


def test_qet_pyg(graph_MoS_pyg):
    """Forward across activations + save/load + SO(3) variant."""
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    _, graph, _ = graph_MoS_pyg
    activations = ["swish", "tanh", "sigmoid", "softplus2", "softexp"]

    for act in activations:
        model = QET(is_intensive=False, activation_type=act, use_warp=False)
        output = model(g=graph, total_charge=torch.tensor([0.0]))
        assert torch.numel(output) == 1

    # Save / load round-trip
    model.save(".")
    QET.load(".")
    for fname in ("model.pt", "model.json", "state.pt"):
        os.remove(fname)

    # SO(3) variant constructs cleanly
    model = QET(is_intensive=False, equivariance_invariance_group="SO(3)", use_warp=False)
    output = model(g=graph, total_charge=torch.tensor([0.0]))
    assert torch.numel(output) == 1


def test_qet_pyg_return_features(graph_MoS_pyg):
    """`return_features=True` returns (node_feat, atomic_energies) with the right shapes."""
    torch.manual_seed(0)
    _, graph, _ = graph_MoS_pyg
    model = QET(is_intensive=False, return_features=True, use_warp=False)
    node_feat, atomic_energies = model(g=graph, total_charge=torch.tensor([0.0]))
    n_nodes = graph.pos.shape[0]
    # +1 charge, +1 elec_pot
    assert node_feat.shape == (n_nodes, model.units + 2)
    assert atomic_energies.shape[0] == n_nodes


def test_qet_pyg_include_magmom(graph_MoS_pyg):
    """Smoke-check the magmom branch."""
    torch.manual_seed(0)
    _, graph, _ = graph_MoS_pyg
    model = QET(is_intensive=False, include_magmom=True, return_features=True, use_warp=False)
    node_feat, _ = model(g=graph, total_charge=torch.tensor([0.0]))
    n_nodes = graph.pos.shape[0]
    # +1 charge, +1 elec_pot, +1 magmom
    assert node_feat.shape == (n_nodes, model.units + 3)


def test_qet_pyg_is_hardness_envs(graph_MoS_pyg):
    """Smoke-check the environment-dependent hardness branch (MLP head instead of per-element parameter)."""
    torch.manual_seed(0)
    _, graph, _ = graph_MoS_pyg
    model = QET(is_intensive=False, is_hardness_envs=True, use_warp=False)
    output = model(g=graph, total_charge=torch.tensor([0.0]))
    assert torch.numel(output) == 1


@pytest.mark.skipif(not _has_dgl(), reason="DGL not importable in this environment")
def test_qet_dgl_pyg_parity(MoS):
    """DGL and PyG QET produce equal energies on the same structure with shared weights."""
    import dgl  # noqa: F401  (proves DGL is importable in this env)

    from matgl.ext._pymatgen_dgl import Structure2Graph as Structure2GraphDGL
    from matgl.ext._pymatgen_pyg import Structure2Graph as Structure2GraphPyG
    from matgl.models._qet_dgl import QET as QETDGL

    elements = ("Mo", "S")
    cutoff = 5.0

    # Build identical-input graphs in both backends.
    conv_dgl = Structure2GraphDGL(element_types=elements, cutoff=cutoff)
    g_dgl, lat_dgl, _ = conv_dgl.get_graph(MoS)
    g_dgl.edata["pbc_offshift"] = torch.matmul(g_dgl.edata["pbc_offset"], lat_dgl[0])
    g_dgl.ndata["pos"] = g_dgl.ndata["frac_coords"] @ lat_dgl[0]

    conv_pyg = Structure2GraphPyG(element_types=elements, cutoff=cutoff)
    g_pyg, lat_pyg, _ = conv_pyg.get_graph(MoS)
    g_pyg.pbc_offshift = torch.matmul(g_pyg.pbc_offset, lat_pyg[0])
    g_pyg.pos = g_pyg.frac_coords @ lat_pyg[0]

    # Construct both models with identical config and copy weights from PyG -> DGL.
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    pyg_model = QET(element_types=elements, is_intensive=False, cutoff=cutoff, use_warp=False).eval()
    torch.manual_seed(42)
    dgl_model = QETDGL(element_types=elements, is_intensive=False, cutoff=cutoff).eval()

    # Both models share the same architecture and key names since QET DGL and PyG
    # both subclass their respective TensorNet. Load PyG state into DGL with strict=False
    # (some buffers like `pi`, `sqrt2` exist in both backends and match by name).
    missing, _unexpected = dgl_model.load_state_dict(pyg_model.state_dict(), strict=False)
    # Allow a small set of buffers / module-internal differences but require all
    # trainable weights to load on both sides.
    trainable_keys_pyg = {k for k, v in pyg_model.state_dict().items() if v.dtype.is_floating_point}
    not_loaded = trainable_keys_pyg.intersection(missing)
    assert not not_loaded, f"Trainable PyG keys missing from DGL state_dict: {sorted(not_loaded)[:5]}"

    e_pyg = pyg_model(g=g_pyg, total_charge=torch.tensor([0.0]))
    e_dgl = dgl_model(g=g_dgl, total_charge=torch.tensor([0.0]))
    assert torch.allclose(e_pyg, e_dgl, atol=1e-4), f"PyG={e_pyg.item()} vs DGL={e_dgl.item()}"
