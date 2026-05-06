"""Tests for the QET model on whichever backend is currently active.

DGL <-> PyG numerical-parity tests are also defined here and run only when both
backends are importable in the current environment.
"""

from __future__ import annotations

import importlib
import os

import numpy as np
import pytest
import torch

import matgl

BACKEND = matgl.config.BACKEND

if BACKEND == "DGL":
    from matgl.models._qet_dgl import QET
elif BACKEND == "PYG":
    from matgl.models._qet_pyg import QET  # type: ignore[assignment]
else:
    pytest.skip(f"Unsupported backend: {BACKEND}", allow_module_level=True)


def _has_dgl() -> bool:
    try:
        importlib.import_module("dgl")
    except Exception:  # DGL has many import-time failure modes (missing libs, version skew)
        return False
    return True


def _make_qet(**overrides):
    """Construct QET, suppressing the warp kernel on PYG so the pure-PyTorch path runs."""
    if BACKEND == "PYG":
        overrides.setdefault("use_warp", False)
    return QET(**overrides)


def test_qet(graph_MoS):
    """Forward across activations + save/load + SO(3) variant."""
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    expected = {
        "swish": torch.tensor(-0.0112),
        "tanh": torch.tensor(-0.1138),
        "sigmoid": torch.tensor(0.1377),
        "softplus2": torch.tensor(0.0127),
        "softexp": torch.tensor(0.0819),
    }

    _, graph, _ = graph_MoS

    activations = ["swish", "tanh", "sigmoid", "softplus2", "softexp"]

    for act in activations:
        model = _make_qet(is_intensive=False, activation_type=act)
        output = model(g=graph, total_charge=torch.tensor([0.0]))
        assert torch.numel(output) == 1
        if BACKEND == "DGL" and act in expected:
            assert torch.allclose(output, expected[act], atol=1e-4)

    model.save(".")
    QET.load(".")
    for fname in ("model.pt", "model.json", "state.pt"):
        os.remove(fname)

    model = _make_qet(is_intensive=False, equivariance_invariance_group="SO(3)")
    output = model(g=graph, total_charge=torch.tensor([0.0]))
    assert torch.numel(output) == 1


def test_qet_return_features(graph_MoS):
    """`return_features=True` returns (node_feat, atomic_energies) with the right shapes."""
    if BACKEND != "PYG":
        pytest.skip("`return_features` is currently only exposed by the PYG QET implementation.")
    torch.manual_seed(0)
    _, graph, _ = graph_MoS
    model = _make_qet(is_intensive=False, return_features=True)
    node_feat, atomic_energies = model(g=graph, total_charge=torch.tensor([0.0]))
    n_nodes = graph.pos.shape[0]
    # +1 charge, +1 elec_pot
    assert node_feat.shape == (n_nodes, model.units + 2)
    assert atomic_energies.shape[0] == n_nodes


def test_qet_include_magmom(graph_MoS):
    if BACKEND != "PYG":
        pytest.skip("`include_magmom` is currently only exposed by the PYG QET implementation.")
    torch.manual_seed(0)
    _, graph, _ = graph_MoS
    model = _make_qet(is_intensive=False, include_magmom=True, return_features=True)
    node_feat, _ = model(g=graph, total_charge=torch.tensor([0.0]))
    n_nodes = graph.pos.shape[0]
    # +1 charge, +1 elec_pot, +1 magmom
    assert node_feat.shape == (n_nodes, model.units + 3)


def test_qet_is_hardness_envs(graph_MoS):
    if BACKEND != "PYG":
        pytest.skip("`is_hardness_envs` is currently only exposed by the PYG QET implementation.")
    torch.manual_seed(0)
    _, graph, _ = graph_MoS
    model = _make_qet(is_intensive=False, is_hardness_envs=True)
    output = model(g=graph, total_charge=torch.tensor([0.0]))
    assert torch.numel(output) == 1


@pytest.mark.skipif(not _has_dgl(), reason="DGL not importable in this environment")
def test_qet_dgl_pyg_parity(MoS):
    """DGL and PyG QET produce equal energies on the same structure with shared weights."""
    if BACKEND != "PYG":
        pytest.skip("Cross-backend parity test is driven from the PYG side.")
    import dgl  # noqa: F401  (proves DGL is importable in this env)

    from matgl.ext._pymatgen_dgl import Structure2Graph as Structure2GraphDGL
    from matgl.ext._pymatgen_pyg import Structure2Graph as Structure2GraphPyG
    from matgl.models._qet_dgl import QET as QETDGL
    from matgl.models._qet_pyg import QET as QETPyG

    elements = ("Mo", "S")
    cutoff = 5.0

    conv_dgl = Structure2GraphDGL(element_types=elements, cutoff=cutoff)
    g_dgl, lat_dgl, _ = conv_dgl.get_graph(MoS)
    g_dgl.edata["pbc_offshift"] = torch.matmul(g_dgl.edata["pbc_offset"], lat_dgl[0])
    g_dgl.ndata["pos"] = g_dgl.ndata["frac_coords"] @ lat_dgl[0]

    conv_pyg = Structure2GraphPyG(element_types=elements, cutoff=cutoff)
    g_pyg, lat_pyg, _ = conv_pyg.get_graph(MoS)
    g_pyg.pbc_offshift = torch.matmul(g_pyg.pbc_offset, lat_pyg[0])
    g_pyg.pos = g_pyg.frac_coords @ lat_pyg[0]

    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    pyg_model = QETPyG(element_types=elements, is_intensive=False, cutoff=cutoff, use_warp=False).eval()
    torch.manual_seed(42)
    dgl_model = QETDGL(element_types=elements, is_intensive=False, cutoff=cutoff).eval()

    missing, _unexpected = dgl_model.load_state_dict(pyg_model.state_dict(), strict=False)
    trainable_keys_pyg = {k for k, v in pyg_model.state_dict().items() if v.dtype.is_floating_point}
    not_loaded = trainable_keys_pyg.intersection(missing)
    assert not not_loaded, f"Trainable PyG keys missing from DGL state_dict: {sorted(not_loaded)[:5]}"

    e_pyg = pyg_model(g=g_pyg, total_charge=torch.tensor([0.0]))
    e_dgl = dgl_model(g=g_dgl, total_charge=torch.tensor([0.0]))
    assert torch.allclose(e_pyg, e_dgl, atol=1e-4), f"PyG={e_pyg.item()} vs DGL={e_dgl.item()}"


@pytest.mark.skipif(not _has_dgl(), reason="DGL not importable in this environment")
def test_qet_dgl_pyg_training_parity(MoS):
    """Train QET DGL and QET PyG on a 5-structure MoS toy dataset and assert
    per-structure energies / forces stay equal across backends."""
    if BACKEND != "PYG":
        pytest.skip("Cross-backend training-parity test is driven from the PYG side.")
    import dgl  # noqa: F401

    from matgl.ext._pymatgen_dgl import Structure2Graph as Structure2GraphDGL
    from matgl.ext._pymatgen_pyg import Structure2Graph as Structure2GraphPyG
    from matgl.models._qet_dgl import QET as QETDGL
    from matgl.models._qet_pyg import QET as QETPyG

    elements = ("Mo", "S")
    cutoff = 5.0
    n_structures = 5
    n_epochs = 10
    seed = 0

    rng = np.random.default_rng(seed)
    structures, ref_q, ref_E, ref_F = [], [], [], []
    for i in range(n_structures):
        s = MoS.copy()
        if i > 0:
            for site in s:
                site.frac_coords = site.frac_coords + rng.uniform(-0.02, 0.02, 3)
        structures.append(s)
        eps = 0.0 if i == 0 else float(rng.uniform(-0.05, 0.05))
        ref_q.append(np.array([4.0 + eps, -2.0 - eps], dtype=np.float64))
        ref_E.append(0.0 if i == 0 else float(rng.uniform(-0.1, 0.1)))
        ref_F.append(rng.uniform(-0.05, 0.05, (2, 3)) if i > 0 else np.zeros((2, 3), dtype=np.float64))

    ref_E_t = [torch.tensor(e, dtype=torch.get_default_dtype()) for e in ref_E]
    ref_F_t = [torch.tensor(f, dtype=torch.get_default_dtype()) for f in ref_F]
    total_q = [torch.tensor([float(q.sum())], dtype=torch.get_default_dtype()) for q in ref_q]

    conv_dgl = Structure2GraphDGL(element_types=elements, cutoff=cutoff)
    conv_pyg = Structure2GraphPyG(element_types=elements, cutoff=cutoff)

    def _dgl_graphs():
        out = []
        for s in structures:
            g, lat, _ = conv_dgl.get_graph(s)
            g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lat[0])
            out.append((g, lat))
        return out

    def _pyg_graphs():
        out = []
        for s in structures:
            g, lat, _ = conv_pyg.get_graph(s)
            g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
            out.append((g, lat))
        return out

    def _forward_dgl(model, g, lat, tot_q):
        pos = (g.ndata["frac_coords"] @ lat[0]).detach().clone().requires_grad_(True)
        g.ndata["pos"] = pos
        e = model(g=g, total_charge=tot_q)
        (grads,) = torch.autograd.grad(e.sum(), pos, create_graph=True, retain_graph=True)
        return e, -grads

    def _forward_pyg(model, g, lat, tot_q):
        pos = (g.frac_coords @ lat[0]).detach().clone().requires_grad_(True)
        g.pos = pos
        e = model(g=g, total_charge=tot_q)
        (grads,) = torch.autograd.grad(e.sum(), pos, create_graph=True, retain_graph=True)
        return e, -grads

    def _train(model, fwd, graphs, lr=1e-3):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(n_epochs):
            opt.zero_grad()
            loss = torch.zeros(())
            for i, (g, lat) in enumerate(graphs):
                e_p, f_p = fwd(model, g, lat, total_q[i])
                loss = loss + (e_p.squeeze() - ref_E_t[i]) ** 2 + ((f_p - ref_F_t[i]) ** 2).mean()
            loss.backward()
            opt.step()

    def _predict(model, fwd, graphs):
        es, fs = [], []
        for i, (g, lat) in enumerate(graphs):
            e_p, f_p = fwd(model, g, lat, total_q[i])
            es.append(e_p.detach().clone())
            fs.append(f_p.detach().clone())
        return es, fs

    torch.manual_seed(seed)
    pyg_model = QETPyG(element_types=elements, is_intensive=False, cutoff=cutoff, use_warp=False).train()
    torch.manual_seed(seed)
    dgl_model = QETDGL(element_types=elements, is_intensive=False, cutoff=cutoff).train()
    missing, _ = dgl_model.load_state_dict(pyg_model.state_dict(), strict=False)
    trainable_pyg = {k for k, v in pyg_model.state_dict().items() if v.dtype.is_floating_point}
    not_loaded = trainable_pyg.intersection(missing)
    assert not not_loaded, f"Trainable PyG keys missing from DGL state_dict: {sorted(not_loaded)[:5]}"

    _train(pyg_model, _forward_pyg, _pyg_graphs())
    _train(dgl_model, _forward_dgl, _dgl_graphs())

    e_pyg, f_pyg = _predict(pyg_model, _forward_pyg, _pyg_graphs())
    e_dgl, f_dgl = _predict(dgl_model, _forward_dgl, _dgl_graphs())

    max_de = max(float((ep - ed).abs().max()) for ep, ed in zip(e_pyg, e_dgl, strict=True))
    max_df = max(float((fp - fd).abs().max()) for fp, fd in zip(f_pyg, f_dgl, strict=True))
    assert max_de < 1e-6, f"max |dE| after training = {max_de:.3e}"
    assert max_df < 1e-6, f"max |dF| after training = {max_df:.3e}"
