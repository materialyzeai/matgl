from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

import matgl

BACKEND = matgl.config.BACKEND

_warp_available: bool = False
if BACKEND == "DGL":
    from matgl.models._tensornet_dgl import TensorNet  # type: ignore[assignment]
elif BACKEND == "PYG":
    from matgl.models._tensornet_pyg import TensorNet, _warp_available  # type: ignore[assignment,no-redef]
else:
    pytest.skip(f"Unsupported backend: {BACKEND}", allow_module_level=True)


def _set_pos_and_pbc(graph, lat):
    """Attach `pos` and `pbc_offshift` to `graph` for the active backend."""
    if BACKEND == "DGL":
        graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
        graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
    else:
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
        graph.pos = graph.frac_coords @ lat[0]


def _device_of(graph):
    if BACKEND == "DGL":
        return graph.device
    return graph.pos.device


def _make_tensornet(**kwargs):
    """Construct TensorNet with `use_warp=False` on PYG so non-warp paths are exercised."""
    if BACKEND == "PYG":
        kwargs.setdefault("use_warp", False)
    return TensorNet(**kwargs)


def _check_scalar_output(output):
    assert torch.numel(output) == 1
    assert torch.isfinite(output).all()


def test_model(graph_MoS):
    torch.manual_seed(0)

    _, graph, _ = graph_MoS

    activations = ["swish", "tanh", "sigmoid", "softplus2", "softexp"]
    for act in activations:
        model = _make_tensornet(is_intensive=False, activation_type=act)
        if BACKEND == "PYG":
            model.to(_device_of(graph))
        output = model(g=graph)
        _check_scalar_output(output)

    model.save(".")
    TensorNet.load(".")
    os.remove("model.pt")
    os.remove("model.json")
    os.remove("state.pt")

    model = _make_tensornet(is_intensive=False, equivariance_invariance_group="SO(3)")
    if BACKEND == "PYG":
        model.to(_device_of(graph))
    output = model(g=graph)
    _check_scalar_output(output)


def test_exceptions():
    with pytest.raises(ValueError, match="Invalid activation type"):
        _ = _make_tensornet(element_types=None, is_intensive=False, activation_type="whatever")
    with pytest.raises(ValueError, match=r"Classification task cannot be extensive."):
        _ = _make_tensornet(element_types=["Mo", "S"], is_intensive=False, task_type="classification")


def test_model_intensive(graph_MoS):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=_device_of(graph))
    _set_pos_and_pbc(graph, lat)
    model = _make_tensornet(element_types=["Mo", "S"], is_intensive=True)
    if BACKEND == "PYG":
        model.to(_device_of(graph))
    output = model(g=graph)
    _check_scalar_output(output)


def test_model_intensive_with_weighted_atom(graph_MoS):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=_device_of(graph))
    _set_pos_and_pbc(graph, lat)
    model = _make_tensornet(element_types=["Mo", "S"], is_intensive=True, readout_type="weighted_atom")
    if BACKEND == "PYG":
        model.to(_device_of(graph))
    output = model(g=graph)
    _check_scalar_output(output)


def test_model_intensive_with_ReduceReadOut(graph_MoS):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=_device_of(graph))
    _set_pos_and_pbc(graph, lat)
    model = _make_tensornet(is_intensive=True, readout_type="reduce_atom")
    if BACKEND == "PYG":
        model.to(_device_of(graph))
    output = model(g=graph)
    _check_scalar_output(output)


def test_model_intensive_with_classification(graph_MoS):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=_device_of(graph))
    _set_pos_and_pbc(graph, lat)
    model = _make_tensornet(element_types=["Mo", "S"], is_intensive=True, task_type="classification")
    if BACKEND == "PYG":
        model.to(_device_of(graph))
    output = model(g=graph)
    _check_scalar_output(output)
    if BACKEND == "PYG":
        assert 0.0 <= output.item() <= 1.0


def test_model_intensive_set2set_classification(graph_MoS):
    if BACKEND != "DGL":
        pytest.skip("set2set readout in TensorNet is currently only validated for the DGL backend.")
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    _set_pos_and_pbc(graph, lat)
    model = _make_tensornet(
        element_types=["Mo", "S"], is_intensive=True, task_type="classification", readout_type="set2set"
    )
    output = model(g=graph)
    _check_scalar_output(output)


def test_return_features(graph_MoS):
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=_device_of(graph))
    _set_pos_and_pbc(graph, lat)

    model = _make_tensornet(element_types=["Mo", "S"], is_intensive=True)
    if BACKEND == "PYG":
        model.to(_device_of(graph))

    out = model.predict_structure(structure, return_features=False)
    assert isinstance(out, torch.Tensor)

    out_feats = model.predict_structure(structure, return_features=True)
    assert isinstance(out_feats, dict)
    assert "final" in out_feats
    assert "readout" in out_feats
    assert "edge_attr" in out_feats
    assert "embedding" in out_feats
    assert "gc_1" in out_feats

    assert out_feats["final"].shape == torch.Size([])
    assert out_feats["readout"].shape[0] == structure.num_sites

    out_feats_subset = model.predict_structure(structure, return_features=True, output_layers=["final", "gc_1"])
    assert set(out_feats_subset.keys()) == {"final", "gc_1"}


# ---------------------------------------------------------------------------
# Warp-only regression and parity tests (PYG backend, nvalchemiops installed)
# ---------------------------------------------------------------------------


def _skip_if_no_warp():
    if BACKEND != "PYG" or not _warp_available:
        pytest.skip("Warp tests require the PYG backend with nvalchemiops installed.")


def test_warp_model_regression(graph_MoS):
    _skip_if_no_warp()
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    expected = {
        "swish": torch.tensor(0.0827),
        "tanh": torch.tensor(-0.0258),
        "sigmoid": torch.tensor(0.0360),
        "softplus2": torch.tensor(0.1165),
        "softexp": torch.tensor(0.1100),
    }

    _, graph, _ = graph_MoS
    for act, ref in expected.items():
        model = TensorNet(is_intensive=False, activation_type=act, use_warp=True)
        output = model(g=graph)
        assert torch.allclose(output, ref, atol=1e-4)


def test_warp_model_intensive(graph_MoS):
    _skip_if_no_warp()
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]
    model = TensorNet(element_types=["Mo", "S"], is_intensive=True, use_warp=True)
    output = model(g=graph)
    _check_scalar_output(output)


def test_warp_model_intensive_with_weighted_atom(graph_MoS):
    _skip_if_no_warp()
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]
    model = TensorNet(element_types=["Mo", "S"], is_intensive=True, readout_type="weighted_atom", use_warp=True)
    output = model(g=graph)
    _check_scalar_output(output)


def test_warp_model_intensive_with_ReduceReadOut(graph_MoS):
    _skip_if_no_warp()
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]
    model = TensorNet(is_intensive=True, readout_type="reduce_atom", use_warp=True)
    output = model(g=graph)
    _check_scalar_output(output)


def test_warp_backward(graph_MoS):
    _skip_if_no_warp()
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    expected_cell_grad = torch.tensor(
        [
            [-0.000909, 0.000000, 0.000000],
            [0.000000, -0.000909, 0.000000],
            [0.000000, 0.000000, -0.000909],
        ]
    )

    structure, graph, _ = graph_MoS
    cell = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th).requires_grad_(True)

    graph.pbc_offshift = torch.matmul(graph.pbc_offset, cell)
    graph.pos = graph.frac_coords @ cell

    model = TensorNet(is_intensive=False, activation_type="swish")
    model.train()

    energy = model(g=graph)
    cell_grad = torch.autograd.grad(energy, cell, create_graph=True)[0]

    assert torch.allclose(cell_grad, expected_cell_grad, atol=1e-6)


def test_warp_double_backward(graph_MoS):
    _skip_if_no_warp()
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    expected_cell_grad2 = torch.tensor(
        [
            [-0.0000037, -0.000000, -0.000000],
            [-0.000000, -0.0000037, -0.000000],
            [-0.000000, -0.000000, -0.0000037],
        ]
    )

    structure, graph, _ = graph_MoS
    cell = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th).requires_grad_(True)
    cell.retain_grad()

    graph.pbc_offshift = torch.matmul(graph.pbc_offset, cell)
    graph.pos = graph.frac_coords @ cell

    model = TensorNet(is_intensive=False, activation_type="swish")
    model.train()

    energy = model(g=graph)
    cell_grad = torch.autograd.grad(energy, cell, create_graph=True)[0]
    loss = (cell_grad * cell_grad).sum()
    loss.backward()

    assert torch.allclose(cell.grad, expected_cell_grad2, atol=1e-6)


def _build_pair_from_pretrained(repo_id: str) -> tuple[TensorNet, TensorNet]:
    from matgl.utils.io import _get_file_paths

    fpaths = _get_file_paths(Path(repo_id))
    map_location = "cpu" if not torch.cuda.is_available() else None
    state = torch.load(fpaths["state.pt"], map_location=map_location, weights_only=False)
    init_blob = torch.load(fpaths["model.pt"], map_location=map_location, weights_only=False)
    inner_init_args = dict(init_blob["model"]["init_args"])

    inner_state = {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}

    model_warp = TensorNet(**{**inner_init_args, "use_warp": True})
    model_pyg = TensorNet(**{**inner_init_args, "use_warp": False})
    model_warp.load_state_dict(inner_state, strict=False)
    model_pyg.load_state_dict(inner_state, strict=False)
    model_warp.eval()
    model_pyg.eval()
    return model_warp, model_pyg


@pytest.mark.parametrize("structure_fixture", ["MoS", "LiFePO4", "Li3InCl6"])
def test_warp_pyg_parity_pretrained(structure_fixture, request):
    _skip_if_no_warp()
    structure = request.getfixturevalue(structure_fixture)
    model_warp, model_pyg = _build_pair_from_pretrained("materialyze/TensorNet-PES-MatPES-PBE-2025.2")

    from matgl.ext._pymatgen_pyg import Structure2Graph

    converter = Structure2Graph(element_types=model_pyg.element_types, cutoff=model_pyg.cutoff)
    g, lat, _ = converter.get_graph(structure)
    g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
    g.pos = g.frac_coords @ lat[0]

    with torch.no_grad():
        out_warp = model_warp(g=g)
        out_pyg = model_pyg(g=g)

    assert torch.allclose(out_warp, out_pyg, atol=1e-5, rtol=1e-5), (
        f"{structure_fixture}: warp={out_warp.detach().cpu()} vs pyg={out_pyg.detach().cpu()}"
    )
