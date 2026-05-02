from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("Skipping PYG tests", allow_module_level=True)

from matgl.models._tensornet_pyg import TensorNet, _warp_available
from matgl.utils.io import _get_file_paths

if not _warp_available:
    pytest.skip("Skipping warp tests: nvalchemiops not installed", allow_module_level=True)


def test_model(graph_MoS_pyg):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    # Optional regression-check values
    EXPECTED = {
        "swish": torch.tensor(0.0827),
        "tanh": torch.tensor(-0.0258),
        "sigmoid": torch.tensor(0.0360),
        "softplus2": torch.tensor(0.1165),
        "softexp": torch.tensor(0.1100),
    }

    _, graph, _ = graph_MoS_pyg

    activations = ["swish", "tanh", "sigmoid", "softplus2", "softexp"]

    outputs = {}
    for act in activations:
        model = TensorNet(is_intensive=False, activation_type=act, use_warp=True)

        output = model(g=graph)
        print(act, output.item())

        assert torch.numel(output) == 1

        # Optional strict regression test
        if act in EXPECTED:
            assert torch.allclose(output, EXPECTED[act], atol=1e-4)

        outputs[act] = output.item()

    # ---- SAVE/LOAD TEST ----
    model.save(".")
    TensorNet.load(".")
    os.remove("model.pt")
    os.remove("model.json")
    os.remove("state.pt")

    # ---- SECOND MODEL TEST ----
    model = TensorNet(is_intensive=False, equivariance_invariance_group="SO(3)")
    output = model(g=graph)

    # this model outputs a 2-vector (as you wanted)
    assert torch.numel(output) == 1


def test_exceptions():
    with pytest.raises(ValueError, match="Invalid activation type"):
        _ = TensorNet(element_types=None, is_intensive=False, activation_type="whatever")
    with pytest.raises(ValueError, match=r"Classification task cannot be extensive."):
        _ = TensorNet(element_types=["Mo", "S"], is_intensive=False, task_type="classification")


def test_model_intensive(graph_MoS_pyg):
    structure, graph, _ = graph_MoS_pyg
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]
    model = TensorNet(element_types=["Mo", "S"], is_intensive=True)
    output = model(g=graph)
    assert torch.allclose(output, torch.tensor([-0.0906]), atol=1e-4)


def test_model_intensive_with_weighted_atom(graph_MoS_pyg):
    structure, graph, _ = graph_MoS_pyg
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]
    model = TensorNet(element_types=["Mo", "S"], is_intensive=True, readout_type="weighted_atom")
    output = model(g=graph)
    assert torch.allclose(output, torch.tensor([-0.0210]), atol=1e-4)


def test_model_intensive_with_ReduceReadOut(graph_MoS_pyg):
    structure, graph, _ = graph_MoS_pyg
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]
    model = TensorNet(is_intensive=True, readout_type="reduce_atom")
    output = model(g=graph)
    assert torch.allclose(output, torch.tensor([-0.1075]), atol=1e-4)


def test_model_intensive_with_classification(graph_MoS_pyg):
    structure, graph, _ = graph_MoS_pyg
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]
    model = TensorNet(
        element_types=["Mo", "S"],
        is_intensive=True,
        task_type="classification",
    )
    output = model(g=graph)
    assert torch.numel(output) == 1


def test_backward(graph_MoS_pyg):
    """Test cell gradient (dE/dcell)."""
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    EXPECTED_CELL_GRAD = torch.tensor(
        [
            [-0.000909, 0.000000, 0.000000],
            [0.000000, -0.000909, 0.000000],
            [0.000000, 0.000000, -0.000909],
        ]
    )

    structure, graph, _ = graph_MoS_pyg
    cell = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th).requires_grad_(True)

    graph.pbc_offshift = torch.matmul(graph.pbc_offset, cell)
    graph.pos = graph.frac_coords @ cell

    model = TensorNet(is_intensive=False, activation_type="swish")
    model.train()

    energy = model(g=graph)
    cell_grad = torch.autograd.grad(energy, cell, create_graph=True)[0]

    assert torch.allclose(cell_grad, EXPECTED_CELL_GRAD, atol=1e-6)


def test_double_backward(graph_MoS_pyg):
    """Test double backward: loss = sum(cell_grad^2), compare cell.grad."""
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    EXPECTED_CELL_GRAD2 = torch.tensor(
        [
            [-0.0000037, -0.000000, -0.000000],
            [-0.000000, -0.0000037, -0.000000],
            [-0.000000, -0.000000, -0.0000037],
        ]
    )

    structure, graph, _ = graph_MoS_pyg
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

    assert torch.allclose(cell.grad, EXPECTED_CELL_GRAD2, atol=1e-6)


def _build_pair_from_pretrained(repo_id: str) -> tuple[TensorNet, TensorNet]:
    """Build a (warp, non-warp) pair of TensorNet models loaded with identical pretrained weights."""
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


def test_warp_pyg_parity_pretrained(MoS):
    """Warp and non-warp TensorNet must produce identical outputs from the same pretrained weights."""
    model_warp, model_pyg = _build_pair_from_pretrained("materialyze/TensorNet-PES-MatPES-PBE-2025.2")

    from matgl.ext._pymatgen_pyg import Structure2Graph

    converter = Structure2Graph(element_types=model_pyg.element_types, cutoff=model_pyg.cutoff)
    g, lat, _ = converter.get_graph(MoS)
    g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
    g.pos = g.frac_coords @ lat[0]

    with torch.no_grad():
        out_warp = model_warp(g=g)
        out_pyg = model_pyg(g=g)

    assert torch.allclose(out_warp, out_pyg, atol=1e-5, rtol=1e-5), (
        f"warp={out_warp.detach().cpu()} vs pyg={out_pyg.detach().cpu()}"
    )
