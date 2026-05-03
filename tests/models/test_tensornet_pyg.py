from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("Skipping PYG tests", allow_module_level=True)

from matgl.models._tensornet_pyg import TensorNet, _warp_available

if _warp_available:
    pytest.skip(
        "Skipping PYG tests: TensorNet with warp kernel will be tested with test_tensornet_warp",
        allow_module_level=True,
    )


def _check_scalar_output(output):
    """Validate that an output is a finite scalar tensor."""
    assert torch.numel(output) == 1
    assert torch.isfinite(output).all()


def test_model(graph_MoS_pyg):
    torch.manual_seed(0)

    _, graph, _ = graph_MoS_pyg

    activations = ["swish", "tanh", "sigmoid", "softplus2", "softexp"]

    for act in activations:
        model = TensorNet(is_intensive=False, activation_type=act, use_warp=False)
        model.to(graph.pos.device)

        output = model(g=graph)
        _check_scalar_output(output)

    # ---- SAVE/LOAD TEST ----
    model.save(".")
    TensorNet.load(".")
    os.remove("model.pt")
    os.remove("model.json")
    os.remove("state.pt")

    # ---- SECOND MODEL TEST ----
    model = TensorNet(is_intensive=False, equivariance_invariance_group="SO(3)")
    model.to(graph.pos.device)
    output = model(g=graph)

    _check_scalar_output(output)


def test_exceptions():
    with pytest.raises(ValueError, match="Invalid activation type"):
        _ = TensorNet(element_types=None, is_intensive=False, activation_type="whatever")
    with pytest.raises(ValueError, match=r"Classification task cannot be extensive."):
        _ = TensorNet(element_types=["Mo", "S"], is_intensive=False, task_type="classification")


def test_model_intensive(graph_MoS_pyg):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS_pyg
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=graph.pos.device)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]
    model = TensorNet(element_types=["Mo", "S"], is_intensive=True, use_warp=False)
    model.to(graph.pos.device)
    output = model(g=graph)
    _check_scalar_output(output)


def test_model_intensive_with_weighted_atom(graph_MoS_pyg):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS_pyg
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=graph.pos.device)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]
    model = TensorNet(element_types=["Mo", "S"], is_intensive=True, readout_type="weighted_atom", use_warp=False)
    model.to(graph.pos.device)
    output = model(g=graph)
    _check_scalar_output(output)


def test_model_intensive_with_ReduceReadOut(graph_MoS_pyg):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS_pyg
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=graph.pos.device)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]
    model = TensorNet(is_intensive=True, readout_type="reduce_atom", use_warp=False)
    model.to(graph.pos.device)
    output = model(g=graph)
    _check_scalar_output(output)


def test_model_intensive_with_classification(graph_MoS_pyg):
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS_pyg
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=graph.pos.device)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]
    model = TensorNet(element_types=["Mo", "S"], is_intensive=True, task_type="classification", use_warp=False)
    model.to(graph.pos.device)
    output = model(g=graph)
    _check_scalar_output(output)
    # Classification output should be a probability via sigmoid.
    assert 0.0 <= output.item() <= 1.0
