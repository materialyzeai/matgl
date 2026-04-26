from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("Skipping PYG tests", allow_module_level=True)
from matgl.models._qet_pyg import QET


def test_qet(graph_MoS_pyg):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    EXPECTED = {
        "swish": torch.tensor(-0.0510),
        "tanh": torch.tensor(0.1544),
        "sigmoid": torch.tensor(-0.1195),
        "softplus2": torch.tensor(-0.0799),
        "softexp": torch.tensor(-0.0234),
    }

    _, graph, _ = graph_MoS_pyg

    activations = ["swish", "tanh", "sigmoid", "softplus2", "softexp"]

    outputs = {}
    for act in activations:
        model = QET(is_intensive=False, activation_type=act)

        output = model(g=graph, total_charge=torch.tensor([0.0]))
        assert torch.numel(output) == 1

        if act in EXPECTED:
            assert torch.allclose(output, EXPECTED[act], atol=1e-4)

        outputs[act] = output.item()

    # ---- SAVE/LOAD TEST ----
    model.save(".")
    QET.load(".")
    os.remove("model.pt")
    os.remove("model.json")
    os.remove("state.pt")

    model = QET(is_intensive=False, equivariance_invariance_group="SO(3)")
    output = model(g=graph, total_charge=torch.tensor([0.0]))
    assert torch.numel(output) == 1


def test_qet_exceptions():
    with pytest.raises(ValueError, match="Invalid activation type"):
        _ = QET(element_types=None, is_intensive=False, activation_type="whatever")


def test_qet_charge_conservation(graph_MoS_pyg):
    """Outputs with different total charges must differ."""
    torch.manual_seed(0)
    _, graph, _ = graph_MoS_pyg
    model = QET(is_intensive=False)
    out_0 = model(g=graph, total_charge=torch.tensor([0.0]))
    out_1 = model(g=graph, total_charge=torch.tensor([1.0]))
    assert not torch.allclose(out_0, out_1)


def test_qet_return_features(graph_MoS_pyg):
    """return_features=True returns (node_feat, atomic_energy) with correct shapes."""
    torch.manual_seed(0)
    _, graph, _ = graph_MoS_pyg
    model = QET(is_intensive=False, return_features=True)
    node_feat, atomic_energy = model(g=graph, total_charge=torch.tensor([0.0]))
    num_nodes = graph.node_type.shape[0]
    assert node_feat.shape[0] == num_nodes
    assert atomic_energy.shape[0] == num_nodes


def test_qet_hardness_envs(graph_MoS_pyg):
    """Environment-dependent hardness (MLP) produces a scalar output."""
    torch.manual_seed(0)
    _, graph, _ = graph_MoS_pyg
    model = QET(is_intensive=False, is_hardness_envs=True)
    output = model(g=graph, total_charge=torch.tensor([0.0]))
    assert torch.numel(output) == 1


def test_qet_sigma_train(graph_MoS_pyg):
    """Trainable sigma appears in model.parameters() and produces a scalar output."""
    torch.manual_seed(0)
    _, graph, _ = graph_MoS_pyg
    model = QET(is_intensive=False, is_sigma_train=True)
    output = model(g=graph, total_charge=torch.tensor([0.0]))
    assert torch.numel(output) == 1
    assert any("sigma" in name for name, _ in model.named_parameters())


def test_qet_include_magmom(graph_MoS_pyg):
    """include_magmom=True produces a scalar output."""
    torch.manual_seed(0)
    _, graph, _ = graph_MoS_pyg
    model = QET(is_intensive=False, include_magmom=True)
    output = model(g=graph, total_charge=torch.tensor([0.0]))
    assert torch.numel(output) == 1


def test_qet_ext_pot(graph_MoS_pyg):
    """Non-zero external potential (ext_pot) shifts chi and changes the output."""
    torch.manual_seed(0)
    _, graph, _ = graph_MoS_pyg
    model = QET(is_intensive=False)
    num_nodes = graph.node_type.shape[0]

    out_no_ext = model(g=graph, total_charge=torch.tensor([0.0]))
    ext_pot = torch.zeros(num_nodes)
    ext_pot[0] = 0.5
    out_ext = model(g=graph, total_charge=torch.tensor([0.0]), ext_pot=ext_pot)
    assert not torch.allclose(out_no_ext, out_ext)


def test_qet_backward(graph_MoS_pyg):
    """Cell gradient (dE/dcell) is computable and non-zero."""
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    structure, graph, _ = graph_MoS_pyg
    cell = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th).requires_grad_(True)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, cell)
    graph.pos = graph.frac_coords @ cell

    model = QET(is_intensive=False, activation_type="swish")
    model.train()

    energy = model(g=graph, total_charge=torch.tensor([0.0]))
    (cell_grad,) = torch.autograd.grad(energy, cell, create_graph=False)

    assert cell_grad.shape == cell.shape
    assert not torch.all(cell_grad == 0)


def test_qet_real_structure(graph_MoS_pyg):
    """Forward pass with explicit lattice-derived positions works correctly."""
    torch.manual_seed(0)
    structure, graph, _ = graph_MoS_pyg
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]
    model = QET(element_types=["Mo", "S"], is_intensive=False)
    output = model(g=graph, total_charge=torch.tensor([0.0]))
    assert torch.numel(output) == 1
