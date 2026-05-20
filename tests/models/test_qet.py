"""Tests for the QET model."""

from __future__ import annotations

import os

import torch

from matgl.models._qet import QET


def _make_qet(**overrides):
    """Construct QET, suppressing the warp kernel so the pure-PyTorch path runs."""
    overrides.setdefault("use_warp", False)
    return QET(**overrides)


def test_qet(graph_MoS):
    """Forward across activations + save/load + SO(3) variant."""
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    _, graph, _ = graph_MoS

    activations = ["swish", "tanh", "sigmoid", "softplus2", "softexp"]

    for act in activations:
        model = _make_qet(is_intensive=False, activation_type=act)
        output = model(g=graph, total_charge=torch.tensor([0.0]))
        assert torch.numel(output) == 1

    model.save(".")
    QET.load(".")
    for fname in ("model.pt", "model.json", "state.pt"):
        os.remove(fname)

    model = _make_qet(is_intensive=False, equivariance_invariance_group="SO(3)")
    output = model(g=graph, total_charge=torch.tensor([0.0]))
    assert torch.numel(output) == 1


def test_qet_return_features(graph_MoS):
    """`return_features=True` returns (node_feat, atomic_energies) with the right shapes."""
    torch.manual_seed(0)
    _, graph, _ = graph_MoS
    model = _make_qet(is_intensive=False, return_features=True)
    node_feat, atomic_energies = model(g=graph, total_charge=torch.tensor([0.0]))
    n_nodes = graph.pos.shape[0]
    # +1 charge, +1 elec_pot
    assert node_feat.shape == (n_nodes, model.units + 2)
    assert atomic_energies.shape[0] == n_nodes


def test_qet_include_magmom(graph_MoS):
    torch.manual_seed(0)
    _, graph, _ = graph_MoS
    model = _make_qet(is_intensive=False, include_magmom=True, return_features=True)
    node_feat, _ = model(g=graph, total_charge=torch.tensor([0.0]))
    n_nodes = graph.pos.shape[0]
    # +1 charge, +1 elec_pot, +1 magmom
    assert node_feat.shape == (n_nodes, model.units + 3)


def test_qet_is_hardness_envs(graph_MoS):
    torch.manual_seed(0)
    _, graph, _ = graph_MoS
    model = _make_qet(is_intensive=False, is_hardness_envs=True)
    output = model(g=graph, total_charge=torch.tensor([0.0]))
    assert torch.numel(output) == 1
