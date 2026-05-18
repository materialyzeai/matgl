from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import matgl

if matgl.config.BACKEND != "DGL":
    pytest.skip("Skipping DGL tests", allow_module_level=True)
from matgl.models import SO3Net


def test_model(graph_MoS):
    _, graph, _ = graph_MoS
    for act in ["swish", "tanh", "sigmoid", "softplus2", "softexp"]:
        model = SO3Net(is_intensive=False, activation_type=act)
        output = model(g=graph)
        assert torch.numel(output) == 1
    model.save(".")
    SO3Net.load(".")
    os.remove("model.pt")
    os.remove("model.json")
    os.remove("state.pt")


def test_exceptions():
    with pytest.raises(ValueError, match="Invalid activation type"):
        _ = SO3Net(element_types=None, is_intensive=False, activation_type="whatever")
    with pytest.raises(ValueError, match=r"Classification task cannot be extensive."):
        _ = SO3Net(element_types=["Mo", "S"], is_intensive=False, task_type="classification")


def test_model_intensive(graph_MoS):
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
    graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
    model = SO3Net(element_types=["Mo", "S"], is_intensive=True)
    output = model(g=graph)
    assert torch.numel(output) == 2


def test_model_intensive_reduce_atom_classification(graph_MoS):
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
    graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
    model = SO3Net(element_types=["Mo", "S"], is_intensive=True, readout_type="reduce_atom", target_property="graph")
    output = model(g=graph)
    assert torch.numel(output) == 1


def test_model_intensive_weighted_atom_classification(graph_MoS):
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
    graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
    model = SO3Net(element_types=["Mo", "S"], is_intensive=True, task_type="classification", target_property="graph")
    output = model(g=graph)
    assert torch.numel(output) == 1


def test_model_intensive_set2set_classification(graph_MoS):
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
    graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
    model = SO3Net(
        element_types=["Mo", "S"],
        is_intensive=True,
        task_type="classification",
        readout_type="set2set",
        target_property="graph",
    )
    output = model(g=graph)
    assert torch.numel(output) == 1


def test_model_dipole_moment(graph_MoS):
    _, graph, _ = graph_MoS
    model = SO3Net(element_types=["Mo", "S"], target_property="dipole_moment", return_vector_representation=True)
    charges, dipole_moment = model(g=graph)
    assert torch.numel(charges) == 2
    assert dipole_moment.shape == torch.Size([2, 3])
    # correct charges
    model = SO3Net(
        element_types=["Mo", "S"],
        target_property="dipole_moment",
        return_vector_representation=True,
        correct_charges=True,
    )
    charges, dipole_moment = model(g=graph, total_charges=torch.Tensor([0.0]))
    assert torch.numel(charges) == 2
    assert dipole_moment.shape == torch.Size([2, 3])


def test_model_dipole_moment_including_use_vector_representation(graph_MoS):
    _, graph, _ = graph_MoS
    model = SO3Net(
        element_types=["Mo", "S"],
        target_property="dipole_moment",
        use_vector_representation=True,
        return_vector_representation=True,
    )
    charges, dipole_moment = model(g=graph)
    assert torch.numel(charges) == 2
    assert dipole_moment.shape == torch.Size([3])
    # predict magnatide and chare corrections
    model = SO3Net(
        element_types=["Mo", "S"],
        target_property="dipole_moment",
        use_vector_representation=True,
        return_vector_representation=True,
        predict_dipole_magnitude=True,
        correct_charges=True,
    )
    charges, dipole_moment = model(g=graph, total_charges=torch.Tensor([0.0]))
    assert torch.numel(charges) == 2
    assert torch.numel(dipole_moment) == 1


def test_model_polarizability(graph_MoS):
    _, graph, _ = graph_MoS
    model = SO3Net(
        element_types=["Mo", "S"],
        target_property="polarizability",
        use_vector_representation=True,
        return_vector_representation=True,
    )
    alpha = model(g=graph)
    assert alpha.shape == torch.Size([3, 3])
