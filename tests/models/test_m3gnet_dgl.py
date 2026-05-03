from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import matgl

if matgl.config.BACKEND != "DGL":
    pytest.skip("Skipping DGL tests", allow_module_level=True)
from matgl.models import M3GNet


def test_model(graph_MoS):
    _, graph, _ = graph_MoS
    for act in ["swish", "tanh", "sigmoid", "softplus2", "softexp"]:
        model = M3GNet(is_intensive=False, activation_type=act)
        output = model(g=graph)
        assert torch.numel(output) == 1
    model.save(".")
    M3GNet.load(".")
    os.remove("model.pt")
    os.remove("model.json")
    os.remove("state.pt")


def test_exceptions():
    with pytest.raises(ValueError, match="Invalid activation type"):
        _ = M3GNet(element_types=None, is_intensive=False, activation_type="whatever")
    with pytest.raises(ValueError, match=r"Classification task cannot be extensive."):
        _ = M3GNet(element_types=["Mo", "S"], is_intensive=False, task_type="classification")


def test_model_intensive(graph_MoS):
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
    graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
    model = M3GNet(element_types=["Mo", "S"], is_intensive=True)
    output = model(g=graph)
    assert torch.numel(output) == 1


def test_model_intensive_reduced_atom(graph_MoS):
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
    graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
    model = M3GNet(element_types=["Mo", "S"], is_intensive=True, readout_type="reduced_atom")
    output = model(g=graph)
    assert torch.numel(output) == 1


def test_model_intensive_with_classification(graph_MoS):
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
    graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
    model = M3GNet(
        element_types=["Mo", "S"],
        is_intensive=True,
        task_type="classification",
    )
    output = model(g=graph)
    assert torch.numel(output) == 1


def test_model_intensive_set2set_classification(graph_MoS):
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
    graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
    model = M3GNet(element_types=["Mo", "S"], is_intensive=True, task_type="classification", readout_type="set2set")
    output = model(g=graph)
    assert torch.numel(output) == 1


def test_predict_structure(graph_MoS):
    structure, _, _ = graph_MoS
    model_intensive = M3GNet(element_types=["Mo", "S"], is_intensive=True)
    model_extensive = M3GNet(is_intensive=False)
    for model in [model_extensive, model_intensive]:
        output_final = model.predict_structure(structure)
        assert torch.numel(output_final) == 1


def test_featurize_structure(graph_MoS):
    structure, _, _ = graph_MoS
    model_intensive = M3GNet(element_types=["Mo", "S"], is_intensive=True)
    model_extensive = M3GNet(is_intensive=False)
    for model in [model_extensive, model_intensive]:
        with pytest.raises(ValueError, match="Invalid output_layers"):
            model.predict_structure(structure, output_layers=["whatever"], return_features=True)
        features = model.predict_structure(structure, return_features=True)
        assert torch.numel(features["bond_expansion"]) == 252
        assert torch.numel(features["three_body_basis"]) == 3276
        for output_layer in ["embedding", "gc_1", "gc_2", "gc_3"]:
            assert torch.numel(features[output_layer]["node_feat"]) == 128
            assert torch.numel(features[output_layer]["edge_feat"]) == 1792
            assert features[output_layer]["state_feat"] is None
        if model.is_intensive:
            assert torch.numel(features["readout"]) == 64
        else:
            assert torch.numel(features["readout"]) == 2
        assert torch.numel(features["final"]) == 1
        assert list(model.predict_structure(structure, output_layers=["gc_1"], return_features=True).keys()) == ["gc_1"]
