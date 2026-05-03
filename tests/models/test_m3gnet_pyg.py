from __future__ import annotations

import os

import numpy as np
import pytest
import torch as th

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("Skipping PYG tests", allow_module_level=True)

from matgl.graph._compute_pyg import compute_pair_vector_and_distance
from matgl.models import M3GNet


def _prep_graph(graph, structure):
    lat = th.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.pbc_offshift = th.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]
    bond_vec, bond_dist = compute_pair_vector_and_distance(graph.pos, graph.edge_index, graph.pbc_offshift)
    graph.bond_vec = bond_vec
    graph.bond_dist = bond_dist
    return graph


def test_model(graph_MoS_pyg):
    structure, graph, _ = graph_MoS_pyg
    graph = _prep_graph(graph, structure)
    for act in ["swish", "tanh", "sigmoid", "softplus2", "softexp"]:
        model = M3GNet(is_intensive=False, activation_type=act)
        output = model(g=graph)
    assert th.numel(output) == 1


def test_exceptions():
    with pytest.raises(ValueError, match="Invalid activation type"):
        _ = M3GNet(element_types=None, is_intensive=False, activation_type="whatever")
    with pytest.raises(ValueError, match=r"Classification task cannot be extensive."):
        _ = M3GNet(element_types=["Mo", "S"], is_intensive=False, task_type="classification")


def test_model_intensive(graph_MoS_pyg):
    structure, graph, _ = graph_MoS_pyg
    graph = _prep_graph(graph, structure)
    model = M3GNet(element_types=["Mo", "S"], is_intensive=True)
    output = model(g=graph)
    assert th.numel(output) == 1


def test_model_intensive_reduce_atom(graph_MoS_pyg):
    structure, graph, _ = graph_MoS_pyg
    graph = _prep_graph(graph, structure)
    model = M3GNet(element_types=["Mo", "S"], is_intensive=True, readout_type="reduce_atom")
    output = model(g=graph)
    assert th.numel(output) == 1


def test_model_intensive_set2set_classification(graph_MoS_pyg):
    structure, graph, _ = graph_MoS_pyg
    graph = _prep_graph(graph, structure)
    model = M3GNet(
        element_types=["Mo", "S"],
        is_intensive=True,
        readout_type="set2set",
        task_type="classification",
        niters_set2set=2,
        nlayers_set2set=1,
    )
    output = model(g=graph)
    assert th.numel(output) == 1


def test_predict_structure(MoS):
    model = M3GNet(element_types=["Mo", "S"], is_intensive=False)
    output = model.predict_structure(MoS)
    assert th.numel(output) == 1


def test_save_load(tmp_path):
    model = M3GNet(element_types=("Mo", "S"), is_intensive=True)
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        model.save(".")
        M3GNet.load(".")
    finally:
        os.chdir(cwd)
