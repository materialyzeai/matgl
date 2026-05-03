from __future__ import annotations

import os

import numpy as np
import pytest
import torch as th
from pymatgen.core import Lattice, Structure

import matgl

if matgl.config.BACKEND != "DGL":
    pytest.skip("Skipping DGL tests", allow_module_level=True)
from matgl.graph._compute_dgl import compute_pair_vector_and_distance
from matgl.models import MEGNet


def test_megnet(graph_MoS):
    structure, graph, state = graph_MoS
    lat = th.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.edata["pbc_offshift"] = th.matmul(graph.edata["pbc_offset"], lat[0])
    graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
    _, bond_dist = compute_pair_vector_and_distance(graph)
    graph.edata["bond_dist"] = bond_dist
    for act in ["tanh", "sigmoid", "softplus2", "softexp", "swish"]:
        model = MEGNet(
            dim_node_embedding=16,
            dim_edge_embedding=100,
            dim_state_embedding=2,
            nblocks=3,
            include_states=True,
            hidden_layer_sizes_input=(64, 32),
            hidden_layer_sizes_conv=(64, 64, 32),
            activation_type=act,
            nlayers_set2set=4,
            niters_set2set=3,
            hidden_layer_sizes_output=(32, 16),
            is_classification=True,
        )
        state = th.tensor(np.array(state))
        output = model(g=graph, state_attr=state)
    with pytest.raises(ValueError, match="Invalid activation type"):
        _ = MEGNet(is_intensive=False, activation_type="whatever")
    assert [th.numel(output)] == [1]


def test_megnet_isolated_atom(graph_MoS):
    structure = Structure(Lattice.cubic(10.0), ["Mo"], [[0.0, 0, 0]])

    model = MEGNet(
        dim_node_embedding=16,
        dim_edge_embedding=100,
        dim_state_embedding=2,
        nblocks=3,
        include_states=True,
        hidden_layer_sizes_input=(64, 32),
        hidden_layer_sizes_conv=(64, 64, 32),
        activation_type="swish",
        nlayers_set2set=4,
        niters_set2set=3,
        hidden_layer_sizes_output=(32, 16),
        is_classification=True,
        dropout=0.1,
    )

    output = model.predict_structure(structure)
    assert [th.numel(output)] == [1]


def test_save_load():
    model = MEGNet(
        dim_node_embedding=16,
        dim_edge_embedding=100,
        dim_state_embedding=2,
        nblocks=3,
        include_states=True,
        hidden_layer_sizes_input=(64, 32),
        hidden_layer_sizes_conv=(64, 64, 32),
        activation_type="swish",
        nlayers_set2set=4,
        niters_set2set=3,
        hidden_layer_sizes_output=(32, 16),
        is_classification=True,
        dropout=0.1,
    )
    model.save(".", metadata={"description": "forme model"})
    MEGNet.load(".")
    os.remove("model.pt")
    os.remove("model.json")
    os.remove("state.pt")
