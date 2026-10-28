"""Unit tests for ZBL repulsion."""

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

import matgl
from matgl.layers._zbl import NuclearRepulsion


@pytest.fixture
def example_data():
    element_types = ("H",)
    data = Data()
    data.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data.num_nodes = 2
    data.node_type = torch.tensor([0, 0], dtype=matgl.int_th)
    data.bond_dist = torch.tensor([1.0, 1.0], dtype=matgl.float_th)
    data.batch = torch.tensor([0, 0], dtype=torch.long)
    return element_types, data


def test_nuclear_repulsion(example_data):
    element_types, graph = example_data
    nuclear_repulsion = NuclearRepulsion(r_cut=3.0, trainable=True)
    energy = nuclear_repulsion(element_types, graph)
    assert energy.shape in (torch.Size([]), torch.Size([1]))


def test_nuclear_repulsion_trailing_isolated_atom():
    """Atoms without neighbours at the end of the node list must not break the per-graph sum."""
    data = Data()
    data.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    data.num_nodes = 3
    data.node_type = torch.tensor([0, 0, 0], dtype=matgl.int_th)
    data.bond_dist = torch.tensor([1.0, 1.0], dtype=matgl.float_th)
    data.batch = torch.tensor([0, 0, 0], dtype=torch.long)
    energy = NuclearRepulsion(r_cut=3.0)(("H",), data)
    expected = NuclearRepulsion(r_cut=3.0)(("H",), Data(**{**data.to_dict(), "num_nodes": 2, "batch": data.batch[:2]}))
    assert torch.allclose(energy, expected)
