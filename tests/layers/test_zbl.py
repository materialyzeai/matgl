"""United tests for ZBL repulsion."""

from __future__ import annotations

import pytest
import torch

import matgl

BACKEND = matgl.config.BACKEND

if BACKEND == "DGL":
    import dgl

    from matgl.layers import NuclearRepulsion
elif BACKEND == "PYG":
    from torch_geometric.data import Data

    from matgl.layers._zbl_pyg import NuclearRepulsion
else:
    pytest.skip(f"Unsupported backend: {BACKEND}", allow_module_level=True)


@pytest.fixture
def example_data():
    if BACKEND == "DGL":
        element_types = "H"
        g = dgl.graph(([0, 1], [1, 0]))
        g.ndata["node_type"] = torch.tensor([0, 0], dtype=matgl.int_th)
        g.edata["bond_dist"] = torch.tensor([1.0, 1.0], dtype=matgl.float_th)
        return element_types, g

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
