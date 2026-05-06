from __future__ import annotations

import numpy as np
import pytest
import torch

import matgl

BACKEND = matgl.config.BACKEND

if BACKEND == "DGL":
    import dgl

    from matgl.layers._atom_ref_dgl import AtomRef
elif BACKEND == "PYG":
    from torch_geometric.data import Batch

    from matgl.layers._atom_ref_pyg import AtomRef
else:
    pytest.skip(f"Unsupported backend: {BACKEND}", allow_module_level=True)


def _batch(graphs):
    return dgl.batch(graphs) if BACKEND == "DGL" else Batch.from_data_list(graphs)


def test_atom_ref(graph_MoSH):
    _, g1, _ = graph_MoSH
    element_ref = AtomRef(torch.tensor([0.5, 1.0, 2.0]))
    assert element_ref(g1) == 3.5


def test_atom_ref_without_property_offset(graph_MoSH):
    _, g1, _ = graph_MoSH
    element_ref = AtomRef()
    assert element_ref(g1) == 0.0


def test_atom_ref_property_offset_as_list(graph_MoSH):
    _, g1, _ = graph_MoSH
    element_ref = AtomRef([0.5, 1.0, 2.0])
    assert element_ref(g1) == 3.5


def test_atom_ref_fit(graph_MoSH):
    _, g1, _ = graph_MoSH
    element_ref = AtomRef(torch.tensor([0.5, 1.0, 2.0]))
    properties = torch.tensor([2.0, 2.0])
    bg = _batch([g1, g1])
    element_ref.fit([g1, g1], properties)

    atom_ref = element_ref(bg)
    assert list(np.round(atom_ref.numpy())) == [2.0, 2.0]


def test_atom_ref_with_states(graph_MoSH):
    _, g1, _ = graph_MoSH
    element_ref = AtomRef(torch.tensor([[0.5, 1.0, 2.0], [2.0, 3.0, 5.0]]))
    state_label = torch.tensor([1])
    assert element_ref(g1, state_label) == 10
