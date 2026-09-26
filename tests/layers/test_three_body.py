from __future__ import annotations

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

import matgl
from matgl.graph._compute import compute_pair_vector_and_distance, create_line_graph
from matgl.layers import ThreeBodyInteractions
from matgl.models import M3GNet


def test_three_body_uses_global_neighbor_cutoffs_and_scatter_destination():
    """Reproduce the m3gnet-lite update using non-contiguous parent bond IDs."""
    layer = ThreeBodyInteractions(nn.Identity(), nn.Identity())
    edge_dst_atom = torch.tensor([1, 2, 3, 4, 5, 0])
    line_edge_index = torch.tensor(
        [[0, 0, 2, 2, 4, 4], [2, 4, 0, 4, 0, 2]],
        dtype=torch.long,
    )
    node_feat = torch.arange(1, 7, dtype=torch.float).unsqueeze(1)
    edge_feat = torch.zeros((6, 1))
    three_basis = torch.arange(1, 7, dtype=torch.float).unsqueeze(1)
    three_cutoff = torch.tensor([0.2, 0.0, 0.4, 0.0, 0.6, 0.8])

    actual = layer(
        edge_dst_atom,
        line_edge_index,
        torch.tensor([2, 0, 2, 0, 2, 0]),
        6,
        three_basis,
        three_cutoff,
        node_feat,
        edge_feat,
    )

    expected_messages = (
        three_basis
        * node_feat[edge_dst_atom[line_edge_index[1]]]
        * (three_cutoff[line_edge_index[0]] * three_cutoff[line_edge_index[1]]).unsqueeze(1)
    )
    expected = torch.zeros_like(edge_feat).index_add_(0, line_edge_index[0], expected_messages)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual[[1, 3, 5]], torch.zeros((3, 1)))


@pytest.mark.parametrize("num_bonds", [0, 2], ids=["isolated-atom", "dimer"])
def test_three_body_is_noop_without_triplets(num_bonds):
    """An isolated atom or dimer has no angle, so the three-body update is exactly zero."""
    layer = ThreeBodyInteractions(nn.Identity(), nn.Identity())
    edge_dst_atom = torch.empty(0, dtype=torch.long) if num_bonds == 0 else torch.tensor([1, 0])
    node_feat = torch.arange(4, dtype=torch.float).reshape(2, 2)
    edge_feat = torch.arange(2 * num_bonds, dtype=torch.float).reshape(num_bonds, 2)

    actual = layer(
        edge_dst_atom,
        torch.empty((2, 0), dtype=torch.long),
        torch.zeros(num_bonds, dtype=torch.long),
        num_bonds,
        torch.empty((0, 2)),
        torch.ones(num_bonds),
        node_feat,
        edge_feat,
    )

    assert actual is edge_feat
    torch.testing.assert_close(actual, edge_feat)


@pytest.mark.parametrize("use_cached_line_graph", [False, True], ids=["dynamic", "cached"])
def test_three_body_position_gradient_matches_finite_difference(monkeypatch, use_cached_line_graph):
    """The corrected three-body path must preserve coordinate gradients with or without cached topology."""
    monkeypatch.setattr(matgl, "float_th", torch.float64)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        torch.manual_seed(7)
        edge_index = torch.tensor([[0, 0, 0, 1, 2, 3], [1, 2, 3, 0, 0, 0]], dtype=torch.long)
        base_pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.1, 0.0], [0.2, 1.2, 0.1], [2.6, 0.2, 0.0]])
        node_type = torch.zeros(4, dtype=torch.long)
        model = M3GNet(
            element_types=("H",),
            dim_node_embedding=8,
            dim_edge_embedding=8,
            units=8,
            max_n=2,
            max_l=2,
            nblocks=1,
            cutoff=3.0,
            threebody_cutoff=2.0,
            is_intensive=False,
        ).eval()

        bond_vec, bond_dist = compute_pair_vector_and_distance(base_pos, edge_index, None)
        cached_line_graph = create_line_graph(edge_index, bond_dist, bond_vec, None, 4, 2.0)

        def energy(pos):
            graph = Data(pos=pos, edge_index=edge_index, node_type=node_type, num_nodes=4)
            return model(g=graph, l_g=cached_line_graph if use_cached_line_graph else None).sum()

        direction = torch.randn_like(base_pos)
        direction -= direction.mean(dim=0, keepdim=True)
        direction /= torch.linalg.vector_norm(direction)

        pos = base_pos.clone().requires_grad_(True)
        (gradient,) = torch.autograd.grad(energy(pos), pos)
        autodiff = torch.sum(gradient * direction)

        epsilon = 5e-5
        with torch.no_grad():
            finite_difference = (energy(base_pos + epsilon * direction) - energy(base_pos - epsilon * direction)) / (
                2 * epsilon
            )
        torch.testing.assert_close(autodiff, finite_difference, atol=1e-11, rtol=1e-5)
    finally:
        torch.set_default_dtype(old_dtype)
