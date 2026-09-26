from __future__ import annotations

import numpy as np
import pytest
import torch

import matgl
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph._compute import (
    compute_pair_vector_and_distance,
    compute_theta_and_phi,
    create_line_graph,
    create_line_graph_torch,
    ensure_line_graph_compatibility,
    separate_node_edge_keys,
)


def _loop_indices(bond_atom_indices, pair_dist, cutoff=4.0):
    bin_count = np.bincount(bond_atom_indices[:, 0], minlength=bond_atom_indices[-1, 0] + 1)
    indices = []
    start = 0
    for bcont in bin_count:
        for i in range(bcont):
            for j in range(bcont):
                if start + i == start + j:
                    continue
                if pair_dist[start + i] > cutoff or pair_dist[start + j] > cutoff:
                    continue
                indices.append([start + i, start + j])
        start += bcont
    return np.array(indices)


def _calculate_cos_loop(graph, threebody_cutoff=4.0):
    """
    Calculate the cosine theta of triplets using loops
    Args:
        graph: List
    Returns: a list of cosine theta values.
    """
    _, _, n_sites = torch.unique(graph.edge_index[0], return_inverse=True, return_counts=True)
    start_index = 0
    cos = []
    for n_site in n_sites:
        for i in range(n_site):
            for j in range(n_site):
                if i == j:
                    continue
                vi = graph.bond_vec[i + start_index].detach().numpy()
                vj = graph.bond_vec[j + start_index].detach().numpy()
                di = np.linalg.norm(vi)
                dj = np.linalg.norm(vj)
                if (di <= threebody_cutoff) and (dj <= threebody_cutoff):
                    cos.append(vi.dot(vj) / np.linalg.norm(vi) / np.linalg.norm(vj))
        start_index += n_site
    return cos


class TestCompute:
    def test_compute_pair_vector(self, graph_Mo):
        s1, g1, _ = graph_Mo
        lattice = torch.tensor(s1.lattice.matrix, dtype=matgl.float_th, device=g1.pos.device).unsqueeze(dim=0)
        g1.pbc_offshift = torch.matmul(g1.pbc_offset, lattice[0])
        g1.pos = g1.frac_coords @ lattice[0]
        bv, _ = compute_pair_vector_and_distance(g1.pos, g1.edge_index, g1.pbc_offshift)
        g1.bond_vec = bv
        d = torch.linalg.norm(g1.bond_vec, axis=1)

        _, _, _, d2 = s1.get_neighbor_list(r=5.0)

        np.testing.assert_array_almost_equal(np.sort(d.cpu().numpy()), np.sort(d2))

    def test_compute_pair_vector_for_molecule(self, graph_CH4):
        _, g2, _ = graph_CH4
        lattice = torch.tensor(np.identity(3), dtype=matgl.float_th, device=g2.pos.device).unsqueeze(dim=0)
        g2.pbc_offshift = torch.matmul(g2.pbc_offset, lattice[0])
        g2.pos = g2.frac_coords @ lattice[0]
        bv, _ = compute_pair_vector_and_distance(g2.pos, g2.edge_index, g2.pbc_offshift)
        g2.bond_vec = bv
        d = torch.linalg.norm(g2.bond_vec, axis=1).cpu().numpy()

        d2 = np.array(
            [
                1.089,
                1.089,
                1.089,
                1.089,
                1.089,
                1.089,
                1.089,
                1.089,
                1.77833,
                1.77833,
                1.77833,
                1.77833,
                1.77833,
                1.77833,
                1.77833,
                1.77833,
                1.77833,
                1.77833,
                1.77833,
                1.77833,
            ]
        )

        np.testing.assert_array_almost_equal(np.sort(d), np.sort(d2))


class TestSeparateNodeEdgeKeys:
    """Coverage for ``separate_node_edge_keys`` which classifies tensors on a Data object."""

    def test_partitions_node_edge_and_other_keys(self, graph_LiFePO4):
        _, g, _ = graph_LiFePO4
        # Inject extra attributes representing each bucket.
        g.scalar_meta = torch.tensor(1.0)  # 0-dim → other
        g.misshape = torch.zeros(2, 4)  # leading dim matches neither N nor E → other
        node_keys, edge_keys, other_keys = separate_node_edge_keys(g)

        assert "edge_index" in other_keys
        assert "scalar_meta" in other_keys
        assert "misshape" in other_keys
        # Standard PyG/MatGL fields land in their canonical buckets.
        assert "node_type" in node_keys or "frac_coords" in node_keys
        assert "pbc_offset" in edge_keys
        assert set(node_keys).isdisjoint(edge_keys)
        assert set(node_keys).isdisjoint(other_keys)
        assert set(edge_keys).isdisjoint(other_keys)

    def test_node_count_collision_prefers_node_bucket(self):
        """When num_nodes happens to equal num_edges, the first matching branch wins."""
        from torch_geometric.data import Data

        # 2 nodes, 2 edges → both buckets share the same first-dim size.
        d = Data(
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            node_attr=torch.zeros(2, 4),  # matches num_nodes first → node bucket
            edge_attr=torch.zeros(2, 5),  # matches num_edges first → still node bucket (collision)
        )
        node_keys, edge_keys, other_keys = separate_node_edge_keys(d)
        assert set(node_keys) == {"node_attr", "edge_attr"}
        assert edge_keys == []
        assert "edge_index" in other_keys


class TestCreateDirectedLineGraph:
    """Unit tests for create_directed_line_graph autograd correctness (#834)."""

    def test_line_graph_preserves_autograd_gradients(self):
        """Continuous geometry features must retain autograd connection to pos."""
        from matgl.graph._compute import create_directed_line_graph

        pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]], requires_grad=True)
        edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 0]], dtype=torch.long)
        pbc_offshift = torch.zeros((4, 3))
        pbc_offset = torch.zeros((4, 3), dtype=torch.long)

        bond_vec, bond_dist = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)
        lg_edge_index, lg_bond_vec, lg_bond_dist, lg_pbc_offset, lg_src_bond_sign = create_directed_line_graph(
            edge_index, pbc_offset, bond_vec, bond_dist, threebody_cutoff=3.0
        )

        # 1. Verify continuous features have autograd history
        assert lg_bond_vec.requires_grad
        assert lg_bond_dist.requires_grad
        assert lg_bond_vec.grad_fn is not None
        assert lg_bond_dist.grad_fn is not None

        # 2. Verify discrete topology tensors are not tracked
        assert not lg_edge_index.requires_grad
        assert not lg_pbc_offset.requires_grad
        assert not lg_src_bond_sign.requires_grad

        # 3. Verify gradients propagate back to positions
        loss = lg_bond_vec.sum() + lg_bond_dist.sum()
        loss.backward()
        assert pos.grad is not None
        assert torch.isfinite(pos.grad).all()
        assert (pos.grad.abs() > 0).any()


def _pruned_graph(structure, cutoff=5.0):
    """Graph whose cutoff exceeds the 4 A three-body cutoff, so the line graph is built on a pruned bond list."""
    g, lat, _ = Structure2Graph(element_types=get_element_list([structure]), cutoff=cutoff).get_graph(structure)
    g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
    g.pos = g.frac_coords @ lat[0]
    g.bond_vec, g.bond_dist = compute_pair_vector_and_distance(g.pos, g.edge_index, g.pbc_offshift)
    return g


class TestCreateLineGraph:
    """M3GNet line graphs must index parent-graph bonds, as in the reference TF implementation."""

    @pytest.mark.parametrize("builder", ["numpy", "torch"])
    def test_line_graph_indexes_parent_bonds(self, LiFePO4, builder):
        g = _pruned_graph(LiFePO4)
        threebody_cutoff = 4.0
        n_bonds = g.edge_index.size(1)
        assert (g.bond_dist > threebody_cutoff).any(), "test requires bonds beyond the three-body cutoff"

        if builder == "numpy":
            l_g = create_line_graph(g.edge_index, g.bond_dist, g.bond_vec, g.pbc_offset, g.num_nodes, threebody_cutoff)
        else:
            l_g = create_line_graph_torch(g.edge_index, g.bond_dist, g.bond_vec, g.num_nodes, threebody_cutoff)
        line_edge_index = l_g["line_edge_index"].long()

        # Every triplet (j, i, k) pairs two distinct parent bonds i->j and i->k within the three-body cutoff.
        expected = _loop_indices(g.edge_index.T.numpy(), g.bond_dist.detach().numpy(), cutoff=threebody_cutoff)
        np.testing.assert_array_equal(line_edge_index.T.numpy(), expected)
        assert torch.equal(g.edge_index[0][line_edge_index[0]], g.edge_index[0][line_edge_index[1]])

        # One triple count per parent bond; zero beyond the three-body cutoff.
        n_triple_ij = l_g["n_triple_ij"].long()
        assert n_triple_ij.numel() == n_bonds
        assert torch.equal(n_triple_ij, torch.bincount(line_edge_index[0], minlength=n_bonds))
        assert (n_triple_ij[g.bond_dist > threebody_cutoff] == 0).all()

        # Angles evaluated on parent bond vectors match an explicit loop.
        cos_theta = compute_theta_and_phi(g.bond_vec, g.bond_dist, line_edge_index)["cos_theta"]
        np.testing.assert_allclose(
            cos_theta.detach().numpy(), _calculate_cos_loop(g, threebody_cutoff), rtol=1e-5, atol=1e-6
        )

    def test_torch_and_numpy_builders_agree(self, LiFePO4):
        g = _pruned_graph(LiFePO4)
        l_np = create_line_graph(g.edge_index, g.bond_dist, g.bond_vec, g.pbc_offset, g.num_nodes, 4.0)
        l_th = create_line_graph_torch(g.edge_index, g.bond_dist, g.bond_vec, g.num_nodes, 4.0)
        assert torch.equal(l_np["line_edge_index"].long(), l_th["line_edge_index"].long())
        assert torch.equal(l_np["n_triple_ij"].long(), l_th["n_triple_ij"].long())
        assert torch.equal(l_np["kept_edge_ids"], l_th["kept_edge_ids"])

    def test_ensure_line_graph_compatibility(self, LiFePO4):
        g = _pruned_graph(LiFePO4)
        l_g = create_line_graph(g.edge_index, g.bond_dist, g.bond_vec, g.pbc_offset, g.num_nodes, 4.0)
        new_vec = g.bond_vec * 1.01
        new_dist = g.bond_dist * 1.01
        refreshed = ensure_line_graph_compatibility(l_g, new_dist, new_vec, g.pbc_offset, 4.0)
        assert refreshed["bond_vec"] is new_vec
        assert refreshed["bond_dist"] is new_dist
        assert torch.equal(refreshed["line_edge_index"], l_g["line_edge_index"])

        # A bundle that indexes a different (e.g. pruned) bond list is rejected.
        stale = dict(l_g, n_triple_ij=l_g["n_triple_ij"][l_g["kept_edge_ids"]])
        with pytest.raises(ValueError, match="must index parent-graph bonds"):
            ensure_line_graph_compatibility(stale, g.bond_dist, g.bond_vec, g.pbc_offset, 4.0)
