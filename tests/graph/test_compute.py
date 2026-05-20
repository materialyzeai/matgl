from __future__ import annotations

import numpy as np
import torch

import matgl
from matgl.graph._compute import (
    compute_pair_vector_and_distance,
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
