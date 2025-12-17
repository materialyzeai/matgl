from __future__ import annotations

import numpy as np
import pytest
import torch

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("Skipping PYG tests", allow_module_level=True)
from matgl.graph._compute_pyg import (
    compute_pair_vector_and_distance,
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
    def test_compute_pair_vector(self, graph_Mo_pyg):
        s1, g1, _ = graph_Mo_pyg
        lattice = torch.tensor(s1.lattice.matrix, dtype=matgl.float_th, device=g1.pos.device).unsqueeze(dim=0)
        g1.pbc_offshift = torch.matmul(g1.pbc_offset, lattice[0])
        g1.pos = g1.frac_coords @ lattice[0]
        bv, _ = compute_pair_vector_and_distance(g1.pos, g1.edge_index, g1.pbc_offshift)
        g1.bond_vec = bv
        d = torch.linalg.norm(g1.bond_vec, axis=1)

        _, _, _, d2 = s1.get_neighbor_list(r=5.0)

        np.testing.assert_array_almost_equal(np.sort(d.cpu().numpy()), np.sort(d2))

    def test_compute_pair_vector_for_molecule(self, graph_CH4_pyg):
        _, g2, _ = graph_CH4_pyg
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
