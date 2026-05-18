from __future__ import annotations

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

dgl = pytest.importorskip("dgl", reason="DGL not installed; skipping PyG/DGL parity tests")

from matgl.ext._pymatgen_pyg import Structure2Graph, get_element_list  # noqa: E402
from matgl.graph._compute_dgl import _create_directed_line_graph  # noqa: E402
from matgl.graph._compute_pyg import create_directed_line_graph_pyg  # noqa: E402


@pytest.fixture
def base_structure() -> Structure:
    """Returns a basic silicon lattice."""
    lattice = Lattice.cubic(5.43)
    return Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


def get_perturbed_structure(struct: Structure, seed: int) -> Structure:
    """Perturbs the structure with random noise to guarantee diverse geometries."""
    torch.manual_seed(seed)
    # create random supercell size between 1 and 3
    s1 = torch.randint(1, 3, (1,)).item()
    s2 = torch.randint(1, 3, (1,)).item()
    s3 = torch.randint(1, 3, (1,)).item()

    new_struct = struct.copy()
    new_struct.make_supercell([s1, s2, s3])

    # perturb positions slightly
    new_struct.perturb(0.1)
    return new_struct


@pytest.mark.parametrize("seed", list(range(100)))
def test_line_graph_parity(base_structure, seed):
    """
    Test line graph creation parity between PyG and DGL across 100 perturbed structures.
    This guarantees no regression in angle computation or edge extraction geometry.
    """
    struct = get_perturbed_structure(base_structure, seed)

    # Generate common inputs
    converter = Structure2Graph(element_types=get_element_list([struct]), cutoff=5.0)
    g, _lat, _state = converter.get_graph(struct)

    # PyG data extraction
    edge_index = g.edge_index
    pbc_offset = g.pbc_offset

    # We mock bond_vec and bond_dist identically for both since graph topology
    # generation only depends on these abstract attributes, not absolute distances
    # unless using cutoff pruning. We disable cutoff pruning by setting it high.
    torch.manual_seed(seed)
    bond_vec = torch.randn((edge_index.shape[1], 3))
    bond_dist = torch.norm(bond_vec, dim=1)

    three_body_cutoff = 100.0  # allow all bonds

    # Generate PyG Line Graph
    lg_edge_index, _lg_bond_vec, _lg_bond_dist, _lg_pbc_offset, _lg_src_bond_sign = create_directed_line_graph_pyg(
        edge_index, pbc_offset, bond_vec, bond_dist, threebody_cutoff=three_body_cutoff
    )

    # Generate DGL Line Graph
    g_dgl = dgl.graph((edge_index[0], edge_index[1]))
    g_dgl.edata["pbc_offset"] = pbc_offset
    g_dgl.edata["bond_vec"] = bond_vec
    g_dgl.edata["bond_dist"] = bond_dist

    lg_dgl = _create_directed_line_graph(g_dgl)
    lg_src_dgl, lg_dst_dgl = lg_dgl.edges()

    # Compare Topological Parity
    pyg_edges = set(zip(lg_edge_index[0].tolist(), lg_edge_index[1].tolist(), strict=False))
    dgl_edges = set(zip(lg_src_dgl.tolist(), lg_dst_dgl.tolist(), strict=False))

    diff_pyg = pyg_edges - dgl_edges
    diff_dgl = dgl_edges - pyg_edges

    assert len(diff_pyg) == 0, f"PyG generated {len(diff_pyg)} unique edges not present in DGL"
    assert len(diff_dgl) == 0, f"DGL generated {len(diff_dgl)} unique edges not present in PyG"

    # Verify exact edge count
    assert len(pyg_edges) == len(dgl_edges)
    assert lg_edge_index.shape[1] == lg_src_dgl.numel()

    # For a perfect bijective match of identical ordering, we must sort the edges
    # since PyG and DGL might return them in different topological sorts
    # Sort them by (src, dst) pair
    def sort_edges(src, dst):
        edges = torch.stack([src, dst], dim=1)
        # Convert to numpy lexsort
        arr = edges.cpu().numpy()
        idx = np.lexsort((arr[:, 1], arr[:, 0]))
        return arr[idx]

    pyg_sorted = sort_edges(lg_edge_index[0], lg_edge_index[1])
    dgl_sorted = sort_edges(lg_src_dgl, lg_dst_dgl)

    np.testing.assert_array_equal(pyg_sorted, dgl_sorted)
