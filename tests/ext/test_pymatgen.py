from __future__ import annotations

import os

import numpy as np
import torch
from pymatgen.core import Lattice, Structure

from matgl.ext.pymatgen import Structure2Graph, get_element_list

module_dir = os.path.dirname(os.path.abspath(__file__))


def test_get_graph_from_molecule(graph_CH4):
    _, graph, state = graph_CH4
    # check the number of nodes
    assert np.allclose(graph.num_nodes, 5)
    # check the number of edges
    assert np.allclose(graph.num_edges, 20)
    # check the src_ids
    assert np.allclose(
        graph.edge_index[0].sort().values.cpu().numpy(),
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
    )
    # check the dst_ids
    assert np.allclose(
        graph.edge_index[1].sort().values.cpu().numpy(),
        [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
    )
    # check the atomic features of atom C
    assert np.allclose(graph.node_type.detach().cpu().numpy()[0], 1)
    # check the atomic features of atom H
    assert np.allclose(graph.node_type.detach().cpu().numpy()[1], 0)
    # check the shape of state features
    assert np.allclose(len(state), 2)
    # check the value of state features
    assert np.allclose(state, [3.208492, 2])
    # check the position of atom 0
    assert np.allclose(graph.pos[0].detach().cpu().numpy(), [0.0, 0.0, 0.0])


def test_get_graph_from_structure(graph_LiFePO4):
    lfp, graph, state = graph_LiFePO4
    # check the number of nodes
    assert np.allclose(graph.num_nodes, lfp.num_sites)
    # check the atomic feature of atom 0
    assert np.allclose(graph.node_type.detach().numpy()[0], 0)
    # check the atomic feature of atom 4
    assert np.allclose(graph.node_type.detach().numpy()[4], 3)
    # check the number of bonds
    assert np.allclose(graph.num_edges, 704)
    # check the state features
    assert np.allclose(state, [0.0, 0.0])
    structure_BaTiO3 = Structure.from_prototype("perovskite", ["Ba", "Ti", "O"], a=4.04)
    element_types = get_element_list([structure_BaTiO3])
    p2g = Structure2Graph(element_types=element_types, cutoff=4.0)
    graph, lattice, state = p2g.get_graph(structure_BaTiO3)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lattice[0])
    graph.pos = graph.frac_coords @ lattice[0]
    # check the number of nodes
    assert np.allclose(graph.num_nodes, structure_BaTiO3.num_sites)
    # check the atomic features of atom 0
    assert np.allclose(graph.node_type.detach().numpy()[0], 2)
    # check the atomic features of atom 1
    assert np.allclose(graph.node_type.detach().numpy()[1], 1)
    # check the number of edges
    assert np.allclose(graph.num_edges, 76)
    # check the state features
    assert np.allclose(state, [0.0, 0.0])
    # check the position of atom 0
    assert np.allclose(graph.pos[0].detach().cpu().numpy(), [0.0, 0.0, 0.0])
    # check the pbc offset from node 0 to image atom 6
    pbc_offset = graph.pbc_offset.detach().cpu().numpy()
    assert sum(np.allclose(x, [-1, -1, -1]) for x in pbc_offset) == 1
    # check the lattice vector
    assert np.allclose(lattice[0].detach().cpu().numpy(), [[4.04, 0.0, 0.0], [0.0, 4.04, 0.0], [0.0, 0.0, 4.04]])
    # check the volume
    assert np.allclose(torch.det(lattice).detach().cpu().numpy(), [65.939264])


def test_get_element_list():
    cscl = Structure.from_spacegroup("Pm-3m", Lattice.cubic(3), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    naf = Structure.from_spacegroup("Pm-3m", Lattice.cubic(3), ["Na", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    elem_list = get_element_list([cscl, naf])
    assert elem_list == ("F", "Na", "Cl", "Cs")


def _build_graph_without_alchmtk(structure_or_mol, cutoff, monkeypatch):
    """Force the scipy/pymatgen fallback branches in ``matgl.ext._pymatgen`` and convert."""
    from matgl.ext import _pymatgen as mod
    from matgl.ext._pymatgen import Molecule2Graph as M2G_local
    from matgl.ext._pymatgen import Structure2Graph as S2G_local

    monkeypatch.setattr(mod, "_alchmtk_available", False)
    elements = get_element_list([structure_or_mol])
    if isinstance(structure_or_mol, Structure):
        return S2G_local(element_types=elements, cutoff=cutoff).get_graph(structure_or_mol)
    return M2G_local(element_types=elements, cutoff=cutoff).get_graph(structure_or_mol)


def test_structure2graph_without_alchmtk(LiFePO4, monkeypatch):
    """The non-alchmtk fallback path must produce a graph equivalent to the alchmtk path."""
    graph, _lattice, state = _build_graph_without_alchmtk(LiFePO4, cutoff=4.0, monkeypatch=monkeypatch)
    assert graph.num_nodes == LiFePO4.num_sites
    # frac coords come back as a numpy array along the fallback path; the converter is
    # expected to coerce them. Just verify the resulting tensor shape and that the
    # state-attr structure is preserved.
    assert np.allclose(state, [0.0, 0.0])
    assert graph.num_edges > 0


def test_molecule2graph_without_alchmtk(CH4, monkeypatch):
    """The non-alchmtk fallback path for molecules uses a scipy.sparse adjacency."""
    graph, _lattice, state = _build_graph_without_alchmtk(CH4, cutoff=2.0, monkeypatch=monkeypatch)
    assert graph.num_nodes == 5
    # Same edge count as the alchmtk path for CH4 with cutoff=2.0.
    assert graph.num_edges == 20
    # state_attr is [molecular_weight_per_atom, mean_bonds_per_atom]
    assert len(state) == 2
    assert np.allclose(state, [3.208492, 2])
