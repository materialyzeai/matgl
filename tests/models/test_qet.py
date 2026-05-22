"""Tests for the QET model."""

from __future__ import annotations

import os

import pytest
import torch
from pymatgen.core import Lattice, Structure

from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph._compute import compute_pair_vector_and_distance
from matgl.models._qet import QET


def _make_qet(**overrides):
    """Construct QET, suppressing the warp kernel so the pure-PyTorch path runs."""
    overrides.setdefault("use_warp", False)
    return QET(**overrides)


def _single_atom_graph(cutoff=5.0):
    """Build a PyG graph for a single-atom structure, mirroring ``conftest.get_graph``."""
    structure = Structure(Lattice.cubic(3.17), ["Mo"], [[0.0, 0.0, 0.0]])
    element_types = get_element_list([structure])
    converter = Structure2Graph(element_types=element_types, cutoff=cutoff)
    graph, lattice, state = converter.get_graph(structure)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lattice[0])
    graph.pos = graph.frac_coords @ lattice[0]
    bond_vec, bond_dist = compute_pair_vector_and_distance(graph.pos, graph.edge_index, graph.pbc_offshift)
    graph.bond_vec = bond_vec
    graph.bond_dist = bond_dist
    return element_types, graph, state


def test_qet(graph_MoS):
    """Forward across activations + save/load + SO(3) variant."""
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    _, graph, _ = graph_MoS

    activations = ["swish", "tanh", "sigmoid", "softplus2", "softexp"]

    for act in activations:
        model = _make_qet(is_intensive=False, activation_type=act)
        output = model(g=graph, total_charge=torch.tensor([0.0]))
        assert torch.numel(output) == 1

    model.save(".")
    QET.load(".")
    for fname in ("model.pt", "model.json", "state.pt"):
        os.remove(fname)

    model = _make_qet(is_intensive=False, equivariance_invariance_group="SO(3)")
    output = model(g=graph, total_charge=torch.tensor([0.0]))
    assert torch.numel(output) == 1


def test_qet_return_features(graph_MoS):
    """`return_features=True` returns (node_feat, atomic_energies) with the right shapes."""
    torch.manual_seed(0)
    _, graph, _ = graph_MoS
    model = _make_qet(is_intensive=False, return_features=True)
    node_feat, atomic_energies = model(g=graph, total_charge=torch.tensor([0.0]))
    n_nodes = graph.pos.shape[0]
    # +1 charge, +1 elec_pot
    assert node_feat.shape == (n_nodes, model.units + 2)
    assert atomic_energies.shape[0] == n_nodes


def test_qet_include_magmom(graph_MoS):
    torch.manual_seed(0)
    _, graph, _ = graph_MoS
    model = _make_qet(is_intensive=False, include_magmom=True, return_features=True)
    node_feat, _ = model(g=graph, total_charge=torch.tensor([0.0]))
    n_nodes = graph.pos.shape[0]
    # +1 charge, +1 elec_pot, +1 magmom
    assert node_feat.shape == (n_nodes, model.units + 3)


def test_qet_is_hardness_envs(graph_MoS):
    torch.manual_seed(0)
    _, graph, _ = graph_MoS
    model = _make_qet(is_intensive=False, is_hardness_envs=True)
    output = model(g=graph, total_charge=torch.tensor([0.0]))
    assert torch.numel(output) == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"include_magmom": True},
        {"is_hardness_envs": True},
        {"is_sigma_train": True},
        {"equivariance_invariance_group": "SO(3)"},
    ],
)
def test_qet_single_atom(overrides):
    """Single-atom systems must not collapse the node dim via ``torch.squeeze``.

    Regression test: with one atom the per-node ``sigma`` tensor has shape
    ``(1,)``; a bare ``torch.squeeze`` collapsed it to a 0-d scalar, which then
    raised ``IndexError`` inside ``ElectrostaticPotential.forward``.
    """
    torch.manual_seed(0)
    element_types, graph, _ = _single_atom_graph()
    model = _make_qet(element_types=element_types, is_intensive=False, **overrides)
    output = model(g=graph, total_charge=torch.tensor([0.0]))
    assert torch.numel(output) == 1
    assert torch.isfinite(output).all()
