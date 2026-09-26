from __future__ import annotations

import itertools
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from pymatgen.core import Structure
from torch_geometric.data import Data

import matgl
from matgl.ext.pymatgen import Structure2Graph, get_element_list
from matgl.graph._compute import compute_pair_vector_and_distance, create_line_graph
from matgl.models import M3GNet
from matgl.utils.cutoff import polynomial_cutoff

PARITY_ARTIFACT = Path(__file__).resolve().parents[1] / "parity_data" / "m3gnet_parity.pt"


def _prep_graph(graph, structure):
    """Attach pos / pbc_offshift / bond_{vec,dist} to ``graph``."""
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=graph.pos.device)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
    graph.pos = graph.frac_coords @ lat[0]
    bond_vec, bond_dist = compute_pair_vector_and_distance(graph.pos, graph.edge_index, graph.pbc_offshift)
    graph.bond_vec = bond_vec
    graph.bond_dist = bond_dist
    return graph


def test_model(graph_MoS):
    structure, graph, _ = graph_MoS
    graph = _prep_graph(graph, structure)
    for act in ["swish", "tanh", "sigmoid", "softplus2", "softexp"]:
        model = M3GNet(is_intensive=False, activation_type=act)
        output = model(g=graph)
        assert torch.numel(output) == 1
    model.save(".")
    M3GNet.load(".")
    os.remove("model.pt")
    os.remove("model.json")
    os.remove("state.pt")


def test_exceptions():
    with pytest.raises(ValueError, match="Invalid activation type"):
        _ = M3GNet(element_types=None, is_intensive=False, activation_type="whatever")
    with pytest.raises(ValueError, match=r"Classification task cannot be extensive."):
        _ = M3GNet(element_types=["Mo", "S"], is_intensive=False, task_type="classification")


@pytest.mark.parametrize(
    ("pos", "edge_index"),
    [
        (torch.tensor([[0.0, 0.0, 0.0]]), torch.empty((2, 0), dtype=torch.long)),
        (torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), torch.tensor([[0, 1], [1, 0]])),
    ],
    ids=["isolated-atom", "dimer"],
)
def test_m3gnet_handles_no_triplet_extremes(pos, edge_index):
    """Full M3GNet forward remains finite and cache-equivalent when no angles exist."""
    torch.manual_seed(3)
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
    node_type = torch.zeros(pos.size(0), dtype=torch.long)
    graph = Data(pos=pos, edge_index=edge_index, node_type=node_type, num_nodes=pos.size(0))
    bond_vec, bond_dist = compute_pair_vector_and_distance(pos, edge_index, None)
    cached_line_graph = create_line_graph(edge_index, bond_dist, bond_vec, None, pos.size(0), 2.0)

    with torch.no_grad():
        dynamic = model(g=graph)
        cached = model(g=graph, l_g=cached_line_graph)

    assert torch.isfinite(dynamic).all()
    torch.testing.assert_close(cached, dynamic)


def test_model_intensive(graph_MoS):
    structure, graph, _ = graph_MoS
    graph = _prep_graph(graph, structure)
    model = M3GNet(element_types=["Mo", "S"], is_intensive=True)
    output = model(g=graph)
    assert torch.numel(output) == 1


def test_model_intensive_reduce_atom(graph_MoS):
    structure, graph, _ = graph_MoS
    graph = _prep_graph(graph, structure)
    model = M3GNet(element_types=["Mo", "S"], is_intensive=True, readout_type="reduce_atom")
    output = model(g=graph)
    assert torch.numel(output) == 1


def test_model_intensive_with_classification(graph_MoS):
    structure, graph, _ = graph_MoS
    graph = _prep_graph(graph, structure)
    model = M3GNet(element_types=["Mo", "S"], is_intensive=True, task_type="classification")
    output = model(g=graph)
    assert torch.numel(output) == 1


def test_model_intensive_set2set_classification(graph_MoS):
    structure, graph, _ = graph_MoS
    graph = _prep_graph(graph, structure)
    model = M3GNet(
        element_types=["Mo", "S"],
        is_intensive=True,
        task_type="classification",
        readout_type="set2set",
        niters_set2set=2,
        nlayers_set2set=1,
    )
    output = model(g=graph)
    assert torch.numel(output) == 1


def test_predict_structure(graph_MoS):
    structure, _, _ = graph_MoS
    model = M3GNet(element_types=["Mo", "S"], is_intensive=False)
    output_final = model.predict_structure(structure)
    assert torch.numel(output_final) == 1


def test_save_load(tmp_path):
    model = M3GNet(element_types=("Mo", "S"), is_intensive=True)
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        model.save(".")
        M3GNet.load(".")
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def parity_artifact():
    """Load the M3GNet parity artifact."""
    if not PARITY_ARTIFACT.exists():
        pytest.skip(f"Parity artifact missing: {PARITY_ARTIFACT}. Generate via tests/parity_data/gen_m3gnet_parity.py.")
    return torch.load(PARITY_ARTIFACT, map_location="cpu", weights_only=False)


def _build_parity_graph(structure, init_args):
    """Build a graph + position tensors mirroring the artifact generator."""
    conv = Structure2Graph(element_types=init_args["element_types"], cutoff=init_args["cutoff"])
    g, lat, _ = conv.get_graph(structure)
    g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
    g.pos = g.frac_coords @ lat[0]
    return g


def test_m3gnet_parity(parity_artifact):
    """A state-dict from the parity artifact must reproduce its golden output."""
    init_args = parity_artifact["init_args"]
    state_dict = parity_artifact["state_dict"]
    expected = parity_artifact["expected_output"]
    structure_kw = parity_artifact["structure_kw"]
    structure = Structure(**structure_kw)

    model = M3GNet(**init_args)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    assert not unexpected, f"Unexpected keys when loading parity state_dict: {unexpected}"
    bad_missing = [k for k in missing if "bond_expansion" not in k]
    assert not bad_missing, f"Missing keys when loading parity state_dict: {bad_missing}"

    model.eval()
    g = _build_parity_graph(structure, init_args)
    with torch.no_grad():
        output = model(g=g)

    assert torch.allclose(output, expected, atol=1e-5, rtol=1e-5), (
        f"M3GNet parity broken: got {output.item()}, expected {expected.item()}"
    )


# ---------------------------------------------------------------------------
# Three-body correctness when threebody_cutoff < cutoff (the pretrained PES setting).
# ---------------------------------------------------------------------------
def _small_m3gnet(element_types, seed=0):
    torch.manual_seed(seed)
    model = M3GNet(
        element_types=element_types,
        dim_node_embedding=16,
        dim_edge_embedding=16,
        units=16,
        max_n=3,
        max_l=3,
        nblocks=2,
        cutoff=5.0,
        threebody_cutoff=4.0,
        is_intensive=False,
    )
    return model.eval()


def _structure_graph(structure, element_types, cutoff=5.0):
    g, lat, _ = Structure2Graph(element_types=element_types, cutoff=cutoff).get_graph(structure)
    g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
    g.pos = g.frac_coords @ lat[0]
    return g


def _bruteforce_three_body(model, g, node_feat, edge_feat):
    """Reference three-body bond update from an explicit loop over triplets (j, i, k)."""
    layer = model.three_body_interactions[0]
    edge_index = g.edge_index
    bond_vec, bond_dist = compute_pair_vector_and_distance(g.pos, edge_index, g.pbc_offshift)
    fc = polynomial_cutoff(bond_dist, model.threebody_cutoff)
    by_center: dict[int, list[int]] = {}
    for b in range(edge_index.size(1)):
        if bond_dist[b] <= model.threebody_cutoff:
            by_center.setdefault(int(edge_index[0, b]), []).append(b)
    ij, ik = torch.tensor([p for bonds in by_center.values() for p in itertools.permutations(bonds, 2)]).T
    cos = ((bond_vec[ij] * bond_vec[ik]).sum(1) / (bond_dist[ij] * bond_dist[ik])).clamp(-1 + 1e-7, 1 - 1e-7)
    basis = model.basis_expansion(bond_dist[ik], cos, torch.zeros_like(cos))
    msg = basis * layer.update_network_atom(node_feat)[edge_index[1][ik]] * (fc[ij] * fc[ik])[:, None]
    new_bonds = torch.zeros(edge_index.size(1), msg.size(1)).index_add_(0, ij, msg)
    return edge_feat + layer.update_network_bond(new_bonds)


def test_three_body_update_matches_bruteforce(LiFePO4):
    """Each triplet message must reach bond i->j and use atom k and f_c(r_ij) f_c(r_ik) of the right bonds."""
    element_types = get_element_list([LiFePO4])
    model = _small_m3gnet(element_types)
    g = _structure_graph(LiFePO4, element_types)
    captured = {}
    handle = model.three_body_interactions[0].register_forward_hook(
        lambda _m, args, out: captured.update(args=args, out=out)
    )
    with torch.no_grad():
        model(g=g)
        handle.remove()
        node_feat, edge_feat = captured["args"][6], captured["args"][7]
        expected = _bruteforce_three_body(model, g, node_feat, edge_feat)
    assert (expected - edge_feat).abs().max() > 1e-3, "three-body update is trivially zero"
    torch.testing.assert_close(captured["out"], expected, atol=1e-5, rtol=1e-5)


def _per_atom_three_body_update(model, structure, element_types):
    """Squared norm of the block-1 three-body bond update, summed over the bonds of each atom."""
    g = _structure_graph(structure, element_types)
    captured = {}
    handle = model.three_body_interactions[0].register_forward_hook(
        lambda _m, args, out: captured.update(args=args, out=out)
    )
    with torch.no_grad():
        model(g=g)
    handle.remove()
    delta = (captured["out"] - captured["args"][7]).pow(2).sum(1)
    return torch.zeros(g.num_nodes).index_add_(0, g.edge_index[0].long(), delta)


def test_three_body_update_permutation_equivariant(LiFePO4):
    """Reordering atoms must permute, not change, the three-body update when bonds are pruned."""
    element_types = get_element_list([LiFePO4])
    model = _small_m3gnet(element_types)
    perm = np.random.default_rng(0).permutation(len(LiFePO4))
    permuted = Structure.from_sites([LiFePO4[int(i)] for i in perm])
    ref = _per_atom_three_body_update(model, LiFePO4, element_types)
    out = _per_atom_three_body_update(model, permuted, element_types)
    torch.testing.assert_close(out, ref[torch.as_tensor(perm)], atol=1e-8, rtol=1e-4)
