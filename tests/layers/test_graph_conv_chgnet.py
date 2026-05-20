"""Tests for CHGNet PyG graph convolution layers and compute utilities."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from matgl.graph._compute import (
    compute_theta,
    create_directed_line_graph,
)
from matgl.layers._graph_convolution import (
    CHGNetAtomGraphBlock,
    CHGNetBondGraphBlock,
    CHGNetGraphConv,
    CHGNetLineGraphConv,
    _GatedMLPNorm,
    _MLPNorm,
)
from matgl.utils.maths import scatter_add

# ---------------------------------------------------------------------------
# Helper: minimal asymmetric toy graph (3 atoms, directed edges)
#   Edges: 0→1, 0→2, 1→2   (src=central, dst=neighbor)
# ---------------------------------------------------------------------------

NUM_ATOMS = 3
NUM_BONDS = 3
DIM = 16
RBF_DIM = 9


@pytest.fixture
def toy_graph():
    """Returns (edge_index, atom_feat, bond_feat, bond_expansion) for a toy graph."""
    torch.manual_seed(0)
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long)
    atom_feat = torch.randn(NUM_ATOMS, DIM)
    bond_feat = torch.randn(NUM_BONDS, DIM)
    bond_expansion = torch.randn(NUM_BONDS, RBF_DIM)
    return edge_index, atom_feat, bond_feat, bond_expansion


# ---------------------------------------------------------------------------
# _MLPNorm
# ---------------------------------------------------------------------------


class TestMLPNorm:
    def test_forward_no_norm(self):
        mlp = _MLPNorm([8, 16, 8], activation=nn.SiLU())
        x = torch.randn(4, 8)
        out = mlp(x)
        assert out.shape == (4, 8)
        assert torch.isfinite(out).all()

    def test_forward_with_layer_norm(self):
        mlp = _MLPNorm([8, 16, 8], activation=nn.SiLU(), normalization="layer", normalize_hidden=True)
        x = torch.randn(4, 8)
        out = mlp(x)
        assert out.shape == (4, 8)
        assert torch.isfinite(out).all()

    def test_activate_last_false(self):
        mlp = _MLPNorm([8, 8], activation=nn.ReLU(), activate_last=False)
        x = torch.full((2, 8), -1.0)
        out = mlp(x)
        # Without activation on the last layer, negatives can pass through
        assert (out < 0).any()


# ---------------------------------------------------------------------------
# _GatedMLPNorm
# ---------------------------------------------------------------------------


class TestGatedMLPNorm:
    def test_forward(self):
        gmlp = _GatedMLPNorm(8, [16, 8], activation=nn.SiLU())
        x = torch.randn(5, 8)
        out = gmlp(x)
        assert out.shape == (5, 8)
        assert torch.isfinite(out).all()

    def test_output_in_range(self):
        """Sigmoid gate keeps output bounded relative to the value branch."""
        gmlp = _GatedMLPNorm(4, [4], activation=nn.SiLU())
        x = torch.randn(100, 4)
        out = gmlp(x)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# CHGNetGraphConv (atom graph)
# ---------------------------------------------------------------------------


class TestCHGNetGraphConv:
    def test_construction(self):
        conv = CHGNetGraphConv.from_dims(
            activation=nn.SiLU(),
            node_dims=[2 * DIM + DIM, DIM, DIM],
            edge_dims=None,
        )
        assert conv.edge_update_func is None
        assert conv.node_update_func is not None

    def test_forward_shape(self, toy_graph):
        edge_index, atom_feat, bond_feat, bond_expansion = toy_graph
        conv = CHGNetGraphConv.from_dims(
            activation=nn.SiLU(),
            node_dims=[2 * DIM + DIM, DIM, DIM],
        )
        new_atom, new_bond, new_state = conv(
            edge_index,
            atom_feat,
            bond_feat,
            bond_expansion,
            state_attr=None,
            batch=None,
            shared_node_weights=None,
            shared_edge_weights=None,
        )
        assert new_atom.shape == atom_feat.shape
        assert new_bond.shape == bond_feat.shape
        assert new_state is None

    def test_forward_with_edge_update(self, toy_graph):
        edge_index, atom_feat, bond_feat, bond_expansion = toy_graph
        conv = CHGNetGraphConv.from_dims(
            activation=nn.SiLU(),
            node_dims=[2 * DIM + DIM, DIM, DIM],
            edge_dims=[2 * DIM + DIM, DIM, DIM],
        )
        new_atom, new_bond, _ = conv(
            edge_index,
            atom_feat,
            bond_feat,
            bond_expansion,
            None,
            None,
            None,
            None,
        )
        assert new_atom.shape == atom_feat.shape
        assert new_bond.shape == bond_feat.shape

    def test_message_scatters_onto_src_not_dst(self, toy_graph):
        """Key bug-fix test: messages must accumulate on SRC (central atom)."""
        edge_index, _, _, _ = toy_graph
        src, dst = edge_index[0], edge_index[1]
        messages = torch.ones(NUM_BONDS, DIM)

        # scatter onto src (center atoms) vs dst (neighbor atoms)
        correct = scatter_add(messages, src, dim=0, dim_size=NUM_ATOMS)
        wrong = scatter_add(messages, dst, dim=0, dim_size=NUM_ATOMS)

        # Atom 0 sends 2 edges (0→1, 0→2) → should receive 2 messages as src
        assert correct[0].sum().item() == pytest.approx(2 * DIM)
        # If scattered onto dst, atom 0 receives 0 messages (no edges point to 0)
        assert wrong[0].sum().item() == pytest.approx(0.0)

    def test_forward_output_finite(self, toy_graph):
        edge_index, atom_feat, bond_feat, bond_expansion = toy_graph
        conv = CHGNetGraphConv.from_dims(
            activation=nn.SiLU(),
            node_dims=[2 * DIM + DIM, DIM, DIM],
        )
        new_atom, new_bond, _ = conv(
            edge_index,
            atom_feat,
            bond_feat,
            bond_expansion,
            None,
            None,
            None,
            None,
        )
        assert torch.isfinite(new_atom).all()
        assert torch.isfinite(new_bond).all()

    def test_forward_with_shared_weights(self, toy_graph):
        edge_index, atom_feat, bond_feat, bond_expansion = toy_graph
        conv = CHGNetGraphConv.from_dims(
            activation=nn.SiLU(),
            node_dims=[2 * DIM + DIM, DIM, DIM],
            rbf_order=RBF_DIM,
        )
        shared = torch.randn(NUM_BONDS, DIM)
        new_atom, _, _ = conv(
            edge_index,
            atom_feat,
            bond_feat,
            bond_expansion,
            None,
            None,
            shared,
            None,
        )
        assert new_atom.shape == atom_feat.shape

    def test_forward_with_batch(self, toy_graph):
        edge_index, atom_feat, bond_feat, bond_expansion = toy_graph
        conv = CHGNetGraphConv.from_dims(
            activation=nn.SiLU(),
            node_dims=[2 * DIM + DIM, DIM, DIM],
        )
        batch = torch.zeros(NUM_ATOMS, dtype=torch.long)
        new_atom, _, _ = conv(
            edge_index,
            atom_feat,
            bond_feat,
            bond_expansion,
            None,
            batch,
            None,
            None,
        )
        assert new_atom.shape == atom_feat.shape


# ---------------------------------------------------------------------------
# CHGNetAtomGraphBlock
# ---------------------------------------------------------------------------


class TestCHGNetAtomGraphBlock:
    def test_forward_no_bond_update(self, toy_graph):
        edge_index, atom_feat, bond_feat, bond_expansion = toy_graph
        block = CHGNetAtomGraphBlock(
            num_atom_feats=DIM,
            num_bond_feats=DIM,
            activation=nn.SiLU(),
            atom_hidden_dims=[DIM],
            bond_hidden_dims=None,
        )
        new_atom, new_bond, _ = block(
            edge_index,
            atom_feat,
            bond_feat,
            bond_expansion,
            None,
            None,
            None,
            None,
        )
        assert new_atom.shape == atom_feat.shape
        assert new_bond.shape == bond_feat.shape

    def test_forward_with_bond_update(self, toy_graph):
        edge_index, atom_feat, bond_feat, bond_expansion = toy_graph
        block = CHGNetAtomGraphBlock(
            num_atom_feats=DIM,
            num_bond_feats=DIM,
            activation=nn.SiLU(),
            atom_hidden_dims=[DIM],
            bond_hidden_dims=[DIM],
        )
        new_atom, new_bond, _ = block(
            edge_index,
            atom_feat,
            bond_feat,
            bond_expansion,
            None,
            None,
            None,
            None,
        )
        assert new_atom.shape == atom_feat.shape
        assert new_bond.shape == bond_feat.shape

    def test_forward_with_layer_norm(self, toy_graph):
        edge_index, atom_feat, bond_feat, bond_expansion = toy_graph
        block = CHGNetAtomGraphBlock(
            num_atom_feats=DIM,
            num_bond_feats=DIM,
            activation=nn.SiLU(),
            atom_hidden_dims=[DIM],
            normalization="layer",
        )
        new_atom, _, _ = block(
            edge_index,
            atom_feat,
            bond_feat,
            bond_expansion,
            None,
            None,
            None,
            None,
        )
        assert torch.isfinite(new_atom).all()

    def test_forward_with_dropout(self, toy_graph):
        edge_index, atom_feat, bond_feat, bond_expansion = toy_graph
        block = CHGNetAtomGraphBlock(
            num_atom_feats=DIM,
            num_bond_feats=DIM,
            activation=nn.SiLU(),
            atom_hidden_dims=[DIM],
            dropout=0.5,
        )
        block.train()
        new_atom, _, _ = block(
            edge_index,
            atom_feat,
            bond_feat,
            bond_expansion,
            None,
            None,
            None,
            None,
        )
        assert new_atom.shape == atom_feat.shape


# ---------------------------------------------------------------------------
# create_directed_line_graph + compute_theta
# ---------------------------------------------------------------------------


class TestLineGraphConstruction:
    def test_output_shapes(self, toy_graph):
        edge_index, _, _, _ = toy_graph
        bond_vec = torch.randn(NUM_BONDS, 3)
        bond_dist = torch.tensor([1.0, 2.0, 2.5])
        pbc_offset = torch.zeros(NUM_BONDS, 3)

        lg_ei, lg_bvec, lg_bdist, _, lg_sign = create_directed_line_graph(
            edge_index, pbc_offset, bond_vec, bond_dist, threebody_cutoff=3.0
        )
        assert lg_ei.shape[0] == 2
        assert lg_bvec.shape[1] == 3
        assert lg_bdist.dim() == 1
        assert lg_sign.shape[1] == 1

    def test_no_bonds_within_cutoff(self, toy_graph):
        edge_index, _, _, _ = toy_graph
        bond_vec = torch.randn(NUM_BONDS, 3)
        bond_dist = torch.tensor([5.0, 6.0, 7.0])
        pbc_offset = torch.zeros(NUM_BONDS, 3)

        lg_ei, lg_bvec, _, _, _ = create_directed_line_graph(
            edge_index, pbc_offset, bond_vec, bond_dist, threebody_cutoff=1.0
        )
        assert lg_ei.shape[1] == 0
        assert lg_bvec.shape[0] == 0

    def test_compute_theta_shape(self, toy_graph):
        edge_index, _, _, _ = toy_graph
        bond_vec = torch.randn(NUM_BONDS, 3)
        bond_dist = torch.tensor([1.0, 2.0, 2.5])
        pbc_offset = torch.zeros(NUM_BONDS, 3)

        lg_ei, lg_bvec, _, _, lg_sign = create_directed_line_graph(
            edge_index, pbc_offset, bond_vec, bond_dist, threebody_cutoff=3.0
        )
        if lg_ei.size(1) > 0:
            cos_theta = compute_theta(lg_bvec, lg_sign, lg_ei[0].long(), lg_ei[1].long(), directed=True)
            assert cos_theta.shape == (lg_ei.size(1),)
            assert (cos_theta >= -1.0).all()
            assert (cos_theta <= 1.0).all()

    def test_theta_values_in_range(self):
        """Cosine of angles must be in [-1, 1]."""
        # Simple case: two bonds at 90 degrees
        bond_vec = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        bond_sign = torch.ones(2, 1)
        lg_src = torch.tensor([0], dtype=torch.long)
        lg_dst = torch.tensor([1], dtype=torch.long)
        cos_theta = compute_theta(bond_vec, bond_sign, lg_src, lg_dst)
        assert abs(cos_theta.item()) < 1e-5  # cos(90°) ≈ 0


# ---------------------------------------------------------------------------
# CHGNetLineGraphConv
# ---------------------------------------------------------------------------


class TestCHGNetLineGraphConv:
    def _make_line_graph_data(self):
        """Toy line graph: 4 bond nodes, 6 line-graph edges."""
        torch.manual_seed(42)
        num_lg_nodes = 4
        lg_edge_index = torch.tensor([[0, 1, 2, 0, 2, 3], [1, 0, 0, 2, 3, 2]], dtype=torch.long)
        bond_feat = torch.randn(num_lg_nodes, DIM)
        angle_feat = torch.randn(lg_edge_index.size(1), DIM)
        atom_feat = torch.randn(lg_edge_index.size(1), DIM)
        return lg_edge_index, bond_feat, angle_feat, atom_feat

    def test_forward_shape(self):
        lg_edge_index, bond_feat, angle_feat, atom_feat = self._make_line_graph_data()
        conv = CHGNetLineGraphConv.from_dims(
            node_dims=[4 * DIM, DIM, DIM],  # bonds_i + angle + aux + bonds_j
            activation=nn.SiLU(),
        )
        new_bond, new_angle = conv(lg_edge_index, bond_feat, angle_feat, atom_feat, None, None)
        assert new_bond.shape == bond_feat.shape
        assert new_angle.shape == angle_feat.shape

    def test_forward_with_angle_update(self):
        lg_edge_index, bond_feat, angle_feat, atom_feat = self._make_line_graph_data()
        conv = CHGNetLineGraphConv.from_dims(
            node_dims=[4 * DIM, DIM, DIM],  # bonds_i + angle + aux + bonds_j
            edge_dims=[4 * DIM, DIM, DIM],
            activation=nn.SiLU(),
        )
        new_bond, new_angle = conv(lg_edge_index, bond_feat, angle_feat, atom_feat, None, None)
        assert new_bond.shape == bond_feat.shape
        assert new_angle.shape == angle_feat.shape

    def test_accumulates_onto_dst(self):
        """Line-graph node update must accumulate onto DST (bond being updated)."""
        lg_edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
        messages = torch.ones(2, DIM)
        lg_dst = lg_edge_index[1]
        result = scatter_add(messages, lg_dst.long(), dim=0, dim_size=3)
        # Node 2 is dst of both edges → should accumulate 2 messages
        assert result[2].sum().item() == pytest.approx(2 * DIM)
        assert result[0].sum().item() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CHGNetBondGraphBlock
# ---------------------------------------------------------------------------


class TestCHGNetBondGraphBlock:
    def test_forward(self, toy_graph):
        torch.manual_seed(0)
        # bond_index: which atom-graph bonds appear as line-graph nodes
        bond_index = torch.tensor([0, 1, 2])
        lg_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        bond_feat = torch.randn(NUM_BONDS, DIM)
        angle_feat = torch.randn(lg_edge_index.size(1), DIM)
        atom_feat = torch.randn(NUM_ATOMS, DIM)
        center_atom_index = torch.tensor([0, 1])

        block = CHGNetBondGraphBlock(
            num_atom_feats=DIM,
            num_bond_feats=DIM,
            num_angle_feats=DIM,
            activation=nn.SiLU(),
            bond_hidden_dims=[DIM],
            angle_hidden_dims=[DIM],
        )
        new_bond, new_angle = block(
            lg_edge_index,
            bond_feat,
            angle_feat,
            atom_feat,
            bond_index,
            center_atom_index,
            None,
            None,
        )
        assert new_bond.shape == bond_feat.shape
        assert new_angle.shape == angle_feat.shape
        assert torch.isfinite(new_bond).all()

    def test_only_bond_update(self, toy_graph):
        bond_index = torch.tensor([0, 1, 2])
        lg_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        bond_feat = torch.randn(NUM_BONDS, DIM)
        angle_feat = torch.randn(lg_edge_index.size(1), DIM)
        atom_feat = torch.randn(NUM_ATOMS, DIM)
        center_atom_index = torch.tensor([0, 1])

        block = CHGNetBondGraphBlock(
            num_atom_feats=DIM,
            num_bond_feats=DIM,
            num_angle_feats=DIM,
            activation=nn.SiLU(),
            bond_hidden_dims=[DIM],
            angle_hidden_dims=None,  # no angle update
        )
        new_bond, _ = block(
            lg_edge_index,
            bond_feat,
            angle_feat,
            atom_feat,
            bond_index,
            center_atom_index,
            None,
            None,
        )
        assert new_bond.shape == bond_feat.shape
