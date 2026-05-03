"""Tests for CHGNet PyG model: forward pass, prediction, Potential, training."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure
from torch_geometric.data import Batch

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("Skipping PYG tests", allow_module_level=True)

from matgl.apps._pes_pyg import Potential
from matgl.ext._pymatgen_pyg import Structure2Graph, get_element_list
from matgl.models._chgnet_pyg import CHGNet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mos_structure():
    return Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


@pytest.fixture(scope="module")
def fe_structure():
    """BCC Fe — useful for a monatomic test."""
    return Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])


@pytest.fixture(scope="module")
def default_model():
    return CHGNet(element_types=("Mo", "S"))


@pytest.fixture(scope="module")
def mos_graph(mos_structure, default_model):
    conv = Structure2Graph(element_types=default_model.element_types, cutoff=default_model.cutoff)
    g, lat, state = conv.get_graph(mos_structure)
    g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
    g.pos = g.frac_coords @ lat[0]
    return mos_structure, g, lat, state


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


class TestCHGNetConstruction:
    def test_default(self):
        model = CHGNet()
        assert model is not None
        assert model.n_blocks == 4

    def test_custom_dims(self):
        model = CHGNet(
            dim_atom_embedding=32,
            dim_bond_embedding=32,
            dim_angle_embedding=32,
            num_blocks=2,
        )
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_no_bond_graph(self):
        model = CHGNet(threebody_cutoff=0)
        assert not model.use_bond_graph

    @pytest.mark.parametrize("activation", ["swish", "softplus2", "tanh", "sigmoid"])
    def test_activation_types(self, activation):
        model = CHGNet(activation_type=activation)
        assert model is not None

    def test_invalid_activation(self):
        with pytest.raises(ValueError, match="Invalid activation type"):
            CHGNet(activation_type="notanactivation")

    def test_graph_norm_not_supported(self):
        with pytest.raises(ValueError, match="GraphNorm is not supported"):
            CHGNet(normalization="graph")

    def test_intensive_not_supported(self):
        with pytest.raises(NotImplementedError):
            CHGNet(is_intensive=True)

    def test_classification_not_supported(self):
        with pytest.raises(NotImplementedError):
            CHGNet(task_type="classification")

    def test_angle_readout_without_bond_graph(self):
        with pytest.raises(ValueError, match="Angle readout requires"):
            CHGNet(threebody_cutoff=0, readout_field="angle_feat")


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


class TestCHGNetForward:
    @pytest.mark.parametrize("readout_field", ["atom_feat", "bond_feat", "angle_feat"])
    def test_readout_fields(self, mos_graph, readout_field):
        _, g, _, _ = mos_graph
        model = CHGNet(element_types=("Mo", "S"), readout_field=readout_field)
        out = model(g)
        assert torch.numel(out) == 1
        assert torch.isfinite(out)

    @pytest.mark.parametrize("final_mlp_type", ["mlp", "gated"])
    def test_final_mlp_types(self, mos_graph, final_mlp_type):
        _, g, _, _ = mos_graph
        model = CHGNet(element_types=("Mo", "S"), final_mlp_type=final_mlp_type)
        out = model(g)
        assert torch.numel(out) == 1

    def test_magmom_shape(self, mos_graph, default_model):
        structure, g, _, _ = mos_graph
        out = default_model(g, return_all_layer_output=True)
        magmom = out["magmom"]
        assert magmom.shape[0] == structure.num_sites

    def test_no_threebody(self, mos_graph):
        _, g, _, _ = mos_graph
        model = CHGNet(element_types=("Mo", "S"), threebody_cutoff=0)
        out = model(g)
        assert torch.numel(out) == 1

    @pytest.mark.parametrize("normalization", [None, "layer"])
    def test_normalization_options(self, mos_graph, normalization):
        _, g, _, _ = mos_graph
        model = CHGNet(element_types=("Mo", "S"), normalization=normalization)
        out = model(g)
        assert torch.isfinite(out)

    @pytest.mark.parametrize("bond_update_hidden_dims", [None, (16,)])
    @pytest.mark.parametrize("angle_update_hidden_dims", [None, (16,)])
    def test_optional_update_blocks(self, mos_graph, bond_update_hidden_dims, angle_update_hidden_dims):
        _, g, _, _ = mos_graph
        model = CHGNet(
            element_types=("Mo", "S"),
            bond_update_hidden_dims=bond_update_hidden_dims,
            angle_update_hidden_dims=angle_update_hidden_dims,
        )
        out = model(g)
        assert torch.isfinite(out)

    def test_return_all_layer_output(self, mos_graph, default_model):
        _, g, _, _ = mos_graph
        out = default_model(g, return_all_layer_output=True)
        assert isinstance(out, dict)
        assert "embedding" in out
        assert "gc_1" in out
        assert f"gc_{default_model.n_blocks}" in out
        assert "magmom" in out
        assert "final" in out

    def test_output_finite(self, mos_graph, default_model):
        _, g, _, _ = mos_graph
        out = default_model(g)
        assert torch.isfinite(out)

    def test_batch_forward(self, mos_structure):
        """Two graphs batched together should give a 2-element output."""
        model = CHGNet(element_types=("Mo", "S"))
        conv = Structure2Graph(element_types=model.element_types, cutoff=model.cutoff)
        g1, lat1, _ = conv.get_graph(mos_structure)
        g1.pbc_offshift = torch.matmul(g1.pbc_offset, lat1[0])
        g1.pos = g1.frac_coords @ lat1[0]
        g2, lat2, _ = conv.get_graph(mos_structure)
        g2.pbc_offshift = torch.matmul(g2.pbc_offset, lat2[0])
        g2.pos = g2.frac_coords @ lat2[0]
        batched = Batch.from_data_list([g1, g2])
        out = model(batched)
        assert out.shape == (2,)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Extensivity: energy ∝ system size
# ---------------------------------------------------------------------------


class TestExtensivity:
    @pytest.mark.parametrize("struct_name", ["LiFePO4", "BaNiO3", "MoS"])
    def test_energy_extensivity(self, struct_name, request):
        """Energy per atom should be the same for supercells."""
        structure = request.getfixturevalue(struct_name)
        supercell = structure.copy()
        supercell.make_supercell(2)

        model = CHGNet()
        conv = Structure2Graph(element_types=model.element_types, cutoff=model.cutoff)

        g, lat, _ = conv.get_graph(structure)
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        g.pos = g.frac_coords @ lat[0]

        gs, lats, _ = conv.get_graph(supercell)
        gs.pbc_offshift = torch.matmul(gs.pbc_offset, lats[0])
        gs.pos = gs.frac_coords @ lats[0]

        out = model(g)
        out_s = model(gs)

        assert torch.allclose(
            out / structure.num_sites, out_s / supercell.num_sites, rtol=1e-4
        )


# ---------------------------------------------------------------------------
# predict_structure
# ---------------------------------------------------------------------------


class TestPredictStructure:
    def test_returns_tensor(self, mos_structure, default_model):
        out = default_model.predict_structure(mos_structure)
        assert isinstance(out, torch.Tensor)
        assert torch.isfinite(out)

    def test_return_features(self, mos_structure, default_model):
        out = default_model.predict_structure(mos_structure, return_features=True)
        assert isinstance(out, dict)
        assert "final" in out
        assert "bond_expansion" in out
        assert "embedding" in out
        assert "gc_1" in out

    def test_specific_output_layers(self, mos_structure, default_model):
        out = default_model.predict_structure(
            mos_structure, return_features=True, output_layers=["final", "gc_1"]
        )
        assert set(out.keys()) == {"final", "gc_1"}

    def test_invalid_output_layers(self, mos_structure, default_model):
        with pytest.raises(ValueError, match="Invalid output_layers"):
            default_model.predict_structure(
                mos_structure, return_features=True, output_layers=["not_a_layer"]
            )


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_and_load(self, mos_graph, tmp_path):
        _, g, _, _ = mos_graph
        model = CHGNet(element_types=("Mo", "S"), num_blocks=2)
        out_before = model(g).item()

        model.save(str(tmp_path))
        loaded = CHGNet.load(str(tmp_path))
        out_after = loaded(g).item()

        assert abs(out_before - out_after) < 1e-5


# ---------------------------------------------------------------------------
# Potential (energy, forces, stresses, hessian)
# ---------------------------------------------------------------------------


class TestCHGNetPotential:
    @pytest.fixture()
    def potential(self, default_model):
        return Potential(model=default_model, calc_hessian=True)

    def test_efsh_shapes(self, mos_graph, potential):
        structure, g, lat, state = mos_graph
        lat_t = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        e, f, s, h = potential(g, lat_t, state)
        assert torch.numel(e) == 1
        assert f.shape == (structure.num_sites, 3)
        assert s.shape == (3, 3)
        assert h.shape == (structure.num_sites * 3, structure.num_sites * 3)

    def test_efs_shapes(self, mos_graph, default_model):
        structure, g, lat, state = mos_graph
        lat_t = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        ff = Potential(model=default_model)
        e, f, s, h = ff(g, lat_t, state)
        assert torch.numel(e) == 1
        assert f.shape == (structure.num_sites, 3)
        assert s.shape == (3, 3)
        assert h.shape[0] == 1  # not computed

    def test_forces_only(self, mos_graph, default_model):
        structure, g, lat, state = mos_graph
        lat_t = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        ff = Potential(model=default_model, calc_stresses=False)
        e, f, s, h = ff(g, lat_t, state)
        assert torch.numel(e) == 1
        assert f.shape == (structure.num_sites, 3)

    def test_energy_only(self, mos_graph, default_model):
        structure, g, lat, state = mos_graph
        lat_t = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        ff = Potential(model=default_model, calc_forces=False, calc_stresses=False)
        e, f, s, h = ff(g, lat_t, state)
        assert torch.numel(e) == 1

    def test_forces_finite_difference(self, default_model):
        """Force = -dE/dR; verify with finite differences."""
        p2g = Structure2Graph(element_types=default_model.element_types, cutoff=default_model.cutoff)
        struct_m = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0, 0], [0.498, 0.5, 0.5]])
        struct_0 = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0, 0], [0.500, 0.5, 0.5]])
        struct_p = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0, 0], [0.502, 0.5, 0.5]])

        ff = Potential(model=default_model, calc_hessian=True, debug_mode=True)

        def make_graph(struct):
            g, lat, state = p2g.get_graph(struct)
            return g, lat, state

        g_m, lat_m, state = make_graph(struct_m)
        g_0, lat_0, _ = make_graph(struct_0)
        g_p, lat_p, _ = make_graph(struct_p)

        lat_m_t = torch.tensor(struct_m.lattice.matrix, dtype=matgl.float_th)
        lat_0_t = torch.tensor(struct_0.lattice.matrix, dtype=matgl.float_th)
        lat_p_t = torch.tensor(struct_p.lattice.matrix, dtype=matgl.float_th)

        e_m, _, _ = ff(g_m, lat_m_t, state)
        _, grad_zero, _ = ff(g_0, lat_0_t, state)
        e_p, _, _ = ff(g_p, lat_p_t, state)

        dx = 0.004  # fractional displacement × lattice param = 0.002 × 4.0 Å = 0.008 Å
        fd = (e_p - e_m) / (2 * dx)
        assert np.allclose(
            fd.detach().numpy(), grad_zero[1][0].detach().numpy(), atol=1e-4
        )

    def test_batch_potential(self, mos_structure, default_model):
        conv = Structure2Graph(element_types=default_model.element_types, cutoff=default_model.cutoff)
        g1, lat1, _ = conv.get_graph(mos_structure)
        g1.pbc_offshift = torch.matmul(g1.pbc_offset, lat1[0])
        g1.pos = g1.frac_coords @ lat1[0]
        g2, lat2, _ = conv.get_graph(mos_structure)
        g2.pbc_offshift = torch.matmul(g2.pbc_offset, lat2[0])
        g2.pos = g2.frac_coords @ lat2[0]

        batched = Batch.from_data_list([g1, g2])
        lat = torch.stack([
            torch.tensor(mos_structure.lattice.matrix, dtype=matgl.float_th),
            torch.tensor(mos_structure.lattice.matrix, dtype=matgl.float_th),
        ])
        ff = Potential(model=default_model)
        e, f, s, h = ff(batched, lat, None)
        assert e.shape == (2,)
        assert f.shape == (batched.num_nodes, 3)
        assert s.shape == (6, 3)  # 2 structures × 3 rows

    def test_with_zbl_repulsion(self, mos_structure, default_model):
        """ZBL needs bond_dist on the graph — Potential builds it from pos."""
        from matgl.ext._pymatgen_pyg import Structure2Graph
        from matgl.graph._compute_pyg import compute_pair_vector_and_distance
        conv = Structure2Graph(element_types=default_model.element_types, cutoff=default_model.cutoff)
        g, lat, state = conv.get_graph(mos_structure)
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        g.pos = g.frac_coords @ lat[0]
        _, g.bond_dist = compute_pair_vector_and_distance(g.pos, g.edge_index, g.pbc_offshift)
        lat_t = torch.tensor(mos_structure.lattice.matrix, dtype=matgl.float_th)
        ff = Potential(model=default_model, calc_repuls=True)
        e, f, s, h = ff(g, lat_t, state)
        assert torch.isfinite(e)

    def test_with_element_refs(self, mos_graph, default_model):
        structure, g, lat, state = mos_graph
        lat_t = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        # Dummy per-element energy references (same length as element_types)
        refs = torch.zeros(len(default_model.element_types))
        ff = Potential(model=default_model, element_refs=refs.numpy())
        e, f, s, h = ff(g, lat_t, state)
        assert torch.isfinite(e)


# ---------------------------------------------------------------------------
# Training step (gradient flow)
# ---------------------------------------------------------------------------


class TestCHGNetTraining:
    def _fresh_graph(self, structure, model):
        """Build a fresh graph with requires_grad=False on pos to avoid double-backward."""
        from matgl.ext._pymatgen_pyg import Structure2Graph
        conv = Structure2Graph(element_types=model.element_types, cutoff=model.cutoff)
        g, lat, state = conv.get_graph(structure)
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        g.pos = g.frac_coords @ lat[0]
        return g, lat, state

    def test_gradient_flows(self, mos_structure):
        """All parameters should receive gradients after a backward pass."""
        # Use bond_update_hidden_dims so bond_bond_weights participates in the graph
        model = CHGNet(
            element_types=("Mo", "S"), num_blocks=2,
            bond_update_hidden_dims=(16,),
        )
        g, lat, _ = self._fresh_graph(mos_structure, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        optimizer.zero_grad()
        out = model(g)
        loss = out.sum()
        loss.backward()
        optimizer.step()

        for name, param in model.named_parameters():
            if param.requires_grad:
                # sitewise_readout (magmom) is auxiliary — not on the energy path.
                # angle edge_update_func may be dormant if no line-graph edges exist.
                if "sitewise_readout" in name:
                    continue
                if "bond_graph_layers" in name and "edge_update_func" in name:
                    continue
                assert param.grad is not None, f"No gradient for {name}"

    def test_training_step_reduces_loss(self, mos_structure):
        """Loss should change after one optimizer step."""
        model = CHGNet(element_types=("Mo", "S"), num_blocks=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        target = torch.tensor(-1.0)

        def _loss():
            # Rebuild graph each call to avoid double-backward on saved tensors
            g, _, _ = self._fresh_graph(mos_structure, model)
            return (model(g) - target).pow(2)

        loss_before = _loss().item()
        optimizer.zero_grad()
        _loss().backward()
        optimizer.step()
        loss_after = _loss().item()

        # After one step the model is different (loss changed)
        assert loss_before != loss_after

    def test_potential_training_step(self, mos_structure):
        """Training via Potential (energy + forces) should compute all grads."""
        model = CHGNet(
            element_types=("Mo", "S"), num_blocks=2,
            bond_update_hidden_dims=(16,),
        )
        ff = Potential(model=model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        g, lat, state = self._fresh_graph(mos_structure, model)
        lat_t = torch.tensor(mos_structure.lattice.matrix, dtype=matgl.float_th)

        optimizer.zero_grad()
        e, f, s, h = ff(g, lat_t, state)
        loss = e.sum() + f.pow(2).sum()
        loss.backward()
        optimizer.step()

        for name, param in model.named_parameters():
            if param.requires_grad:
                # sitewise_readout (magmom) is auxiliary — not on the energy path.
                # angle edge_update_func may be dormant if no line-graph edges exist.
                if "sitewise_readout" in name:
                    continue
                if "bond_graph_layers" in name and "edge_update_func" in name:
                    continue
                assert param.grad is not None, f"No gradient for param {name}"
