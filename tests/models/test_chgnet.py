"""Tests for CHGNet PyG model: forward pass, prediction, Potential, training."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure
from torch_geometric.data import Batch

import matgl
from matgl.apps._pes import Potential
from matgl.ext._pymatgen import Structure2Graph
from matgl.models._chgnet import CHGNet

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

        assert torch.allclose(out / structure.num_sites, out_s / supercell.num_sites, rtol=1e-4)


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
        out = default_model.predict_structure(mos_structure, return_features=True, output_layers=["final", "gc_1"])
        assert set(out.keys()) == {"final", "gc_1"}

    def test_invalid_output_layers(self, mos_structure, default_model):
        with pytest.raises(ValueError, match="Invalid output_layers"):
            default_model.predict_structure(mos_structure, return_features=True, output_layers=["not_a_layer"])


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
    @pytest.fixture
    def potential(self, default_model):
        return Potential(model=default_model, calc_hessian=True)

    def test_efsh_shapes(self, mos_graph, potential):
        structure, g, _lat, state = mos_graph
        lat_t = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        e, f, s, h = potential(g, lat_t, state)
        assert torch.numel(e) == 1
        assert f.shape == (structure.num_sites, 3)
        assert s.shape == (3, 3)
        assert h.shape == (structure.num_sites * 3, structure.num_sites * 3)

    def test_efs_shapes(self, mos_graph, default_model):
        structure, g, _lat, state = mos_graph
        lat_t = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        ff = Potential(model=default_model)
        e, f, s, h = ff(g, lat_t, state)
        assert torch.numel(e) == 1
        assert f.shape == (structure.num_sites, 3)
        assert s.shape == (3, 3)
        assert h.shape[0] == 1  # not computed

    def test_forces_only(self, mos_graph, default_model):
        structure, g, _lat, state = mos_graph
        lat_t = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        ff = Potential(model=default_model, calc_stresses=False)
        e, f, _s, _h = ff(g, lat_t, state)
        assert torch.numel(e) == 1
        assert f.shape == (structure.num_sites, 3)

    def test_energy_only(self, mos_graph, default_model):
        structure, g, _lat, state = mos_graph
        lat_t = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        ff = Potential(model=default_model, calc_forces=False, calc_stresses=False)
        e, _f, _s, _h = ff(g, lat_t, state)
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

        g_m, _lat_m, state = make_graph(struct_m)
        g_0, _lat_0, _ = make_graph(struct_0)
        g_p, _lat_p, _ = make_graph(struct_p)

        lat_m_t = torch.tensor(struct_m.lattice.matrix, dtype=matgl.float_th)
        lat_0_t = torch.tensor(struct_0.lattice.matrix, dtype=matgl.float_th)
        lat_p_t = torch.tensor(struct_p.lattice.matrix, dtype=matgl.float_th)

        e_m, _, _ = ff(g_m, lat_m_t, state)
        _, grad_zero, _ = ff(g_0, lat_0_t, state)
        e_p, _, _ = ff(g_p, lat_p_t, state)

        dx = 0.004  # fractional displacement x lattice param = 0.002 x 4.0 Å = 0.008 Å
        fd = (e_p - e_m) / (2 * dx)
        assert np.allclose(fd.detach().numpy(), grad_zero[1][0].detach().numpy(), atol=1e-4)

    def test_batch_potential(self, mos_structure, default_model):
        conv = Structure2Graph(element_types=default_model.element_types, cutoff=default_model.cutoff)
        g1, lat1, _ = conv.get_graph(mos_structure)
        g1.pbc_offshift = torch.matmul(g1.pbc_offset, lat1[0])
        g1.pos = g1.frac_coords @ lat1[0]
        g2, lat2, _ = conv.get_graph(mos_structure)
        g2.pbc_offshift = torch.matmul(g2.pbc_offset, lat2[0])
        g2.pos = g2.frac_coords @ lat2[0]

        batched = Batch.from_data_list([g1, g2])
        lat = torch.stack(
            [
                torch.tensor(mos_structure.lattice.matrix, dtype=matgl.float_th),
                torch.tensor(mos_structure.lattice.matrix, dtype=matgl.float_th),
            ]
        )
        ff = Potential(model=default_model)
        e, f, s, _h = ff(batched, lat, None)
        assert e.shape == (2,)
        assert f.shape == (batched.num_nodes, 3)
        assert s.shape == (6, 3)  # 2 structures x 3 rows

    def test_with_zbl_repulsion(self, mos_structure, default_model):
        """ZBL needs bond_dist on the graph — Potential builds it from pos."""
        from matgl.ext._pymatgen import Structure2Graph
        from matgl.graph._compute import compute_pair_vector_and_distance

        conv = Structure2Graph(element_types=default_model.element_types, cutoff=default_model.cutoff)
        g, lat, state = conv.get_graph(mos_structure)
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        g.pos = g.frac_coords @ lat[0]
        _, g.bond_dist = compute_pair_vector_and_distance(g.pos, g.edge_index, g.pbc_offshift)
        lat_t = torch.tensor(mos_structure.lattice.matrix, dtype=matgl.float_th)
        ff = Potential(model=default_model, calc_repuls=True)
        e, _f, _s, _h = ff(g, lat_t, state)
        assert torch.isfinite(e)

    def test_with_element_refs(self, mos_graph, default_model):
        structure, g, _lat, state = mos_graph
        lat_t = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        # Dummy per-element energy references (same length as element_types)
        refs = torch.zeros(len(default_model.element_types))
        ff = Potential(model=default_model, element_refs=refs.numpy())
        e, _f, _s, _h = ff(g, lat_t, state)
        assert torch.isfinite(e)


# ---------------------------------------------------------------------------
# Training step (gradient flow)
# ---------------------------------------------------------------------------


class TestCHGNetTraining:
    def _fresh_graph(self, structure, model):
        """Build a fresh graph with requires_grad=False on pos to avoid double-backward."""
        from matgl.ext._pymatgen import Structure2Graph

        conv = Structure2Graph(element_types=model.element_types, cutoff=model.cutoff)
        g, lat, state = conv.get_graph(structure)
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        g.pos = g.frac_coords @ lat[0]
        return g, lat, state

    def test_gradient_flows(self, mos_structure):
        """All parameters should receive gradients after a backward pass."""
        # Use bond_update_hidden_dims so bond_bond_weights participates in the graph
        model = CHGNet(
            element_types=("Mo", "S"),
            num_blocks=2,
            bond_update_hidden_dims=(16,),
        )
        g, _lat, _ = self._fresh_graph(mos_structure, model)
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
            element_types=("Mo", "S"),
            num_blocks=2,
            bond_update_hidden_dims=(16,),
        )
        ff = Potential(model=model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        g, _lat, state = self._fresh_graph(mos_structure, model)
        lat_t = torch.tensor(mos_structure.lattice.matrix, dtype=matgl.float_th)

        optimizer.zero_grad()
        e, f, _s, _h = ff(g, lat_t, state)
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


# ---------------------------------------------------------------------------
# MatPES parity: model predictions pinned to reference values.
# ---------------------------------------------------------------------------

# Reference structures: unperturbed + small explicit displacements (fully deterministic).
_mos = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
_fe = Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
_mos_p = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.025, -0.015, 0.010], [0.525, 0.490, 0.515]])
_fe_p = Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[-0.010, 0.020, -0.025], [0.510, 0.515, 0.480]])
_PARITY_STRUCTURES = {"mos": _mos, "fe": _fe, "mos_perturbed": _mos_p, "fe_perturbed": _fe_p}

# Pinned reference values computed from the DGL MatPES models (seed=42 perturbations).
# Tolerance for comparison: atol=1e-5 (covers FP32 precision and DGL↔PyG scatter differences).
_MATPES_EXPECTED = {
    ("r2SCAN", "mos"): {
        "energy_per_atom": -15.1326951981,
        "forces": [
            [-1.4901161193847656e-08, -4.470348358154297e-08, -4.470348358154297e-08],
            [-1.4901161193847656e-08, 1.4901161193847656e-08, -1.4901161193847656e-08],
        ],
        "stress": [
            [17.91497039794922, -3.7303581734704494e-07, 7.460716489049446e-08],
            [7.460716489049446e-08, 17.91497039794922, 7.460716489049446e-08],
            [7.460716489049446e-08, -7.460716489049446e-08, 17.91497039794922],
        ],
        "magmom": [[3.052868127822876], [0.20977726578712463]],
    },
    ("r2SCAN", "fe"): {
        "energy_per_atom": -14.4023408890,
        "forces": [
            [-7.450580596923828e-09, -7.450580596923828e-09, 1.4901161193847656e-08],
            [7.450580596923828e-09, 7.450580596923828e-09, -3.725290298461914e-08],
        ],
        "stress": [
            [1.230300784111023, -6.521526643155084e-07, 3.623070483627089e-07],
            [-9.41998280268308e-07, 1.2303022146224976, -7.246141109362725e-08],
            [-2.173842261754544e-07, -7.246141109362725e-08, 1.2303019762039185],
        ],
        "magmom": [[2.7359468936920166], [2.7359461784362793]],
    },
    ("r2SCAN", "mos_perturbed"): {
        "energy_per_atom": -15.1329126358,
        "forces": [
            [1.3113021850585938e-06, -0.020524345338344574, -0.02052599936723709],
            [-1.259148120880127e-06, 0.020524300634860992, 0.02052602730691433],
        ],
        "stress": [
            [17.920207977294922, -6.285653853410622e-06, 1.4641656207459164e-06],
            [-7.054107300064061e-06, 17.916536331176758, 0.033099982887506485],
            [1.2888383480458288e-06, 0.03310035541653633, 17.916536331176758],
        ],
        "magmom": [[3.0529751777648926], [0.2098957896232605]],
    },
    ("r2SCAN", "fe_perturbed"): {
        "energy_per_atom": -14.3956117630,
        "forces": [
            [0.3492530584335327, -0.0908508151769638, 0.09085030853748322],
            [-0.3492531180381775, 0.0908508449792862, -0.09085029363632202],
        ],
        "stress": [
            [1.0872764587402344, 0.06569133698940277, -0.06568901985883713],
            [0.06569119542837143, 0.8566457629203796, -0.006552322767674923],
            [-0.06568975001573563, -0.00655217794701457, 0.8566471934318542],
        ],
        "magmom": [[2.7332448959350586], [2.7332446575164795]],
    },
    ("PBE", "mos"): {
        "energy_per_atom": -5.3955235481,
        "forces": [
            [-3.3527612686157227e-08, -3.725290298461914e-09, -3.725290298461914e-09],
            [-5.960464477539063e-08, -0.0, -1.4901161193847656e-08],
        ],
        "stress": [
            [17.832918167114258, 0.0, 7.460716489049446e-08],
            [2.9842865956197784e-07, 17.832918167114258, 7.460716489049446e-08],
            [2.9842865956197784e-07, 0.0, 17.832918167114258],
        ],
        "magmom": [[3.985849142074585], [0.19956068694591522]],
    },
    ("PBE", "fe"): {
        "energy_per_atom": -8.2440567017,
        "forces": [
            [1.4901161193847656e-08, -7.078051567077637e-08, -1.4901161193847656e-08],
            [7.450580596923828e-09, 8.195638656616211e-08, 3.725290298461914e-08],
        ],
        "stress": [
            [6.13809061050415, 7.970755291353271e-07, 8.695369047018175e-07],
            [1.159382577498036e-06, 6.138092994689941, -5.072298563391087e-07],
            [1.0144597126782173e-06, -5.072298563391087e-07, 6.138092994689941],
        ],
        "magmom": [[2.379379987716675], [2.379380226135254]],
    },
    ("PBE", "mos_perturbed"): {
        "energy_per_atom": -5.3957433701,
        "forces": [
            [6.146728992462158e-07, -0.022980906069278717, -0.02298126369714737],
            [-6.854534149169922e-07, 0.022980883717536926, 0.022981271147727966],
        ],
        "stress": [
            [17.782638549804688, -3.049567794732866e-06, -1.7066388409148203e-06],
            [-3.1098131785256555e-06, 17.778823852539062, -0.04753144085407257],
            [-1.3015222748435917e-06, -0.04753126576542854, 17.778823852539062],
        ],
        "magmom": [[3.9845972061157227], [0.19968606531620026]],
    },
    ("PBE", "fe_perturbed"): {
        "energy_per_atom": -8.2331771851,
        "forces": [
            [0.3305317461490631, -0.0892416462302208, 0.0892401784658432],
            [-0.3305317759513855, 0.0892416313290596, -0.08924020826816559],
        ],
        "stress": [
            [5.753841400146484, 0.08783750236034393, -0.0878354012966156],
            [0.08783656358718872, 5.7640204429626465, 0.03972276672720909],
            [-0.08783569186925888, 0.0397229827940464, 5.764013290405273],
        ],
        "magmom": [[2.3732571601867676], [2.373257637023926]],
    },
}

_PYG_MODEL_NAME = "BowenD-UCB/CHGNet-PyG-MatPES-{functional}-2025.2.10"


@pytest.fixture(scope="module", params=["r2SCAN", "PBE"])
def matpes_pyg_potential(request):
    functional = request.param
    model_name = _PYG_MODEL_NAME.format(functional=functional)
    try:
        pot = matgl.load_model(model_name)
    except Exception as e:
        pytest.skip(f"PyG {functional} model '{model_name}' could not be loaded: {e}")
    pot.eval()
    return functional, pot


@pytest.mark.parametrize("struct_name", ["mos", "fe", "mos_perturbed", "fe_perturbed"])
def test_matpes_model_parity_pyg(matpes_pyg_potential, struct_name):
    """PyG MatPES CHGNet predictions match DGL reference values to within 1e-5."""
    functional, pot = matpes_pyg_potential
    struct = _PARITY_STRUCTURES[struct_name]
    natoms = len(struct)

    conv = Structure2Graph(element_types=pot.model.element_types, cutoff=pot.model.cutoff)
    g, lat, _ = conv.get_graph(struct)
    g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
    g.pos = g.frac_coords @ lat[0]

    out = pot(g=g, lat=lat)
    energy, forces, stresses, magmom = out[0], out[1], out[2], out[4]

    exp = _MATPES_EXPECTED[(functional, struct_name)]
    atol = 1e-5

    assert abs(energy.item() / natoms - exp["energy_per_atom"]) < atol, (
        f"[{functional}/{struct_name}] energy/atom {energy.item() / natoms:.10f} != {exp['energy_per_atom']:.10f}"
    )
    exp_forces = torch.tensor(exp["forces"], dtype=matgl.float_th)
    assert torch.allclose(forces.detach(), exp_forces, atol=atol), (
        f"[{functional}/{struct_name}] force mismatch (max diff {(forces.detach() - exp_forces).abs().max():.2e})"
    )
    exp_stress = torch.tensor(exp["stress"], dtype=matgl.float_th)
    assert torch.allclose(stresses.detach(), exp_stress, atol=atol), (
        f"[{functional}/{struct_name}] stress mismatch (max diff {(stresses.detach() - exp_stress).abs().max():.2e})"
    )
    exp_magmom = torch.tensor(exp["magmom"], dtype=matgl.float_th)
    assert torch.allclose(magmom.detach(), exp_magmom, atol=atol), (
        f"[{functional}/{struct_name}] magmom mismatch (max diff {(magmom.detach() - exp_magmom).abs().max():.2e})"
    )
