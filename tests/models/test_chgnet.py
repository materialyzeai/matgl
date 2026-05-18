from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

import matgl

if matgl.config.BACKEND != "DGL":
    pytest.skip("Skipping DGL tests", allow_module_level=True)
from matgl.ext._ase_dgl import PESCalculator
from matgl.ext._pymatgen_dgl import Structure2Graph
from matgl.models import CHGNet


@pytest.mark.parametrize("threebody_cutoff", [0, 3])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
@pytest.mark.parametrize("learn_basis", [True, False])
@pytest.mark.parametrize("bond_dim", [None, (16,)])
@pytest.mark.parametrize("angle_dim", [None, (16,)])
@pytest.mark.parametrize("activation", ["swish", "softplus2"])
def test_model(graph_MoS, threebody_cutoff, activation, angle_dim, bond_dim, learn_basis, dropout):
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
    graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
    for readout_field in ["atom_feat", "bond_feat", "angle_feat"]:
        if readout_field == "angle_feat" and threebody_cutoff == 0:
            continue
        for final_mlp_type in ["gated", "mlp"]:
            model = CHGNet(
                element_types=("Mo", "S"),
                threebody_cutoff=threebody_cutoff,
                activation_type=activation,
                bond_update_hidden_dims=bond_dim,
                learn_basis=learn_basis,
                angle_update_hidden_dims=angle_dim,
                conv_dropout=dropout,
                readout_field=readout_field,
                final_mlp_type=final_mlp_type,
            )
            global_out = model(g=graph)
            assert torch.numel(global_out) == 1
            assert torch.numel(graph.ndata["magmom"]) == graph.num_nodes()
            model.save(".")
            CHGNet.load(".")
            os.remove("model.pt")
            os.remove("model.json")
            os.remove("state.pt")


def test_exceptions():
    with pytest.raises(ValueError, match="Invalid activation type"):
        _ = CHGNet(element_types=None, is_intensive=False, activation_type="whatever")


@pytest.mark.parametrize("structure", ["LiFePO4", "BaNiO3", "MoS"])
def test_prediction_validity(structure, request):
    structure = request.getfixturevalue(structure)
    supercell1 = structure.copy()
    supercell1.make_supercell([2, 4, 3])
    supercell2 = structure.copy()
    supercell2.make_supercell(2)

    model = CHGNet()
    converter = Structure2Graph(element_types=model.element_types, cutoff=model.cutoff)

    g, lattice, _ = converter.get_graph(structure)
    g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lattice[0])
    g.ndata["pos"] = g.ndata["frac_coords"] @ lattice[0]

    g1, lattice2, _ = converter.get_graph(supercell1)
    g1.edata["pbc_offshift"] = torch.matmul(g1.edata["pbc_offset"], lattice2[0])
    g1.ndata["pos"] = g1.ndata["frac_coords"] @ lattice2[0]

    g2, lattice3, _ = converter.get_graph(supercell2)
    g2.edata["pbc_offshift"] = torch.matmul(g2.edata["pbc_offset"], lattice3[0])
    g2.ndata["pos"] = g2.ndata["frac_coords"] @ lattice3[0]

    out = model(g)
    out1 = model(g1)
    out2 = model(g2)

    assert not torch.allclose(out, out1)
    assert not torch.allclose(out, out2)

    assert torch.allclose(out / g.num_nodes(), out1 / g1.num_nodes(), rtol=1e-4)
    assert torch.allclose(out / g.num_nodes(), out2 / g2.num_nodes(), rtol=1e-4)

    assert len(g.ndata["magmom"]) == g.num_nodes()
    assert len(g1.ndata["magmom"]) == g1.num_nodes()
    assert len(g2.ndata["magmom"]) == g2.num_nodes()

    assert torch.allclose(
        torch.unique(torch.round(g.ndata["magmom"], decimals=4), sorted=True),
        torch.unique(torch.round(g2.ndata["magmom"], decimals=4), sorted=True),
    )
    assert torch.allclose(
        torch.unique(torch.round(g.ndata["magmom"], decimals=4), sorted=True),
        torch.unique(torch.round(g2.ndata["magmom"], decimals=4), sorted=True),
    )


@pytest.mark.parametrize("structure", ["Li3InCl6"])
def test_lg_error_handling(structure, request):
    structure = request.getfixturevalue(structure)

    dummy_chgnet = CHGNet(cutoff=6.0, threebody_cutoff=3.0)
    # This structure triggers RuntimeError without error handling
    with pytest.raises(RuntimeError):
        dummy_chgnet.predict_structure(structure, error_handling=False)

    # With error handling it only prints warning
    with pytest.warns(RuntimeWarning):
        out = dummy_chgnet.predict_structure(structure, error_handling=True)
    assert isinstance(out, torch.Tensor)


@pytest.mark.parametrize("structure", ["Li3InCl6"])
@pytest.mark.parametrize("threebody_cutoff", [3, 2.8])
def test_prediction_stability_against_graph_cutoff_perturbation(structure, threebody_cutoff, request):
    # This test ensure that energy and force predictions don't actually get modified after
    # numerical perturbation to solve the RuntimeError
    structure = request.getfixturevalue(structure)

    potential1 = matgl.load_model("CHGNet-PES-MatPES-PBE-2025.2.10")
    potential1.threebody_cutoff = threebody_cutoff
    calculator = PESCalculator(potential1)
    forces1 = calculator.get_forces(AseAtomsAdaptor.get_atoms(structure))

    potential2 = matgl.load_model("CHGNet-PES-MatPES-PBE-2025.2.10")
    potential2.model.threebody_cutoff = threebody_cutoff + 1e-6
    assert potential2.model.threebody_cutoff > threebody_cutoff
    calculator2 = PESCalculator(potential2)
    forces2 = calculator2.get_forces(AseAtomsAdaptor.get_atoms(structure))
    assert np.allclose(forces1, forces2, rtol=1e-3, atol=1e-6)


def test_return_features(graph_MoS):
    structure, graph, _ = graph_MoS
    lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
    graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
    graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]

    model = CHGNet(element_types=("Mo", "S"))

    # Test default return (just final property)
    out = model.predict_structure(structure, return_features=False)
    assert isinstance(out, torch.Tensor)

    # Test return features
    out_feats = model.predict_structure(structure, return_features=True)
    assert isinstance(out_feats, dict)
    assert "final" in out_feats
    assert "readout" in out_feats
    assert "bond_expansion" in out_feats
    assert "embedding" in out_feats
    assert "gc_1" in out_feats

    # Check shapes
    assert out_feats["final"].shape == torch.Size([])  # Scalar output
    assert out_feats["readout"]["atom_feat"].shape[0] == structure.num_sites

    # Test specific output layers
    out_feats_subset = model.predict_structure(structure, return_features=True, output_layers=["final", "gc_1"])
    assert set(out_feats_subset.keys()) == {"final", "gc_1"}


# ---------------------------------------------------------------------------
# MatPES parity: DGL model predictions pinned as reference values.
# The PyG counterpart (test_chgnet_pyg.py::test_matpes_model_parity_pyg)
# compares against the same values with MATGL_BACKEND=PYG. If both pass,
# DGL ↔ PyG parity is guaranteed.
# ---------------------------------------------------------------------------

_mos = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
_fe = Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
_mos_p = Structure(Lattice.cubic(4.0), ["Mo", "S"], [[0.025, -0.015, 0.010], [0.525, 0.490, 0.515]])
_fe_p = Structure(Lattice.cubic(2.87), ["Fe", "Fe"], [[-0.010, 0.020, -0.025], [0.510, 0.515, 0.480]])
_PARITY_STRUCTURES = {"mos": _mos, "fe": _fe, "mos_perturbed": _mos_p, "fe_perturbed": _fe_p}

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
            [-8.195638656616211e-08, -5.21540641784668e-08, -2.2351741790771484e-08],
            [6.705522537231445e-08, 5.960464477539063e-08, 2.2351741790771484e-08],
        ],
        "stress": [
            [1.2303011417388916, -5.79691288749018e-07, -2.173842261754544e-07],
            [-7.970755291353271e-07, 1.2303011417388916, -2.173842261754544e-07],
            [-6.521526643155084e-07, 0.0, 1.2303005456924438],
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
            [4.0978193283081055e-08, -6.705522537231445e-08, 6.705522537231445e-08],
            [3.725290298461914e-09, 1.0803341865539551e-07, -5.587935447692871e-08],
        ],
        "stress": [
            [6.13809061050415, 4.70999140134154e-07, 6.159219765322632e-07],
            [4.70999140134154e-07, 6.138092041015625, 3.6230705546813624e-08],
            [1.086921130877272e-07, -2.5361492816955433e-07, 6.1380934715271],
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


@pytest.fixture(scope="module", params=["r2SCAN", "PBE"])
def matpes_dgl_potential(request):
    functional = request.param
    pot = matgl.load_model(f"CHGNet-PES-MatPES-{functional}-2025.2.10")
    pot.eval()
    return functional, pot


@pytest.mark.parametrize("struct_name", ["mos", "fe", "mos_perturbed", "fe_perturbed"])
def test_matpes_model_parity_dgl(matpes_dgl_potential, struct_name):
    """DGL MatPES CHGNet predictions match pinned reference values to within 1e-5."""
    functional, pot = matpes_dgl_potential
    struct = _PARITY_STRUCTURES[struct_name]
    natoms = len(struct)

    conv = Structure2Graph(element_types=pot.model.element_types, cutoff=pot.model.cutoff)
    g, lat, _ = conv.get_graph(struct)
    g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lat[0])
    g.ndata["pos"] = g.ndata["frac_coords"] @ lat[0]

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
