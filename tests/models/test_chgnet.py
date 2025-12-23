from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from pymatgen.io.ase import AseAtomsAdaptor

import matgl

if matgl.config.BACKEND != "DGL":
    pytest.skip("Skipping DGL tests", allow_module_level=True)
from matgl.ext._ase_dgl import PESCalculator
from matgl.ext._pymatgen_dgl import Structure2Graph
from matgl.models import CHGNet


class TestCHGNet:
    @pytest.mark.parametrize("threebody_cutoff", [0, 3])
    @pytest.mark.parametrize("dropout", [0.0, 0.5])
    @pytest.mark.parametrize("learn_basis", [True, False])
    @pytest.mark.parametrize("bond_dim", [None, (16,)])
    @pytest.mark.parametrize("angle_dim", [None, (16,)])
    @pytest.mark.parametrize("activation", ["swish", "softplus2"])
    def test_model(self, graph_MoS, threebody_cutoff, activation, angle_dim, bond_dim, learn_basis, dropout):
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

    def test_exceptions(self):
        with pytest.raises(ValueError, match="Invalid activation type"):
            _ = CHGNet(element_types=None, is_intensive=False, activation_type="whatever")

    @pytest.mark.parametrize("structure", ["LiFePO4", "BaNiO3", "MoS"])
    def test_prediction_validity(self, structure, request):
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
    def test_lg_error_handling(self, structure, request):
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
    def test_prediction_stability_against_graph_cutoff_perturbation(self, structure, threebody_cutoff, request):
        # This test ensure that energy and force predictions don't actually get modified after
        # numerical perturbation to solve the RuntimeError
        structure = request.getfixturevalue(structure)

        potential1 = matgl.load_model("pretrained_models/CHGNet-MatPES-PBE-2025.2.10-2.7M-PES")
        potential1.threebody_cutoff = threebody_cutoff
        calculator = PESCalculator(potential1)
        forces1 = calculator.get_forces(AseAtomsAdaptor.get_atoms(structure))

        potential2 = matgl.load_model("pretrained_models/CHGNet-MatPES-PBE-2025.2.10-2.7M-PES")
        potential2.model.threebody_cutoff = threebody_cutoff + 1e-6
        assert potential2.model.threebody_cutoff > threebody_cutoff
        calculator2 = PESCalculator(potential2)
        forces2 = calculator2.get_forces(AseAtomsAdaptor.get_atoms(structure))
        assert np.allclose(forces1, forces2, rtol=1e-3, atol=1e-6)
