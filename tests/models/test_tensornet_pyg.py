from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("Skipping PYG tests", allow_module_level=True)
from matgl.models._tensornet_pyg import TensorNet


class TestTensorNet:
    def test_model(self, graph_MoS_pyg):
        torch.manual_seed(0)
        # torch.use_deterministic_algorithms(True)

        # Optional regression-check values
        EXPECTED = {
            "swish": torch.tensor(0.0813),
            "tanh": torch.tensor(-0.0189),
            "sigmoid": torch.tensor(0.0353),
            "softplus2": torch.tensor(0.1164),
            "softexp": torch.tensor(0.1148),
        }

        _, graph, _ = graph_MoS_pyg

        activations = ["swish", "tanh", "sigmoid", "softplus2", "softexp"]

        outputs = {}
        for act in activations:
            model = TensorNet(is_intensive=False, activation_type=act)
            model.to(graph.pos.device)

            output = model(g=graph)
            print(act, output.item())

            assert torch.numel(output) == 1

            # Optional strict regression test
            if act in EXPECTED:
                assert torch.allclose(output.cpu(), EXPECTED[act], atol=1e-4)

            outputs[act] = output.item()

        # ---- SAVE/LOAD TEST ----
        model.save(".")
        TensorNet.load(".")
        os.remove("model.pt")
        os.remove("model.json")
        os.remove("state.pt")

        # ---- SECOND MODEL TEST ----
        model = TensorNet(is_intensive=False, equivariance_invariance_group="SO(3)")
        model.to(graph.pos.device)
        output = model(g=graph)

        # this model outputs a 2-vector (as you wanted)
        assert torch.numel(output) == 1

    def test_exceptions(self):
        with pytest.raises(ValueError, match="Invalid activation type"):
            _ = TensorNet(element_types=None, is_intensive=False, activation_type="whatever")
        with pytest.raises(ValueError, match=r"Classification task cannot be extensive."):
            _ = TensorNet(element_types=["Mo", "S"], is_intensive=False, task_type="classification")

    def test_model_intensive(self, graph_MoS_pyg):
        structure, graph, _ = graph_MoS_pyg
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=graph.pos.device)
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
        graph.pos = graph.frac_coords @ lat[0]
        model = TensorNet(element_types=["Mo", "S"], is_intensive=True)
        model.to(graph.pos.device)
        output = model(g=graph)
        assert torch.allclose(output.detach().cpu(), torch.tensor([-0.0897]), atol=1e-4)

    def test_model_intensive_with_weighted_atom(self, graph_MoS_pyg):
        structure, graph, _ = graph_MoS_pyg
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=graph.pos.device)
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
        graph.pos = graph.frac_coords @ lat[0]
        model = TensorNet(element_types=["Mo", "S"], is_intensive=True, readout_type="weighted_atom")
        model.to(graph.pos.device)
        output = model(g=graph)
        assert torch.allclose(output.detach().cpu(), torch.tensor([-0.0217]), atol=1e-4)

    def test_model_intensive_with_ReduceReadOut(self, graph_MoS_pyg):
        structure, graph, _ = graph_MoS_pyg
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=graph.pos.device)
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
        graph.pos = graph.frac_coords @ lat[0]
        model = TensorNet(is_intensive=True, readout_type="reduce_atom")
        model.to(graph.pos.device)
        output = model(g=graph)
        assert torch.allclose(output.detach().cpu(), torch.tensor([-0.1045]), atol=1e-4)

    def test_model_intensive_with_classification(self, graph_MoS_pyg):
        structure, graph, _ = graph_MoS_pyg
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th, device=graph.pos.device)
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
        graph.pos = graph.frac_coords @ lat[0]
        model = TensorNet(
            element_types=["Mo", "S"],
            is_intensive=True,
            task_type="classification",
        )
        model.to(graph.pos.device)
        output = model(g=graph)
        assert torch.allclose(output.detach().cpu(), torch.tensor([0.5090]), atol=1e-4)
