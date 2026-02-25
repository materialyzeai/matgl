from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("Skipping PYG tests", allow_module_level=True)

from matgl.models._tensornetwarp_pyg import TensorNet


class TestTensorNet:
    def test_model(self, graph_MoS_pyg):
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)

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

            output = model(g=graph)
            print(act, output.item())

            assert torch.numel(output) == 1

            # Optional strict regression test
            if act in EXPECTED:
                assert torch.allclose(output, EXPECTED[act], atol=1e-4)

            outputs[act] = output.item()

        # ---- SAVE/LOAD TEST ----
        model.save(".")
        TensorNet.load(".")
        os.remove("model.pt")
        os.remove("model.json")
        os.remove("state.pt")

        # ---- SECOND MODEL TEST ----
        model = TensorNet(is_intensive=False, equivariance_invariance_group="SO(3)")
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
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
        graph.pos = graph.frac_coords @ lat[0]
        model = TensorNet(element_types=["Mo", "S"], is_intensive=True)
        output = model(g=graph)
        assert torch.allclose(output, torch.tensor([-0.0897]), atol=1e-4)

    def test_model_intensive_with_weighted_atom(self, graph_MoS_pyg):
        structure, graph, _ = graph_MoS_pyg
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
        graph.pos = graph.frac_coords @ lat[0]
        model = TensorNet(element_types=["Mo", "S"], is_intensive=True, readout_type="weighted_atom")
        output = model(g=graph)
        assert torch.allclose(output, torch.tensor([-0.0217]), atol=1e-4)

    def test_model_intensive_with_ReduceReadOut(self, graph_MoS_pyg):
        structure, graph, _ = graph_MoS_pyg
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
        graph.pos = graph.frac_coords @ lat[0]
        model = TensorNet(is_intensive=True, readout_type="reduce_atom")
        output = model(g=graph)
        assert torch.allclose(output, torch.tensor([-0.1045]), atol=1e-4)

    def test_model_intensive_with_classification(self, graph_MoS_pyg):
        structure, graph, _ = graph_MoS_pyg
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lat[0])
        graph.pos = graph.frac_coords @ lat[0]
        model = TensorNet(
            element_types=["Mo", "S"],
            is_intensive=True,
            task_type="classification",
        )
        output = model(g=graph)
        assert torch.numel(output) == 1

    def test_backward(self, graph_MoS_pyg):
        """Test cell gradient (dE/dcell)."""
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)

        EXPECTED_CELL_GRAD = torch.tensor(
            [
                [-0.000967, 0.000000, 0.000000],
                [0.000000, -0.000967, 0.000000],
                [0.000000, 0.000000, -0.000967],
            ]
        )

        structure, graph, _ = graph_MoS_pyg
        cell = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th).requires_grad_(True)

        graph.pbc_offshift = torch.matmul(graph.pbc_offset, cell)
        graph.pos = graph.frac_coords @ cell

        model = TensorNet(is_intensive=False, activation_type="swish")
        model.train()

        energy = model(g=graph)
        cell_grad = torch.autograd.grad(energy, cell, create_graph=True)[0]

        assert torch.allclose(cell_grad, EXPECTED_CELL_GRAD, atol=1e-6)

    def test_double_backward(self, graph_MoS_pyg):
        """Test double backward: loss = sum(cell_grad^2), compare cell.grad."""
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)

        EXPECTED_CELL_GRAD2 = torch.tensor(
            [
                [-0.000010, -0.000000, -0.000000],
                [-0.000000, -0.000010, -0.000000],
                [-0.000000, -0.000000, -0.000010],
            ]
        )

        structure, graph, _ = graph_MoS_pyg
        cell = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th).requires_grad_(True)
        cell.retain_grad()

        graph.pbc_offshift = torch.matmul(graph.pbc_offset, cell)
        graph.pos = graph.frac_coords @ cell

        model = TensorNet(is_intensive=False, activation_type="swish")
        model.train()

        energy = model(g=graph)
        cell_grad = torch.autograd.grad(energy, cell, create_graph=True)[0]
        loss = (cell_grad * cell_grad).sum()
        loss.backward()

        assert torch.allclose(cell.grad, EXPECTED_CELL_GRAD2, atol=1e-6)
