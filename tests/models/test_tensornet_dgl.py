from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import matgl

if matgl.config.BACKEND != "DGL":
    pytest.skip("Skipping DGL tests", allow_module_level=True)
from matgl.models._tensornet_dgl import TensorNet


class TestTensorNet:
    def test_model(self, graph_MoS):
        _, graph, _ = graph_MoS
        for act in ["swish", "tanh", "sigmoid", "softplus2", "softexp"]:
            model = TensorNet(is_intensive=False, activation_type=act)
            output = model(g=graph)
            assert torch.numel(output) == 1
        model.save(".")
        TensorNet.load(".")
        os.remove("model.pt")
        os.remove("model.json")
        os.remove("state.pt")
        model = TensorNet(is_intensive=False, equivariance_invariance_group="SO(3)")
        assert torch.numel(output) == 1

    def test_exceptions(self):
        with pytest.raises(ValueError, match="Invalid activation type"):
            _ = TensorNet(element_types=None, is_intensive=False, activation_type="whatever")
        with pytest.raises(ValueError, match=r"Classification task cannot be extensive."):
            _ = TensorNet(element_types=["Mo", "S"], is_intensive=False, task_type="classification")

    def test_model_intensive(self, graph_MoS):
        structure, graph, _ = graph_MoS
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
        graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
        graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
        model = TensorNet(element_types=["Mo", "S"], is_intensive=True)
        output = model(g=graph)
        assert torch.numel(output) == 1

    def test_model_intensive_with_weighted_atom(self, graph_MoS):
        structure, graph, _ = graph_MoS
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
        graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
        graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
        model = TensorNet(element_types=["Mo", "S"], is_intensive=True, readout_type="weighted_atom")
        output = model(g=graph)
        assert torch.numel(output) == 1

    def test_model_intensive_with_ReduceReadOut(self, graph_MoS):
        structure, graph, _ = graph_MoS
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
        graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
        graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
        model = TensorNet(is_intensive=True, readout_type="reduce_atom")
        output = model(g=graph)
        assert torch.numel(output) == 1

    def test_model_intensive_with_classification(self, graph_MoS):
        structure, graph, _ = graph_MoS
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
        graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
        graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
        model = TensorNet(
            element_types=["Mo", "S"],
            is_intensive=True,
            task_type="classification",
        )
        output = model(g=graph)
        assert torch.numel(output) == 1

    def test_model_intensive_set2set_classification(self, graph_MoS):
        structure, graph, _ = graph_MoS
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
        graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
        graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]
        model = TensorNet(
            element_types=["Mo", "S"], is_intensive=True, task_type="classification", readout_type="set2set"
        )
        output = model(g=graph)
        assert torch.numel(output) == 1

    def test_return_features(self, graph_MoS):
        structure, graph, _ = graph_MoS
        lat = torch.tensor(np.array([structure.lattice.matrix]), dtype=matgl.float_th)
        graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lat[0])
        graph.ndata["pos"] = graph.ndata["frac_coords"] @ lat[0]

        model = TensorNet(element_types=["Mo", "S"], is_intensive=True)

        # Test default return (just final property)
        out = model.predict_structure(structure, return_features=False)
        assert isinstance(out, torch.Tensor)

        # Test return features
        out_feats = model.predict_structure(structure, return_features=True)
        assert isinstance(out_feats, dict)
        assert "final" in out_feats
        assert "readout" in out_feats
        assert "edge_attr" in out_feats
        assert "embedding" in out_feats
        assert "gc_1" in out_feats

        # Check shapes
        assert out_feats["final"].shape == torch.Size([])  # Scalar output
        assert out_feats["readout"].shape[0] == structure.num_sites

        # Test specific output layers
        out_feats_subset = model.predict_structure(structure, return_features=True, output_layers=["final", "gc_1"])
        assert set(out_feats_subset.keys()) == {"final", "gc_1"}
