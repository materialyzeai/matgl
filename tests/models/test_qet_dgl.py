from __future__ import annotations

import os

import pytest
import torch

import matgl

if matgl.config.BACKEND != "DGL":
    pytest.skip("Skipping DGL tests", allow_module_level=True)
from matgl.models._qet_dgl import QET


def test_qet(graph_MoS):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)

    # Optional regression-check values
    EXPECTED = {
        "swish": torch.tensor(-0.0112),
        "tanh": torch.tensor(-0.1138),
        "sigmoid": torch.tensor(0.1377),
        "softplus2": torch.tensor(0.0127),
        "softexp": torch.tensor(0.0819),
    }

    _, graph, _ = graph_MoS

    activations = ["swish", "tanh", "sigmoid", "softplus2", "softexp"]

    outputs = {}
    for act in activations:
        model = QET(is_intensive=False, activation_type=act)

        output = model(g=graph, total_charge=torch.tensor([0.0]))
        assert torch.numel(output) == 1

        # Optional strict regression test
        if act in EXPECTED:
            assert torch.allclose(output, EXPECTED[act], atol=1e-4)

        outputs[act] = output.item()

    # ---- SAVE/LOAD TEST ----
    model.save(".")
    QET.load(".")
    os.remove("model.pt")
    os.remove("model.json")
    os.remove("state.pt")
    model = QET(is_intensive=False, equivariance_invariance_group="SO(3)")

    assert torch.numel(output) == 1
