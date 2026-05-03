"""Tests for the QEq solver, including DGL <-> PyG numerical parity."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

import matgl
from matgl.electrostatics._fast_qeq_pyg import LinearQeq as LinearQeqPyG


def _has_dgl() -> bool:
    try:
        importlib.import_module("dgl")
    except Exception:  # DGL has many import-time failure modes (missing libs, version skew)
        return False
    return True


def test_qeq_pyg_two_atoms_analytic():
    """Closed-form solution must hold for a 2-atom graph with known chi/hardness."""
    chi = torch.tensor([0.4, -0.6], dtype=matgl.float_th)
    hardness = torch.tensor([2.0, 1.5], dtype=matgl.float_th)
    total_charge = torch.tensor([0.0], dtype=matgl.float_th)

    h_inv = hardness.reciprocal()
    expected = -chi * h_inv + h_inv * (total_charge.sum() + (chi * h_inv).sum()) / h_inv.sum()

    g = SimpleNamespace(batch=torch.zeros(2, dtype=torch.long), num_graphs=1)

    out = LinearQeqPyG()(g, total_charge, chi, hardness)
    assert torch.allclose(out, expected, atol=1e-6)


def test_qeq_pyg_batch_charge_conservation():
    """Per-graph charges must sum to the per-graph total_charge constraint."""
    torch.manual_seed(0)
    chi = torch.randn(7, dtype=matgl.float_th)
    hardness = torch.rand(7, dtype=matgl.float_th) + 0.5
    batch = torch.tensor([0, 0, 1, 1, 1, 2, 2], dtype=torch.long)
    total_charge = torch.tensor([0.0, 1.0, -2.0], dtype=matgl.float_th)

    g = SimpleNamespace(batch=batch, num_graphs=3)

    charges = LinearQeqPyG()(g, total_charge, chi, hardness)
    summed = torch.zeros(3, dtype=matgl.float_th).scatter_add_(0, batch, charges)
    assert torch.allclose(summed, total_charge, atol=1e-6)


def test_qeq_pyg_q_ref_overrides_total_charge():
    """When ``q_ref`` is present on the graph, its per-graph sum overrides ``total_charge``."""
    chi = torch.tensor([0.1, -0.2, 0.3], dtype=matgl.float_th)
    hardness = torch.tensor([1.5, 1.0, 2.0], dtype=matgl.float_th)
    q_ref = torch.tensor([0.4, 0.4, -1.0], dtype=matgl.float_th)

    g = SimpleNamespace(batch=torch.tensor([0, 0, 1], dtype=torch.long), num_graphs=2, q_ref=q_ref)

    charges = LinearQeqPyG()(g, total_charge=torch.tensor([99.0, 99.0]), chi=chi, hardness=hardness)
    summed = torch.zeros(2, dtype=matgl.float_th).scatter_add_(0, g.batch, charges)
    expected = torch.tensor([0.8, -1.0], dtype=matgl.float_th)
    assert torch.allclose(summed, expected, atol=1e-6)


@pytest.mark.skipif(not _has_dgl(), reason="DGL not importable in this environment")
def test_qeq_dgl_pyg_parity():
    """DGL and PyG QEq solvers must produce identical charges on the same input."""
    import dgl

    from matgl.electrostatics._fast_qeq_dgl import LinearQeq as LinearQeqDGL

    torch.manual_seed(7)
    n1, n2 = 4, 3
    chi = torch.randn(n1 + n2, dtype=matgl.float_th)
    hardness = torch.rand(n1 + n2, dtype=matgl.float_th) + 0.5
    total_charge = torch.tensor([0.5, -1.5], dtype=matgl.float_th)

    # DGL side: build two trivial graphs (edges don't matter for QEq) and batch them.
    g1 = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 0]), num_nodes=n1)
    g2 = dgl.graph(([0, 1, 2], [1, 2, 0]), num_nodes=n2)
    g1.ndata["chi"] = chi[:n1]
    g1.ndata["hardness"] = hardness[:n1]
    g2.ndata["chi"] = chi[n1:]
    g2.ndata["hardness"] = hardness[n1:]
    bg = dgl.batch([g1, g2])
    out_dgl = LinearQeqDGL()(bg, total_charge.reshape(-1, 1)).ndata["charge"]

    # PyG side: equivalent batch metadata.
    g = SimpleNamespace(batch=torch.tensor([0] * n1 + [1] * n2, dtype=torch.long), num_graphs=2)
    out_pyg = LinearQeqPyG()(g, total_charge, chi, hardness)

    assert torch.allclose(out_dgl, out_pyg, atol=1e-6)
