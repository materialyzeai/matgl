"""Tests for the electrostatic-potential aggregator, including DGL <-> PyG parity."""

from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import pytest
import torch

import matgl
from matgl.config import COULOMB_CONSTANT
from matgl.electrostatics._elec_pot_pyg import ElectrostaticPotential as ElectrostaticPotentialPyG
from matgl.utils.cutoff import polynomial_cutoff


def _has_dgl() -> bool:
    try:
        importlib.import_module("dgl")
    except Exception:  # DGL has many import-time failure modes (missing libs, version skew)
        return False
    return True


def _make_pyg_graph(pos: torch.Tensor, edge_index: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(pos=pos, edge_index=edge_index)


def test_elec_pot_pyg_two_atoms_against_analytic():
    """Two atoms, single bidirectional bond: hand-computed potential must match."""
    cutoff = 5.0
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=matgl.float_th)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    charge = torch.tensor([0.7, -0.7], dtype=matgl.float_th)
    sigma = torch.tensor([0.5, 0.6], dtype=matgl.float_th)

    g = _make_pyg_graph(pos, edge_index)
    out = ElectrostaticPotentialPyG(element_types=("X", "Y"), cutoff=cutoff)(g, charge=charge, sigma=sigma)

    r = float(torch.linalg.norm(pos[0] - pos[1]))
    gamma = math.sqrt(0.5**2 + 0.6**2)
    cutoff_factor = float(polynomial_cutoff(torch.tensor(r, dtype=matgl.float_th), cutoff))
    edge_factor = math.erf(r / math.sqrt(2.0) / gamma) * cutoff_factor / r * COULOMB_CONSTANT
    expected = torch.tensor([charge[1].item() * edge_factor, charge[0].item() * edge_factor], dtype=matgl.float_th)

    assert torch.allclose(out, expected, atol=1e-6)


def test_elec_pot_pyg_zero_charges_give_zero_potential():
    """No charge anywhere -> identically zero potential (sanity)."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.7, 0.0]], dtype=matgl.float_th)
    edge_index = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long)
    charge = torch.zeros(3, dtype=matgl.float_th)
    sigma = torch.tensor([0.5, 0.5, 0.5], dtype=matgl.float_th)

    g = _make_pyg_graph(pos, edge_index)
    out = ElectrostaticPotentialPyG(element_types=("X",), cutoff=5.0)(g, charge=charge, sigma=sigma)
    assert torch.allclose(out, torch.zeros_like(out))


def test_elec_pot_pyg_gradient_flow():
    """Gradients must flow back to atomic positions via the differentiable cutoff and erf."""
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.double, requires_grad=True)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    charge = torch.tensor([0.4, -0.4], dtype=torch.double)
    sigma = torch.tensor([0.5, 0.5], dtype=torch.double)

    g = _make_pyg_graph(pos, edge_index)
    module = ElectrostaticPotentialPyG(element_types=("X",), cutoff=5.0).double()
    out = module(g, charge=charge, sigma=sigma).sum()
    out.backward()
    assert pos.grad is not None
    assert torch.isfinite(pos.grad).all()
    # Symmetric bond: grads on the two atoms must be exact opposites.
    assert torch.allclose(pos.grad[0], -pos.grad[1], atol=1e-10)


@pytest.mark.skipif(not _has_dgl(), reason="DGL not importable in this environment")
def test_elec_pot_dgl_pyg_parity():
    """Identical inputs must give identical per-atom potentials in DGL and PyG."""
    import dgl

    from matgl.electrostatics._elec_pot_dgl import ElectrostaticPotential as ElectrostaticPotentialDGL

    cutoff = 5.0
    torch.manual_seed(0)
    n = 6
    pos = torch.randn(n, 3, dtype=matgl.float_th)
    # Build a fully connected (no self-loops) symmetric edge list.
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            src.append(i)
            dst.append(j)
    src_t = torch.tensor(src, dtype=torch.long)
    dst_t = torch.tensor(dst, dtype=torch.long)
    bond_dist = torch.linalg.norm(pos[src_t] - pos[dst_t], dim=1)

    charge = torch.randn(n, dtype=matgl.float_th)
    sigma = torch.rand(n, dtype=matgl.float_th) + 0.3

    # DGL side
    g_dgl = dgl.graph((src_t, dst_t), num_nodes=n)
    g_dgl.ndata["charge"] = charge
    g_dgl.ndata["sigma"] = sigma
    g_dgl.edata["bond_dist"] = bond_dist
    out_dgl = ElectrostaticPotentialDGL(element_types=("X",), cutoff=cutoff)(g_dgl).ndata["elec_pot"]

    # PyG side
    g_pyg = _make_pyg_graph(pos, torch.stack([src_t, dst_t], dim=0))
    out_pyg = ElectrostaticPotentialPyG(element_types=("X",), cutoff=cutoff)(g_pyg, charge=charge, sigma=sigma)

    assert torch.allclose(out_dgl, out_pyg, atol=1e-6)
