"""Tests for the electrostatic-potential aggregator."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

import matgl
from matgl.config import COULOMB_CONSTANT
from matgl.electrostatics._elec_pot import ElectrostaticPotential as ElectrostaticPotentialPyG
from matgl.utils.cutoff import polynomial_cutoff


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
