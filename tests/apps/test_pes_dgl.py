from __future__ import annotations

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

import matgl

if matgl.config.BACKEND != "DGL":
    pytest.skip("Skipping DGL tests", allow_module_level=True)
from matgl.apps._pes_dgl import Potential
from matgl.ext._pymatgen_dgl import Structure2Graph, get_element_list
from matgl.models import CHGNet, M3GNet, SO3Net, TensorNet


@pytest.fixture
def model():
    return M3GNet(element_types=["Mo", "S"], is_intensive=False, use_smooth=True)


@pytest.fixture
def model_so3net():
    return SO3Net(element_types=["Mo", "S"], is_intensive=False, lmax=2)


@pytest.fixture
def model_tensornet():
    return TensorNet(
        elment_types=["Mo", "S"], is_intensive=False, units=64, use_smooth=True, max_n=5, rbf_type="SphericalBessel"
    )


@pytest.fixture
def model_chgnet():
    return CHGNet(element_types=["Mo", "S"], is_intensive=False)


class TestPotential:
    def test_potential_efsh(self, graph_MoS, model, model_tensornet, model_so3net):
        structure, graph, state = graph_MoS
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        for m in [model, model_tensornet, model_so3net]:
            ff = Potential(model=m, calc_hessian=True)
            e, f, s, h = ff(graph, lat, state)
            assert [torch.numel(e)] == [1]
            assert [f.size(dim=0), f.size(dim=1)] == [2, 3]
            assert [s.size(dim=0), s.size(dim=1)] == [3, 3]
            assert [h.size(dim=0), h.size(dim=1)] == [6, 6]

    def test_potential_efshm(self, graph_MoS, model_chgnet):
        structure, graph, state = graph_MoS
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        ff = Potential(model=model_chgnet, calc_hessian=True, calc_magmom=True)
        e, f, s, h, m = ff(graph, lat, state)
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0), f.size(dim=1)] == [2, 3]
        assert [s.size(dim=0), s.size(dim=1)] == [3, 3]
        assert [h.size(dim=0), h.size(dim=1)] == [6, 6]
        assert [m.size(dim=0), m.size(dim=1)] == [2, 1]

    def test_potential_efs(self, graph_MoS, model):
        structure, graph, state = graph_MoS
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        ff = Potential(model=model)
        e, f, s, h = ff(graph, lat, state)
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0), f.size(dim=1)] == [2, 3]
        assert [s.size(dim=0), s.size(dim=1)] == [3, 3]
        assert [h.size(dim=0)] == [1]

    def test_potential_ef(self, graph_MoS, model):
        structure, graph, state = graph_MoS
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        ff = Potential(model=model, calc_stresses=False)
        e, f, s, h = ff(graph, lat, state)
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0), f.size(dim=1)] == [2, 3]
        assert [s.size(dim=0)] == [1]
        assert [h.size(dim=0)] == [1]

    def test_potential_e(self, graph_MoS, model):
        structure, graph, state = graph_MoS
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        ff = Potential(model=model, calc_forces=False, calc_stresses=False)
        e, f, s, h = ff(graph, lat, state)
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0)] == [1]
        assert [s.size(dim=0)] == [1]
        assert [h.size(dim=0)] == [1]

    def test_including_repulsion(self, graph_MoS, model):
        structure, graph, state = graph_MoS
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        ff = Potential(model=model, calc_forces=True, calc_stresses=True, calc_hessian=True, calc_repuls=True)
        e, f, s, h = ff(graph, lat, state)
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0), f.size(dim=1)] == [2, 3]
        assert [s.size(dim=0), s.size(dim=1)] == [3, 3]
        assert [h.size(dim=0), h.size(dim=1)] == [6, 6]

    def test_potential_two_body(self, model):
        structure = Structure(Lattice.cubic(10.0), ["Mo", "Mo"], [[0.0, 0, 0], [0.2, 0.0, 0.0]])
        element_types = get_element_list([structure])
        p2g = Structure2Graph(element_types=element_types, cutoff=5.0)
        graph, lat, state = p2g.get_graph(structure)
        ff = Potential(model=model, calc_hessian=True)
        e, f, s, h = ff(graph, lat, torch.tensor(state))
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0), f.size(dim=1)] == [2, 3]
        assert [s.size(dim=0), s.size(dim=1)] == [3, 3]
        assert [h.size(dim=0), h.size(dim=1)] == [6, 6]

    def test_potential_isolated_atom(self, model):
        structure = Structure(Lattice.cubic(10.0), ["Mo"], [[0.0, 0, 0]])
        element_types = get_element_list([structure])
        p2g = Structure2Graph(element_types=element_types, cutoff=5.0)
        graph, lat, state = p2g.get_graph(structure)
        ff = Potential(model=model, calc_hessian=True)
        e, f, s, h = ff(graph, lat, torch.tensor(state))
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0), f.size(dim=1)] == [1, 3]
        assert [s.size(dim=0), s.size(dim=1)] == [3, 3]
        assert [h.size(dim=0), h.size(dim=1)] == [3, 3]

    def test_potential_efsh_with_force_fd(self, graph_MoS, model):
        p2g = Structure2Graph(element_types=("O", "Zr"), cutoff=5.0)
        struct_minus = Structure.from_spacegroup(
            "Pm-3m", Lattice.cubic(5.0), ["Zr", "O"], [[0.0, 0, 0], [0.498, 0.5, 0.5]]
        )
        g_minus, lat_minus, state = p2g.get_graph(struct_minus)
        struct_zero = Structure.from_spacegroup("Pm-3m", Lattice.cubic(5.0), ["Zr", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        g_zero, lat_zero, state = p2g.get_graph(struct_zero)
        struct_plus = Structure.from_spacegroup(
            "Pm-3m", Lattice.cubic(5.0), ["Zr", "O"], [[0, 0, 0], [0.502, 0.5, 0.5]]
        )
        g_plus, lat_plus, state = p2g.get_graph(struct_plus)
        ff = Potential(model=model, calc_hessian=True, debug_mode=True)
        e_minus, _, _ = ff(g_minus, lat_minus, state)
        _, grad_dx_zero, _ = ff(g_zero, lat_zero, state)
        e_plus, _, _ = ff(g_plus, lat_plus, state)
        fd = (e_plus - e_minus) / (2 * 0.01)
        assert np.allclose(fd.detach().numpy(), grad_dx_zero[0][0].detach().numpy(), atol=1e-05)

    def test_potential_with_stress_fd(self, graph_MoS, model):
        p2g = Structure2Graph(element_types=("O", "Zr"), cutoff=5.0)
        struct_minus = Structure.from_spacegroup(
            "Pm-3m", Lattice.cubic(5.99), ["Zr", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        )
        g_minus, lat_minus, state = p2g.get_graph(struct_minus)
        struct_zero = Structure.from_spacegroup("Pm-3m", Lattice.cubic(6.0), ["Zr", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        g_zero, lat_zero, state = p2g.get_graph(struct_zero)
        struct_plus = Structure.from_spacegroup("Pm-3m", Lattice.cubic(6.01), ["Zr", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        g_plus, lat_plus, state = p2g.get_graph(struct_plus)
        ff = Potential(model=model, calc_hessian=True, debug_mode=True)
        e_minus, _, _ = ff(g_minus, lat_minus, state)
        _, _, grad_dl_zero = ff(g_zero, lat_zero, state)
        e_plus, _, _ = ff(g_plus, lat_plus, state)
        fd = (e_plus - e_minus) / (2 * 0.01)
        assert np.allclose(fd.detach().numpy(), grad_dl_zero[0][0][0].detach().numpy(), atol=1e-05)
