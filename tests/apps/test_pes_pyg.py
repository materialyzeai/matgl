from __future__ import annotations

import numpy as np
import pytest
import torch

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("Skipping PYG tests", allow_module_level=True)
from pymatgen.core import Lattice, Structure
from torch_geometric.data import Batch

import matgl
from matgl.apps._pes_pyg import Potential
from matgl.ext._pymatgen_pyg import Structure2Graph, get_element_list
from matgl.models._tensornet_pyg import TensorNet


@pytest.fixture
def model_tensornet():
    return TensorNet(
        elment_types=["Mo", "S"], is_intensive=False, units=64, use_smooth=True, max_n=5, rbf_type="SphericalBessel"
    )


class TestPotential:
    def test_potential_efsh(self, graph_MoS_pyg, model_tensornet):
        structure, graph, state = graph_MoS_pyg
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        for m in [model_tensornet]:
            ff = Potential(model=m, calc_hessian=True)
            e, f, s, h = ff(graph, lat, state)
            assert [torch.numel(e)] == [1]
            assert [f.size(dim=0), f.size(dim=1)] == [2, 3]
            assert [s.size(dim=0), s.size(dim=1)] == [3, 3]
            assert [h.size(dim=0), h.size(dim=1)] == [6, 6]

    def test_potential_efsh_batch(self, graph_MoS_pyg, model_tensornet):
        # graph_MoS_pyg_list: list of (structure, graph, state) tuples
        structure, graph, _ = graph_MoS_pyg
        graphs = [graph, graph]
        structures = [structure, structure]
        # Stack lattice matrices into (B, 3, 3)
        lat = torch.stack([torch.tensor(s.lattice.matrix, dtype=matgl.float_th) for s in structures], dim=0)

        # Batch graphs using PyG utility
        batched_graph = Batch.from_data_list(graphs)

        # Stack state attrs (if present), otherwise use None
        state_attr = None

        ff = Potential(model=model_tensornet, calc_hessian=True)

        # Forward pass
        e, f, s, h = ff(batched_graph, lat, state_attr)

        batch_size = len(structures)

        assert e.shape == (batch_size,)
        assert f.shape == (batched_graph.num_nodes, 3)
        assert s.shape == (batch_size * 3, 3)
        assert h.shape == (batched_graph.num_nodes * 3, batched_graph.num_nodes * 3)

    def test_potential_efs(self, graph_MoS_pyg, model_tensornet):
        structure, graph, state = graph_MoS_pyg
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        ff = Potential(model=model_tensornet)
        e, f, s, h = ff(graph, lat, state)
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0), f.size(dim=1)] == [2, 3]
        assert [s.size(dim=0), s.size(dim=1)] == [3, 3]
        assert [h.size(dim=0)] == [1]

    def test_potential_ef(self, graph_MoS_pyg, model_tensornet):
        structure, graph, state = graph_MoS_pyg
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        ff = Potential(model=model_tensornet, calc_stresses=False)
        e, f, s, h = ff(graph, lat, state)
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0), f.size(dim=1)] == [2, 3]
        assert [s.size(dim=0)] == [1]
        assert [h.size(dim=0)] == [1]

    def test_potential_e(self, graph_MoS_pyg, model_tensornet):
        structure, graph, state = graph_MoS_pyg
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        ff = Potential(model=model_tensornet, calc_forces=False, calc_stresses=False)
        e, f, s, h = ff(graph, lat, state)
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0)] == [1]
        assert [s.size(dim=0)] == [1]
        assert [h.size(dim=0)] == [1]

    def test_including_repulsion(self, graph_MoS_pyg, model_tensornet):
        structure, graph, state = graph_MoS_pyg
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        ff = Potential(model=model_tensornet, calc_forces=True, calc_stresses=True, calc_hessian=True, calc_repuls=True)
        e, f, s, h = ff(graph, lat, state)
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0), f.size(dim=1)] == [2, 3]
        assert [s.size(dim=0), s.size(dim=1)] == [3, 3]
        assert [h.size(dim=0), h.size(dim=1)] == [6, 6]

    def test_potential_two_body(self, model_tensornet):
        structure = Structure(Lattice.cubic(10.0), ["Mo", "Mo"], [[0.0, 0, 0], [0.2, 0.0, 0.0]])
        element_types = get_element_list([structure])
        p2g = Structure2Graph(element_types=element_types, cutoff=5.0)
        graph, lat, state = p2g.get_graph(structure)
        ff = Potential(model=model_tensornet, calc_hessian=True)
        e, f, s, h = ff(graph, lat, torch.tensor(state))
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0), f.size(dim=1)] == [2, 3]
        assert [s.size(dim=0), s.size(dim=1)] == [3, 3]
        assert [h.size(dim=0), h.size(dim=1)] == [6, 6]

    def test_potential_isolated_atom(self, model_tensornet):
        structure = Structure(Lattice.cubic(10.0), ["Mo"], [[0.0, 0, 0]])
        element_types = get_element_list([structure])
        p2g = Structure2Graph(element_types=element_types, cutoff=5.0)
        graph, lat, state = p2g.get_graph(structure)
        ff = Potential(model=model_tensornet, calc_hessian=True)
        e, f, s, h = ff(graph, lat, torch.tensor(state))
        assert [torch.numel(e)] == [1]
        assert [f.size(dim=0), f.size(dim=1)] == [1, 3]
        assert [s.size(dim=0), s.size(dim=1)] == [3, 3]
        assert [h.size(dim=0), h.size(dim=1)] == [3, 3]

    def test_potential_efsh_with_force_fd(self, graph_MoS_pyg, model_tensornet):
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
        ff = Potential(model=model_tensornet, calc_hessian=True, debug_mode=True)
        e_minus, _, _ = ff(g_minus, lat_minus, state)
        _, grad_dx_zero, _ = ff(g_zero, lat_zero, state)
        e_plus, _, _ = ff(g_plus, lat_plus, state)
        fd = (e_plus - e_minus) / (2 * 0.01)
        assert np.allclose(fd.detach().numpy(), grad_dx_zero[0][0].detach().numpy(), atol=1e-05)

    def test_forward_does_not_mutate_input_graph(self, graph_MoS_pyg, model_tensornet):
        """``Potential.forward`` must shallow-clone ``g`` so the caller's graph keeps
        its original attribute set and ``frac_coords``/``edge_index`` tensors are
        not mutated. Regression test: previously ``lattice``/``pbc_offshift``/``pos``
        leaked back onto the input graph."""
        structure, graph, state = graph_MoS_pyg
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)
        fields_before = set(graph.keys())
        ff = Potential(model=model_tensornet)
        ff(graph, lat, state)
        fields_after = set(graph.keys())
        assert fields_before == fields_after, (
            f"Potential.forward mutated input graph; added: {fields_after - fields_before}"
        )

    def test_potential_with_stress_fd(self, graph_MoS, model_tensornet):
        p2g = Structure2Graph(element_types=("O", "Zr"), cutoff=5.0)
        struct_minus = Structure.from_spacegroup(
            "Pm-3m", Lattice.cubic(5.99), ["Zr", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        )
        g_minus, lat_minus, state = p2g.get_graph(struct_minus)
        struct_zero = Structure.from_spacegroup("Pm-3m", Lattice.cubic(6.0), ["Zr", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        g_zero, lat_zero, state = p2g.get_graph(struct_zero)
        struct_plus = Structure.from_spacegroup("Pm-3m", Lattice.cubic(6.01), ["Zr", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        g_plus, lat_plus, state = p2g.get_graph(struct_plus)
        ff = Potential(model=model_tensornet, calc_hessian=True, debug_mode=True)
        e_minus, _, _ = ff(g_minus, lat_minus, state)
        _, _, grad_dl_zero = ff(g_zero, lat_zero, state)
        e_plus, _, _ = ff(g_plus, lat_plus, state)
        fd = (e_plus - e_minus) / (2 * 0.01)
        assert np.allclose(fd.detach().numpy(), grad_dl_zero[0][0][0].detach().numpy(), atol=1e-05)
