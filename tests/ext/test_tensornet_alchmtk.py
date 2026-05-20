# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Tests for TensorNetWrapper (nvalchemi-toolkit integration)."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("nvalchemi", reason="nvalchemi-toolkit required for alchmtk tests")

import numpy as np
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import HookContext, NeighborListHook
from pymatgen.core import Element

import matgl
from matgl.apps._pes import Potential
from matgl.ext.alchmtk import TensorNetWrapper
from matgl.graph._compute import compute_pair_vector_and_distance
from matgl.models._tensornet import TensorNet

# 1 eV/A^3 = 160.21766208 GPa
EV_A3_TO_GPA = 160.21766208


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def model_tensornet():
    """Same convention as test_pes_pyg — small TensorNet for MoS."""
    return TensorNet(
        element_types=("Mo", "S"),
        is_intensive=False,
        units=64,
        use_smooth=True,
        max_n=5,
        rbf_type="SphericalBessel",
        use_warp=False,
    )


@pytest.fixture
def model_tensornet_ch4():
    """Small TensorNet for CH4 molecule tests."""
    return TensorNet(
        element_types=("C", "H"),
        is_intensive=False,
        units=32,
        nblocks=1,
        cutoff=2.0,
        use_warp=False,
    )


@pytest.fixture
def potential_ch4(model_tensornet_ch4):
    return Potential(
        model=model_tensornet_ch4,
        data_mean=torch.tensor(0.0),
        data_std=torch.tensor(1.0),
        calc_forces=True,
        calc_stresses=False,
    )


@pytest.fixture
def potential(model_tensornet):
    return Potential(
        model=model_tensornet,
        data_mean=torch.tensor(0.0),
        data_std=torch.tensor(1.0),
        calc_forces=True,
        calc_stresses=True,
    )


@pytest.fixture
def potential_with_refs(model_tensornet):
    refs = torch.tensor([-1.5, -2.3], dtype=torch.float32)  # Mo, S
    return Potential(
        model=model_tensornet,
        data_mean=torch.tensor(0.0),
        data_std=torch.tensor(1.0),
        element_refs=refs,
        calc_forces=True,
        calc_stresses=True,
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _matgl_graph_to_atomic_data(graph, lattice, element_types, is_periodic=True):
    """Convert a matgl PyG graph into nvalchemi AtomicData.

    Uses the exact same neighbor_list and pbc_offset from the matgl graph
    so any numerical difference is from the wrapper, not the neighbor list.
    """
    atomic_numbers = torch.tensor(
        [Element(element_types[t]).Z for t in graph.node_type.tolist()],
        dtype=torch.int64,
    )
    lat = lattice[0] if lattice.dim() == 3 else lattice
    positions = graph.frac_coords @ lat

    kwargs = {
        "atomic_numbers": atomic_numbers,
        "positions": positions,
        "neighbor_list": graph.edge_index.long().T,  # matgl [2, E] -> nvalchemi [E, 2]
        "neighbor_list_shifts": graph.pbc_offset.float(),
    }
    if is_periodic:
        kwargs["cell"] = lat.unsqueeze(0)
        kwargs["pbc"] = torch.tensor([[True, True, True]])

    return AtomicData(**kwargs)


def _set_active_outputs(wrapper, outputs):
    """Set active outputs on wrapper's model_config."""
    wrapper.model_config.active_outputs = set(outputs)


def _run_nl_hook(wrapper, batch):
    """Run NeighborListHook on a batch using HookContext."""
    nl_hook = NeighborListHook(
        wrapper.model_config.neighbor_config,
        stage=DynamicsStage.BEFORE_COMPUTE,
    )
    ctx = HookContext(batch=batch, step_count=0)
    nl_hook(ctx, DynamicsStage.BEFORE_COMPUTE)


# ------------------------------------------------------------------
# Construction validation
# ------------------------------------------------------------------


class TestConstruction:
    def test_is_intensive_rejected(self):
        model = TensorNet(
            element_types=("Mo", "S"),
            is_intensive=True,
            units=32,
            nblocks=1,
            use_warp=False,
        )
        with pytest.raises(ValueError, match="is_intensive=False"):
            TensorNetWrapper(model=model)

    def test_from_potential(self, potential):
        wrapper = TensorNetWrapper.from_potential(potential)
        assert wrapper.model is potential.model
        assert torch.equal(wrapper.data_mean, potential.data_mean)
        assert torch.equal(wrapper.data_std, potential.data_std)

    def test_from_potential_wrong_model_type(self):
        dummy = Potential(model=torch.nn.Linear(10, 1), calc_forces=False, calc_stresses=False)
        with pytest.raises(TypeError, match="TensorNet"):
            TensorNetWrapper.from_potential(dummy)

    def test_from_potential_with_zbl(self):
        model = TensorNet(
            element_types=("Mo", "S"),
            is_intensive=False,
            units=32,
            nblocks=1,
            use_warp=False,
        )
        pot = Potential(model=model, calc_repuls=True)
        wrapper = TensorNetWrapper.from_potential(pot)
        assert wrapper.repuls is not None

    def test_from_potential_with_element_refs(self, potential_with_refs):
        wrapper = TensorNetWrapper.from_potential(potential_with_refs)
        assert wrapper._element_ref_offset is not None
        assert torch.equal(
            wrapper._element_ref_offset,
            potential_with_refs.element_refs.property_offset,
        )

    def test_model_config(self, potential):
        wrapper = TensorNetWrapper.from_potential(potential)
        cfg = wrapper.model_config
        assert "energy" in cfg.outputs
        assert "forces" in cfg.outputs
        assert "stress" in cfg.outputs
        assert "forces" in cfg.autograd_outputs
        assert "stress" in cfg.autograd_outputs
        assert cfg.supports_pbc is True
        assert cfg.needs_pbc is False
        assert cfg.neighbor_config is not None
        assert cfg.neighbor_config.cutoff == potential.model.cutoff

    def test_embedding_shapes(self, potential):
        wrapper = TensorNetWrapper.from_potential(potential)
        shapes = wrapper.embedding_shapes
        assert shapes["node_embeddings"] == (64,)
        assert shapes["graph_embeddings"] == (64,)


# ------------------------------------------------------------------
# Neighbor list consistency
# ------------------------------------------------------------------


class TestNeighborListConsistency:
    def test_same_edges(self, graph_MoS):
        """AtomicData uses the exact same edges as the matgl graph."""
        structure, graph, _ = graph_MoS
        element_types = ("Mo", "S")
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)

        data = _matgl_graph_to_atomic_data(graph, lat.unsqueeze(0), element_types)
        batch = Batch.from_data_list([data])

        # Batch stores [E, 2]; matgl stores [2, E]. Transpose to compare.
        assert torch.equal(batch.neighbor_list.T.long(), graph.edge_index.long())

    def test_same_shifts(self, graph_MoS):
        """neighbor_list_shifts match pbc_offset from matgl graph."""
        structure, graph, _ = graph_MoS
        element_types = ("Mo", "S")
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)

        data = _matgl_graph_to_atomic_data(graph, lat.unsqueeze(0), element_types)
        batch = Batch.from_data_list([data])

        assert torch.allclose(batch.neighbor_list_shifts.float(), graph.pbc_offset.float(), atol=1e-6)


# ------------------------------------------------------------------
# Numerical correctness: periodic crystal (MoS from conftest)
# ------------------------------------------------------------------


class TestNumericalCorrectnessPeriodic:
    def test_energy_matches(self, potential, graph_MoS):
        structure, graph, state = graph_MoS
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)

        e_matgl, _, _, _ = potential(graph, lat, state)

        wrapper = TensorNetWrapper.from_potential(potential)
        _set_active_outputs(wrapper, {"energy"})
        data = _matgl_graph_to_atomic_data(graph, lat.unsqueeze(0), ("Mo", "S"))
        outputs = wrapper(Batch.from_data_list([data]))

        assert torch.allclose(outputs["energy"].squeeze(), e_matgl.squeeze(), atol=1e-5)

    def test_forces_match(self, potential, graph_MoS):
        structure, graph, state = graph_MoS
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)

        _, f_matgl, _, _ = potential(graph, lat, state)

        wrapper = TensorNetWrapper.from_potential(potential)
        _set_active_outputs(wrapper, {"energy", "forces"})
        data = _matgl_graph_to_atomic_data(graph, lat.unsqueeze(0), ("Mo", "S"))
        outputs = wrapper(Batch.from_data_list([data]))

        assert torch.allclose(outputs["forces"], f_matgl, atol=1e-5)

    def test_stresses_match(self, potential, graph_MoS):
        structure, graph, state = graph_MoS
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)

        _, _, s_matgl, _ = potential(graph, lat, state)

        wrapper = TensorNetWrapper.from_potential(potential)
        _set_active_outputs(wrapper, {"energy", "forces", "stress"})
        data = _matgl_graph_to_atomic_data(graph, lat.unsqueeze(0), ("Mo", "S"))
        outputs = wrapper(Batch.from_data_list([data]))

        # wrapper: Cauchy convention -dE/d(strain)/V in eV/A^3
        # matgl: +dE/d(strain)/V * 160.2 in GPa
        # So: -wrapper_stress * GPa_factor == matgl_stress
        s_wrapper_gpa = -outputs["stress"].squeeze() * EV_A3_TO_GPA
        assert torch.allclose(s_wrapper_gpa, s_matgl.squeeze(), atol=1e-4)


# ------------------------------------------------------------------
# Numerical correctness: molecule (CH4 from conftest)
# ------------------------------------------------------------------


class TestNumericalCorrectnessMolecule:
    def test_molecule_energy_matches(self, potential_ch4, graph_CH4):
        """Non-periodic CH4: energy and forces match matgl Potential."""
        _, graph, state = graph_CH4
        element_types = ("C", "H")

        # matgl path — identity lattice for molecules
        lat = torch.eye(3, dtype=matgl.float_th)
        e_matgl, f_matgl, _, _ = potential_ch4(graph, lat, state)

        # wrapper path
        wrapper = TensorNetWrapper.from_potential(potential_ch4)
        _set_active_outputs(wrapper, {"energy", "forces"})

        atomic_numbers = torch.tensor(
            [Element(element_types[t]).Z for t in graph.node_type.tolist()],
            dtype=torch.int64,
        )
        positions = graph.frac_coords.float()

        data = AtomicData(
            atomic_numbers=atomic_numbers,
            positions=positions,
            neighbor_list=graph.edge_index.long().T,
            neighbor_list_shifts=graph.pbc_offset.float(),
        )
        outputs = wrapper(Batch.from_data_list([data]))

        assert torch.allclose(outputs["energy"].squeeze(), e_matgl.squeeze(), atol=1e-5)
        assert torch.allclose(outputs["forces"], f_matgl, atol=1e-5)


# ------------------------------------------------------------------
# Element refs
# ------------------------------------------------------------------


class TestElementRefs:
    def test_energy_offset(self, potential, potential_with_refs, graph_MoS):
        """element_refs shifts energy by the expected per-element sum."""
        structure, graph, _ = graph_MoS
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)

        wrapper_no = TensorNetWrapper.from_potential(potential)
        _set_active_outputs(wrapper_no, {"energy"})

        wrapper_yes = TensorNetWrapper.from_potential(potential_with_refs)
        _set_active_outputs(wrapper_yes, {"energy"})

        data = _matgl_graph_to_atomic_data(graph, lat.unsqueeze(0), ("Mo", "S"))
        batch = Batch.from_data_list([data])

        e_no = wrapper_no(batch)["energy"]
        e_yes = wrapper_yes(batch)["energy"]

        # MoS: 1 Mo (ref=-1.5) + 1 S (ref=-2.3) = -3.8
        expected_offset = -1.5 + -2.3
        assert pytest.approx((e_yes - e_no).item(), abs=1e-5) == expected_offset

    def test_forces_unchanged(self, potential, potential_with_refs, graph_MoS):
        """element_refs are position-independent, so forces are identical."""
        structure, graph, _ = graph_MoS
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)

        wrapper_no = TensorNetWrapper.from_potential(potential)
        _set_active_outputs(wrapper_no, {"energy", "forces"})

        wrapper_yes = TensorNetWrapper.from_potential(potential_with_refs)
        _set_active_outputs(wrapper_yes, {"energy", "forces"})

        data = _matgl_graph_to_atomic_data(graph, lat.unsqueeze(0), ("Mo", "S"))
        batch = Batch.from_data_list([data])

        f_no = wrapper_no(batch)["forces"]
        f_yes = wrapper_yes(batch)["forces"]

        assert torch.allclose(f_no, f_yes, atol=1e-6)


# ------------------------------------------------------------------
# ZBL nuclear repulsion
# ------------------------------------------------------------------


class TestZBL:
    def test_zbl_adds_repulsive_energy(self, model_tensornet, graph_MoS):
        """ZBL adds a positive repulsive energy contribution."""
        structure, graph, _ = graph_MoS
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)

        wrapper_no_zbl = TensorNetWrapper(model=model_tensornet)
        _set_active_outputs(wrapper_no_zbl, {"energy"})

        wrapper_with_zbl = TensorNetWrapper(model=model_tensornet, calc_repuls=True)
        _set_active_outputs(wrapper_with_zbl, {"energy"})

        data = _matgl_graph_to_atomic_data(graph, lat.unsqueeze(0), ("Mo", "S"))
        batch = Batch.from_data_list([data])

        e_no = wrapper_no_zbl(batch)["energy"]
        e_yes = wrapper_with_zbl(batch)["energy"]

        # ZBL is purely repulsive — adds positive energy
        assert (e_yes - e_no).item() > 0

    def test_zbl_matches_matgl(self, model_tensornet, graph_MoS):
        """ZBL energy matches matgl Potential with calc_repuls=True."""
        structure, graph, state = graph_MoS
        lat = torch.tensor(structure.lattice.matrix, dtype=matgl.float_th)

        pot = Potential(
            model=model_tensornet,
            calc_repuls=True,
            calc_forces=False,
            calc_stresses=False,
        )
        e_matgl, _, _, _ = pot(graph, lat, state)

        wrapper = TensorNetWrapper.from_potential(pot)
        _set_active_outputs(wrapper, {"energy"})
        data = _matgl_graph_to_atomic_data(graph, lat.unsqueeze(0), ("Mo", "S"))
        outputs = wrapper(Batch.from_data_list([data]))

        assert torch.allclose(outputs["energy"].squeeze(), e_matgl.squeeze(), atol=1e-5)


# ------------------------------------------------------------------
# End-to-end: neighbor list built independently by both paths
# ------------------------------------------------------------------


class TestEndToEnd:
    def test_neighborlist_distances_match(self, potential, MoS):
        """NeighborListHook and matgl's Structure2Graph produce the same
        pairwise distances (order may differ)."""
        from matgl.ext._pymatgen import Structure2Graph

        cutoff = float(potential.model.cutoff)
        element_types = ("Mo", "S")

        # matgl path: Structure2Graph -> compute distances
        converter = Structure2Graph(element_types=element_types, cutoff=cutoff)
        graph, lattice, _ = converter.get_graph(MoS)
        graph.pbc_offshift = torch.matmul(graph.pbc_offset, lattice[0])
        graph.pos = graph.frac_coords @ lattice[0]
        _, dist_matgl = compute_pair_vector_and_distance(
            graph.pos,
            graph.edge_index,
            graph.pbc_offshift,
        )

        # nvalchemi path: AtomicData -> NeighborListHook -> compute distances
        wrapper = TensorNetWrapper.from_potential(potential)
        atomic_numbers = torch.tensor(
            [Element(element_types[t]).Z for t in graph.node_type.tolist()],
            dtype=torch.int64,
        )
        positions = (graph.frac_coords @ lattice[0]).float()
        cell = lattice[0].float().unsqueeze(0)

        data = AtomicData(
            atomic_numbers=atomic_numbers,
            positions=positions,
            cell=cell,
            pbc=torch.tensor([[True, True, True]]),
        )
        batch = Batch.from_data_list([data])

        _run_nl_hook(wrapper, batch)

        # Compute distances from nvalchemi's neighbor list
        inputs = wrapper.adapt_input(batch)
        _, dist_nval = compute_pair_vector_and_distance(
            inputs["pos"],
            inputs["edge_index"],
            inputs["pbc_offshift"],
        )

        # Same set of distances (order may differ)
        np.testing.assert_array_almost_equal(
            np.sort(dist_matgl.detach().cpu().numpy()),
            np.sort(dist_nval.detach().cpu().numpy()),
            decimal=4,
        )

    def test_energy_forces_end_to_end(self, potential, MoS):
        """Full pipeline: AtomicData.from_structure -> NeighborListHook -> wrapper
        vs matgl's Potential. Energies and forces should match."""
        from matgl.ext._pymatgen import Structure2Graph

        cutoff = float(potential.model.cutoff)
        element_types = ("Mo", "S")

        # matgl path
        converter = Structure2Graph(element_types=element_types, cutoff=cutoff)
        graph, _, state = converter.get_graph(MoS)
        lat = torch.tensor(MoS.lattice.matrix, dtype=matgl.float_th)
        e_matgl, f_matgl, _, _ = potential(graph, lat, state)

        # nvalchemi path
        wrapper = TensorNetWrapper.from_potential(potential)
        _set_active_outputs(wrapper, {"energy", "forces"})

        data = AtomicData.from_structure(MoS)
        batch = Batch.from_data_list([data])

        _run_nl_hook(wrapper, batch)

        outputs = wrapper(batch)

        assert torch.allclose(
            outputs["energy"].squeeze(),
            e_matgl.squeeze(),
            atol=1e-4,
        )
        assert torch.allclose(outputs["forces"], f_matgl, atol=1e-4)


# ------------------------------------------------------------------
# Pretrained model: TensorNet-MatPES-PBE-v2025.1-PES
# ------------------------------------------------------------------


class TestPretrained:
    def test_pretrained_MoS_energy(self, MoS):
        """Pretrained TensorNet on MoS with NL built via NeighborListHook."""
        pot = matgl.load_model("TensorNet-PES-MatPES-PBE-2025.2")
        wrapper = TensorNetWrapper.from_potential(pot)
        _set_active_outputs(wrapper, {"energy", "forces"})

        data = AtomicData.from_structure(MoS)
        batch = Batch.from_data_list([data])

        _run_nl_hook(wrapper, batch)

        outputs = wrapper(batch)

        # Reference from test_ase_pyg.py: -10.4884214
        assert outputs["energy"].squeeze().item() == pytest.approx(
            -10.452658,
            abs=1e-3,
        )
        assert outputs["forces"].shape == (2, 3)

    def test_pretrained_molecule_energy(self, AcAla3NHMe):
        """Pretrained TensorNet on AcAla3NHMe molecule with NL built via NeighborListHook."""
        pot = matgl.load_model("TensorNet-PES-MatPES-PBE-2025.2")
        wrapper = TensorNetWrapper.from_potential(pot)
        _set_active_outputs(wrapper, {"energy", "forces"})

        data = AtomicData.from_structure(AcAla3NHMe)
        batch = Batch.from_data_list([data])

        _run_nl_hook(wrapper, batch)

        outputs = wrapper(batch)

        # Reference from test_ase_pyg.py: -247.286789
        assert outputs["energy"].squeeze().item() == pytest.approx(
            -249.194916,
            abs=1e-3,
        )
        assert outputs["forces"].shape == (42, 3)
