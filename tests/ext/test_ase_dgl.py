from __future__ import annotations

import os.path

import numpy as np
import pytest
import torch
from ase.build import molecule
from pymatgen.io.ase import AseAtomsAdaptor

import matgl
from matgl import load_model

if matgl.config.BACKEND != "DGL":
    pytest.skip("Skipping DGL tests", allow_module_level=True)
from matgl.ext.ase import Atoms2Graph, M3GNetCalculator, MolecularDynamics, PESCalculator, Relaxer


@pytest.mark.integration
def test_PESCalculator_and_M3GNetCalculator(MoS):
    adaptor = AseAtomsAdaptor()

    # ============================================================
    # M3GNet PES (eV/A3)
    # ============================================================
    s_ase = adaptor.get_atoms(MoS)  # type: ignore
    ff = load_model("pretrained_models/M3GNet-MP-2021.2.8-PES/")
    ff.calc_hessian = True

    calc = PESCalculator(
        potential=ff,
        state_attr=None,
        stress_unit="eV/A3",
        stress_weight=1.0,
    )
    s_ase.set_calculator(calc)

    assert isinstance(s_ase.get_potential_energy(), float)
    assert list(s_ase.get_forces().shape) == [2, 3]
    assert list(s_ase.get_stress().shape) == [6]
    assert list(calc.results["hessian"].shape) == [6, 6]

    np.testing.assert_allclose(
        s_ase.get_potential_energy(),
        -10.824362,
        atol=1e-5,
        rtol=1e-6,
    )

    # ============================================================
    # M3GNet PES (default GPa)
    # ============================================================
    calc = PESCalculator(
        potential=ff,
        state_attr=torch.tensor([0.0, 0.0]),
        stress_unit="GPa",
        stress_weight=1.0,
    )
    s_ase.set_calculator(calc)

    assert list(s_ase.get_forces().shape) == [2, 3]
    assert list(s_ase.get_stress().shape) == [6]
    assert list(calc.results["hessian"].shape) == [6, 6]

    np.testing.assert_allclose(
        s_ase.get_potential_energy(),
        -10.824362,
        atol=1e-5,
        rtol=1e-6,
    )

    # ============================================================
    # Invalid stress_unit
    # ============================================================
    with pytest.raises(
        ValueError,
        match=r"Unsupported stress_unit: Pa. Must be 'GPa' or 'eV/A3'.",
    ):
        PESCalculator(
            potential=ff,
            stress_unit="Pa",
            stress_weight=1.0,
        )

    # ============================================================
    # Invalid stress_weight
    # ============================================================
    with pytest.raises(
        ValueError,
        match=r"Invalid stress unit configuration",
    ):
        PESCalculator(
            potential=ff,
            stress_unit="GPa",
            stress_weight=0.5,
        )

    # ============================================================
    # M3GNetCalculator (backward compatibility)
    # ============================================================
    calc = M3GNetCalculator(potential=ff)
    s_ase.set_calculator(calc)

    assert list(s_ase.get_forces().shape) == [2, 3]
    assert list(s_ase.get_stress().shape) == [6]
    assert list(calc.results["hessian"].shape) == [6, 6]

    np.testing.assert_allclose(
        s_ase.get_potential_energy(),
        -10.824362,
        atol=1e-5,
        rtol=1e-6,
    )

    # ============================================================
    # QET PES (charges + stress)
    # ============================================================
    s_ase = adaptor.get_atoms(MoS)  # type: ignore
    ff = matgl.load_model("Materialyze/QET-PES-MatQ")
    ff.calc_hessian = True

    calc = PESCalculator(
        potential=ff,
        state_attr=None,
        stress_unit="eV/A3",
        stress_weight=1.0,
    )
    s_ase.set_calculator(calc)

    assert list(s_ase.get_charges().shape) == [2]
    assert list(s_ase.get_stress().shape) == [6]
    assert list(calc.results["hessian"].shape) == [6, 6]

    np.testing.assert_allclose(
        s_ase.get_potential_energy(),
        -10.798001,
        atol=1e-5,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        s_ase.get_charges(),
        np.array([0.690892, -0.690892]),
        atol=1e-5,
        rtol=1e-6,
    )


def test_CHGNetCalculator(MoS):
    adaptor = AseAtomsAdaptor()
    s_ase = adaptor.get_atoms(MoS)  # type: ignore
    ff = load_model("pretrained_models/CHGNet-MPtrj-2023.12.1-2.7M-PES/")
    ff.calc_hessian = True
    calc = PESCalculator(potential=ff)
    s_ase.set_calculator(calc)
    assert isinstance(s_ase.get_potential_energy(), float)
    assert list(s_ase.get_forces().shape) == [2, 3]
    assert list(s_ase.get_stress().shape) == [6]
    assert list(s_ase.get_magnetic_moments().shape) == [2, 1]
    assert list(calc.results["hessian"].shape) == [6, 6]
    np.testing.assert_allclose(s_ase.get_potential_energy(), -10.983373, atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(s_ase.get_magnetic_moments(), [[2.386847], [0.124443]], atol=1e-5, rtol=1e-6)


def test_PESCalculator_mol(AcAla3NHMe):
    adaptor = AseAtomsAdaptor()
    mol = adaptor.get_atoms(AcAla3NHMe)
    ff = load_model("pretrained_models/M3GNet-MP-2021.2.8-PES/")
    calc = PESCalculator(potential=ff)
    mol.set_calculator(calc)
    assert isinstance(mol.get_potential_energy(), float)
    assert list(mol.get_forces().shape) == [42, 3]
    np.testing.assert_allclose(mol.get_potential_energy(), -250.16803)


def test_Relaxer(MoS):
    pot = load_model("pretrained_models/M3GNet-MP-2021.2.8-PES/")
    r = Relaxer(pot)
    results = r.relax(MoS, traj_file="MoS_relax.traj")
    s = results["final_structure"]
    traj = results["trajectory"].as_pandas()
    assert s.lattice.a < 3.5
    assert traj["energies"].iloc[-1] < traj["energies"].iloc[0]
    for t in results["trajectory"]:
        assert len(t) == 5
    assert os.path.exists("MoS_relax.traj")
    os.remove("MoS_relax.traj")


def test_get_graph_from_atoms(LiFePO4):
    adaptor = AseAtomsAdaptor()
    structure_ase = adaptor.get_atoms(LiFePO4)
    a2g = Atoms2Graph(element_types=["Li", "Fe", "P", "O"], cutoff=4.0)
    graph, _, state = a2g.get_graph(structure_ase)
    # check the number of nodes
    assert np.allclose(graph.num_nodes(), len(structure_ase.get_atomic_numbers()))
    # check the atomic feature of atom 0
    assert np.allclose(graph.ndata["node_type"].detach().numpy()[0], 0)
    # check the atomic feature of atom 4
    assert np.allclose(graph.ndata["node_type"].detach().numpy()[4], 1)
    # check the number of bonds
    assert np.allclose(graph.num_edges(), 704)
    # check the state features
    assert np.allclose(state, [0.0, 0.0])


def test_get_graph_from_atoms_mol():
    mol = molecule("CH4")
    a2g = Atoms2Graph(element_types=["H", "C"], cutoff=4.0)
    graph, _, state = a2g.get_graph(mol)
    # check the number of nodes
    assert np.allclose(graph.num_nodes(), len(mol.get_atomic_numbers()))
    # check the atomic feature of atom 0
    assert np.allclose(graph.ndata["node_type"].detach().numpy()[0], 1)
    # check the atomic feature of atom 4
    assert np.allclose(graph.ndata["node_type"].detach().numpy()[1], 0)
    # check the number of bonds
    assert np.allclose(graph.num_edges(), 20)
    # check the state features
    assert np.allclose(state, [0.0, 0.0])


def test_molecular_dynamics(MoS2):
    pot = load_model("pretrained_models/M3GNet-MP-2021.2.8-PES/")
    for ensemble in [
        "nvt",
        "nve",
        "nvt_langevin",
        "nvt_andersen",
        "nvt_bussi",
        "nvt_nose_hoover_chain",
        "npt",
        "npt_berendsen",
        "npt_nose_hoover",
        "npt_nose_hoover_chain",
    ]:
        md = MolecularDynamics(MoS2, potential=pot, ensemble=ensemble, taut=0.1, taup=0.1, compressibility_au=10)
        md.run(10)
        assert md.dyn is not None
        md.set_atoms(MoS2)
    md = MolecularDynamics(MoS2, potential=pot, ensemble=ensemble, taut=None, taup=None, compressibility_au=10)
    md.run(10)
    with pytest.raises(ValueError, match="Ensemble not supported"):
        MolecularDynamics(MoS2, potential=pot, ensemble="notanensemble")
