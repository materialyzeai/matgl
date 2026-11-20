from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.build import bulk

ts = pytest.importorskip("torch_sim", reason="torch-sim optional dependency required")

from torch_sim.models.interface import validate_model_outputs  # noqa: E402

from matgl.apps.pes import Potential  # noqa: E402
from matgl.ext.ase import PESCalculator  # noqa: E402
from matgl.ext.torch_sim import MatGLModel  # noqa: E402
from matgl.models import QET, M3GNet, TensorNet  # noqa: E402


def _potential(*, stresses: bool = True) -> Potential:
    torch.manual_seed(7)
    model = TensorNet(
        element_types=("Si",),
        units=8,
        nblocks=1,
        num_rbf=8,
        cutoff=3.0,
        is_intensive=False,
        use_warp=False,
    )
    return Potential(
        model,
        data_mean=0.3,
        data_std=1.2,
        element_refs=torch.tensor([-0.2]),
        calc_forces=True,
        calc_stresses=stresses,
    )


@pytest.mark.parametrize("geometry", ["cubic", "skew", "slab"])
def test_torch_sim_batch_matches_ase_energy_forces_and_stress(geometry):
    potential = _potential()
    atoms = [bulk("Si", "diamond", a=5.43, cubic=True), bulk("Si", "diamond", a=5.2, cubic=True)]
    if geometry == "skew":
        for structure in atoms:
            structure.set_cell([[5.4, 0.4, 0.2], [0.1, 5.4, 0.3], [-0.2, 0.15, 5.2]], scale_atoms=True)
    elif geometry == "slab":
        for structure in atoms:
            structure.pbc = [True, True, False]
    expected = []
    for structure in atoms:
        reference = structure.copy()
        reference.calc = PESCalculator(potential, stress_unit="eV/A3", use_voigt=False)
        expected.append(
            (
                reference.get_potential_energy(),
                reference.get_forces(),
                reference.calc.results["stress"],
            )
        )

    model = MatGLModel(potential)
    state = ts.initialize_state(atoms, model.device, model.dtype)
    original_positions = state.positions.clone()
    original_cell = state.cell.clone()
    output = model(state)

    torch.testing.assert_close(output["energy"], torch.tensor([item[0] for item in expected]), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        output["forces"],
        torch.as_tensor(np.concatenate([item[1] for item in expected])),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        output["stress"],
        torch.as_tensor(np.stack([item[2] for item in expected])),
        atol=2e-5,
        rtol=2e-5,
    )
    assert all(not value.requires_grad for value in output.values())
    torch.testing.assert_close(state.positions, original_positions)
    torch.testing.assert_close(state.cell, original_cell)


def test_torch_sim_static_runner_and_isolated_atom():
    model = MatGLModel(_potential(stresses=False), compute_stress=False)
    isolated = Atoms("Si", positions=[[0.0, 0.0, 0.0]])

    properties = ts.static(system=isolated, model=model)

    assert len(properties) == 1
    assert torch.isfinite(properties[0]["potential_energy"])
    torch.testing.assert_close(properties[0]["forces"], torch.zeros((1, 3)), atol=1e-7, rtol=0)


@pytest.mark.parametrize("magmoms", [False, True])
@pytest.mark.parametrize("charge_output", [False, True])
def test_torch_sim_qet_returns_charge_constrained_per_system(magmoms, charge_output):
    torch.manual_seed(5)
    qet = QET(
        element_types=("Li", "F"),
        units=8,
        nblocks=1,
        num_rbf=8,
        cutoff=5.0,
        is_intensive=False,
        use_warp=False,
        include_magmom=magmoms,
    )
    model = MatGLModel(
        Potential(qet, calc_forces=True, calc_stresses=False, calc_charge=charge_output, calc_magmom=magmoms),
        compute_stress=False,
    )
    atoms = [
        Atoms("LiF", positions=[[0, 0, 0], [2, 0, 0]], cell=[8, 8, 8], pbc=True),
        Atoms("LiF", positions=[[0, 0, 0], [2.2, 0, 0]], cell=[8, 8, 8], pbc=True),
    ]
    state = ts.initialize_state(atoms, model.device, model.dtype)
    state.system_extras["total_charge"] = torch.tensor([0.0, 1.0])

    output = model(state)

    charges = qet.feature_dict["charge"]
    if charge_output:
        torch.testing.assert_close(output["partial_charges"], charges)
    else:
        assert "partial_charges" not in output
    charge_sums = torch.zeros(2).scatter_add_(0, state.system_idx, charges)
    torch.testing.assert_close(charge_sums, state.total_charge, atol=1e-6, rtol=1e-6)
    if magmoms:
        assert output["magnetic_moments"].shape == (state.n_atoms,)
        assert torch.isfinite(output["magnetic_moments"]).all()


def test_torch_sim_qet_force_matches_finite_difference():
    torch.manual_seed(13)
    qet = QET(
        element_types=("Li", "F"),
        units=8,
        nblocks=1,
        num_rbf=8,
        cutoff=5.0,
        is_intensive=False,
        use_warp=False,
    )
    model = MatGLModel(
        Potential(qet, calc_forces=True, calc_stresses=False, calc_charge=True),
        dtype=torch.float64,
        compute_stress=False,
    )
    atoms = Atoms("LiF", positions=[[0, 0, 0], [2.0, 0.3, 0]], cell=[8, 8, 8], pbc=True)
    total_charge = torch.tensor([1.0], dtype=model.dtype, device=model.device)

    def evaluate(displacement: float) -> dict[str, torch.Tensor]:
        displaced = atoms.copy()
        displaced.positions[1, 0] += displacement
        state = ts.initialize_state(displaced, model.device, model.dtype)
        state.system_extras["total_charge"] = total_charge
        return model(state)

    step = 1e-4
    output = evaluate(0.0)
    energy_plus = evaluate(step)["energy"][0]
    energy_minus = evaluate(-step)["energy"][0]
    finite_difference_force = -(energy_plus - energy_minus) / (2 * step)

    torch.testing.assert_close(output["forces"][1, 0], finite_difference_force, atol=2e-7, rtol=2e-5)
    torch.testing.assert_close(output["partial_charges"].sum(), total_charge[0], atol=1e-10, rtol=1e-10)


def test_torch_sim_m3gnet_three_body_smoke():
    torch.manual_seed(3)
    energy_model = M3GNet(
        element_types=("Si",),
        dim_node_embedding=8,
        dim_edge_embedding=8,
        units=8,
        max_n=2,
        max_l=2,
        nblocks=1,
        cutoff=3.0,
        threebody_cutoff=2.8,
        is_intensive=False,
    )
    model = MatGLModel(Potential(energy_model, calc_stresses=False), compute_stress=False)
    state = ts.initialize_state(bulk("Si", "diamond", a=5.43), model.device, model.dtype)

    output = model(state)

    assert output["energy"].shape == (1,)
    assert output["forces"].shape == (2, 3)
    assert all(torch.isfinite(value).all() for value in output.values())


def test_torch_sim_model_validates_configuration_and_state():
    intensive = TensorNet(element_types=("Si",), units=8, nblocks=1, num_rbf=8, is_intensive=True, use_warp=False)
    with pytest.raises(ValueError, match="extensive total-energy"):
        MatGLModel(Potential(intensive))

    model = MatGLModel(_potential(), compute_stress=True)
    molecule = ts.initialize_state(Atoms("Si", positions=[[0, 0, 0]]), model.device, model.dtype)
    torch.testing.assert_close(model(molecule)["stress"], torch.zeros((1, 3, 3)))

    wrong_element = ts.initialize_state(
        Atoms("C", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True), model.device, model.dtype
    )
    with pytest.raises(ValueError, match="unsupported atomic numbers"):
        model(wrong_element)

    periodic = ts.initialize_state(
        Atoms("Si", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True), model.device, torch.float64
    )
    with pytest.raises(TypeError, match="dtype"):
        model(periodic)


def test_torch_sim_force_and_stress_switches_stay_synchronized():
    model = MatGLModel(_potential(), compute_forces=False, compute_stress=True)
    assert model.potential.calc_forces
    assert model.potential.calc_stresses

    model.compute_stress = False
    assert not model.potential.calc_forces
    assert not model.potential.calc_stresses

    model.compute_forces = True
    assert model.potential.calc_forces


def test_torch_sim_official_model_interface_validation():
    torch.manual_seed(11)
    energy_model = TensorNet(
        element_types=("H", "C", "Mg", "Si", "Fe"),
        units=8,
        nblocks=1,
        num_rbf=8,
        cutoff=4.0,
        is_intensive=False,
        use_warp=False,
    )
    model = MatGLModel(Potential(energy_model))

    validate_model_outputs(model, device=torch.device("cpu"), dtype=torch.float32)


@pytest.mark.parametrize("context", [torch.no_grad, torch.inference_mode])
def test_torch_sim_autograd_inside_simulation_contexts(context):
    model = MatGLModel(_potential())
    state = ts.initialize_state(bulk("Si", "diamond", a=5.43), model.device, model.dtype)
    expected = model(state)
    with context():
        actual = model(state)
    for key, value in expected.items():
        torch.testing.assert_close(actual[key], value)


def test_torch_sim_continuous_state_follows_model_dtype():
    energy_model = M3GNet(
        element_types=("Si",),
        dim_node_embedding=8,
        dim_edge_embedding=8,
        units=8,
        max_n=2,
        max_l=2,
        nblocks=1,
        cutoff=3.0,
        threebody_cutoff=2.8,
        is_intensive=False,
        include_state=True,
        dim_state_feats=2,
    )
    attrs = torch.tensor([[0.3, 0.5]], dtype=torch.float64)
    model = MatGLModel(Potential(energy_model, calc_stresses=False), compute_stress=False, state_attr=attrs)
    for dtype in (torch.float32, torch.float64):
        model.dtype = dtype
        state = ts.initialize_state(bulk("Si", "diamond", a=5.43), model.device, dtype)
        output = model(state)
        expected = model(state, state_attr=attrs.to(dtype=dtype))
        for key, value in output.items():
            assert torch.isfinite(value).all()
            torch.testing.assert_close(value, expected[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_torch_sim_cuda_matches_cpu():
    atoms = bulk("Si", "diamond", a=5.43)
    model = MatGLModel(_potential())
    expected = model(ts.initialize_state(atoms, model.device, model.dtype))
    model.device = torch.device("cuda")
    output = model(ts.initialize_state(atoms, model.device, model.dtype))
    for key, value in expected.items():
        torch.testing.assert_close(output[key].cpu(), value, atol=2e-5, rtol=2e-5)
