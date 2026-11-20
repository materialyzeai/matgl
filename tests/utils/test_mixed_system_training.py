"""Joint molecular/material training and per-system conditioning."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from pymatgen.core import Lattice, Molecule, Structure
from torch_geometric.data import Batch

from matgl.apps.pes import Potential
from matgl.ext.ase import PESCalculator
from matgl.ext.pymatgen import Molecule2Graph, Structure2Graph, System2Graph
from matgl.graph.converters import get_system_metadata
from matgl.graph.data import MGLDiskDataset, collate_fn_pes, write_mgl_shards
from matgl.models import TensorNet
from matgl.models._qet import QET
from matgl.utils.training import MGLDatasetLoader, MGLPotentialTrainer, PotentialLightningModule


@pytest.fixture(autouse=True)
def reference_neighbors(monkeypatch):
    import matgl.ext.pymatgen as pmg

    monkeypatch.setattr(pmg, "_alchmtk_available", False)


def _model(cls=TensorNet):
    torch.manual_seed(0)
    kwargs = {"is_intensive": False} if cls is TensorNet else {}
    return cls(element_types=("H",), units=8, nblocks=1, num_rbf=8, cutoff=2.5, use_warp=False, **kwargs)


def _systems():
    return (
        Molecule(["H", "H"], [[0, 0, 0], [0.8, 0, 0]], charge=1, spin_multiplicity=2),
        Structure(Lattice.cubic(3), ["H", "H"], [[0, 0, 0], [0.32, 0.37, 0.41]]),
    )


def _frames():
    molecule = Atoms("H2", positions=[[0, 0, 0], [0.8, 0, 0]])
    molecule.info.update(total_charge=1, spin_multiplicity=2)
    crystal = Atoms("H2", positions=[[0, 0, 0], [0.96, 1.11, 1.23]], cell=[3, 3, 3], pbc=True)
    molecule.calc = SinglePointCalculator(molecule, energy=-1.0, forces=np.zeros((2, 3)))
    crystal.calc = SinglePointCalculator(crystal, energy=-2.0, forces=np.zeros((2, 3)), stress=np.zeros(6))
    return molecule, crystal


def test_extxyz_mixed_labels_and_metadata(tmp_path):
    path = tmp_path / "mixed.extxyz"
    write(path, list(_frames()), format="extxyz")
    dataset = MGLDatasetLoader.from_extxyz(path, element_types=("H",), root=str(tmp_path / "ds"))
    first, second = dataset[0], dataset[1]
    assert first[0].pbc.tolist() == [[False, False, False]]
    assert second[0].pbc.tolist() == [[True, True, True]]
    assert first[0].total_charge.item() == 1
    assert first[0].spin_multiplicity.item() == 2
    assert torch.isnan(first[-1]["stresses"]).all()
    batch = collate_fn_pes([first, second])
    assert batch[0].pbc.shape == (2, 3)
    assert batch[0].total_charge.tolist() == [1.0, 0.0]
    assert batch[5].shape == (6, 3)
    reloaded = MGLDatasetLoader.from_extxyz(path, element_types=("H",), root=str(tmp_path / "ds"))
    assert reloaded[0][0].total_charge.item() == 1

    # Conditioning changes invalidate caches even with identical geometry and targets.
    frames = list(_frames())
    frames[0].info.update(total_charge=0, spin_multiplicity=1)
    write(path, frames, format="extxyz")
    updated = MGLDatasetLoader.from_extxyz(path, element_types=("H",), root=str(tmp_path / "ds"))
    assert updated[0][0].total_charge.item() == 0
    assert updated[0][0].spin_multiplicity.item() == 1


def test_json_mixes_molecules_and_structures(tmp_path):
    molecule, crystal = _systems()
    path = tmp_path / "mixed.json"
    path.write_text(
        json.dumps(
            {
                "structures": [molecule.as_dict(), crystal.as_dict()],
                "outputs": {
                    "energies": [-1.0, -2.0],
                    "forces": [np.zeros((2, 3)).tolist()] * 2,
                    "stresses": [None, np.zeros((3, 3)).tolist()],
                },
            }
        )
    )
    dataset = MGLDatasetLoader.from_json(path, element_types=("H",), save_cache=False)
    assert len(dataset) == 2
    assert dataset[0][0].total_charge.item() == 1
    assert torch.isnan(dataset[0][-1]["stresses"]).all()


def test_molecular_stress_is_zero_and_cell_independent():
    molecule, crystal = _systems()
    gm, lm, _ = Molecule2Graph(("H",), cutoff=2.5).get_graph(molecule)
    gc, lc, _ = Structure2Graph(("H",), cutoff=2.5).get_graph(crystal)
    potential = Potential(_model(), calc_stresses=True).eval()
    graph = Batch.from_data_list([gm, gc])
    lattice = torch.cat([lm, lc])
    energy, forces, stress, _ = potential(graph, lattice)
    assert torch.equal(stress[:3], torch.zeros_like(stress[:3]))
    assert stress[3:].abs().sum() > 0
    expanded = graph.clone()
    expanded.frac_coords[:2] /= 10
    changed_cell = lattice.clone()
    changed_cell[0] *= 10
    e2, f2, s2, _ = potential(expanded, changed_cell)
    torch.testing.assert_close(energy, e2)
    torch.testing.assert_close(forces, f2)
    torch.testing.assert_close(stress, s2)
    # Adding a molecule to the batch must not change the crystal prediction.
    ec, fc, sc, _ = potential(gc, lc)
    torch.testing.assert_close(energy[1:], ec.reshape(-1))
    torch.testing.assert_close(forces[2:], fc)
    torch.testing.assert_close(stress[3:], sc)


@pytest.mark.parametrize("supervise_charge", [False, True])
@pytest.mark.parametrize("include_line_graph", [False, True])
def test_qet_net_charge_is_independent_of_charge_supervision(supervise_charge, include_line_graph):
    molecule, crystal = _systems()
    graphs, lattices = [], []
    for converter, system in [(Molecule2Graph(("H",), 2.5), molecule), (Structure2Graph(("H",), 2.5), crystal)]:
        graph, lattice, _ = converter.get_graph(system)
        graphs.append(graph)
        lattices.append(lattice)
    graph = Batch.from_data_list(graphs)
    model = _model(QET)
    lit = PotentialLightningModule(
        model, stress_weight=0, charge_weight=float(supervise_charge), include_line_graph=include_line_graph
    )
    lit(graph, torch.cat(lattices))
    charges = model.feature_dict["charge"]
    sums = torch.zeros(2).index_add(0, graph.batch, charges)
    torch.testing.assert_close(sums, torch.tensor([1.0, 0.0]), atol=1e-6, rtol=1e-6)
    # Explicit arguments take precedence over graph metadata.
    lit(graph, torch.cat(lattices), total_charge=torch.tensor([0.0, -1.0]))
    sums = torch.zeros(2).index_add(0, graph.batch, model.feature_dict["charge"])
    torch.testing.assert_close(sums, torch.tensor([0.0, -1.0]), atol=1e-6, rtol=1e-6)


def test_mixed_optimizer_step_masks_molecular_stress(tmp_path):
    path = tmp_path / "mixed.extxyz"
    write(path, list(_frames()), format="extxyz")
    ds = MGLDatasetLoader.from_extxyz(path, element_types=("H",), save_cache=False)
    # Even a finite molecular stress target must never supervise bulk stress.
    ds.labels["stresses"][0] = np.full((3, 3), 1e8).tolist()
    batch = collate_fn_pes([ds[0], ds[1]])
    lit = PotentialLightningModule(_model(), stress_weight=0.1, allow_missing_labels=True)
    optimizer = torch.optim.Adam(lit.parameters(), lr=1e-3)
    before = [param.detach().clone() for param in lit.parameters()]
    metrics, _ = lit.step(batch)
    assert torch.isfinite(metrics["Total_Loss"])
    assert metrics["Stress_MAE"] < 1e3
    assert lit._last_metric_counts["Stress_MAE"].item() == 9
    metrics["Total_Loss"].backward()
    optimizer.step()
    assert any(not torch.equal(old, param.detach()) for old, param in zip(before, lit.parameters(), strict=False))


@pytest.mark.parametrize("mixed", [False, True])
def test_high_level_training_resolves_stress_and_missing_labels(tmp_path, mixed):
    path = tmp_path / "training.extxyz"
    frames = list(_frames()) if mixed else [_frames()[0]]
    write(path, frames * 4, format="extxyz")
    ds = MGLDatasetLoader.from_extxyz(path, element_types=("H",), save_cache=False)
    trainer = MGLPotentialTrainer(
        _model(),
        max_epochs=1,
        batch_size=2,
        accelerator="cpu",
        devices=1,
        loader_kwargs={"num_workers": 0, "frac_list": [0.5, 0.25, 0.25], "shuffle": False},
        trainer_kwargs={
            "logger": False,
            "enable_checkpointing": False,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "limit_train_batches": 1,
            "limit_val_batches": 1,
            "limit_test_batches": 1,
            "num_sanity_val_steps": 0,
        },
    )
    trainer.fit(ds)
    assert trainer.trainer.global_step == 1
    assert trainer.potential.calc_stresses is mixed
    assert trainer.lit_module.allow_missing_labels


@pytest.mark.parametrize("supervise_charge", [False, True])
def test_qet_optimizer_with_missing_targets(tmp_path, supervise_charge):
    molecule, crystal = _frames()
    molecule.calc = SinglePointCalculator(molecule, forces=np.zeros((2, 3)))
    crystal.calc = SinglePointCalculator(crystal, energy=-2.0, stress=np.zeros(6), charges=np.zeros(2))
    path = tmp_path / "partial.extxyz"
    write(path, [molecule, crystal], format="extxyz")
    dataset = MGLDatasetLoader.from_extxyz(path, element_types=("H",), include_charges=True, save_cache=False)
    # Retaining optional labels does not require enabling their loss.
    batch = collate_fn_pes([dataset[0], dataset[1]], include_charge=True)
    module = PotentialLightningModule(
        _model(QET), stress_weight=0.1, charge_weight=float(supervise_charge), allow_missing_labels=True
    )
    metrics, _ = module.step(batch)
    assert all(torch.isfinite(value).all() for value in metrics.values())
    assert module._last_metric_counts["Energy_MAE"].item() == 1
    assert module._last_metric_counts["Force_MAE"].item() == 6
    assert module._last_metric_counts["Charge_MAE"].item() == (2 if supervise_charge else 0)
    charges = module.model.model.feature_dict["charge"]
    sums = torch.zeros(2).index_add(0, batch[0].batch, charges)
    torch.testing.assert_close(sums, torch.tensor([1.0, 0.0]), atol=1e-6, rtol=1e-6)
    metrics["Total_Loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)


@pytest.mark.parametrize("precompute", [False, True])
def test_disk_data_preserves_mixed_metadata_and_masks(tmp_path, precompute):
    records = [
        {
            "structure": system,
            "labels": {
                "energies": -1.0,
                "forces": np.zeros((2, 3)),
                "stresses": np.full((3, 3), np.nan) if index == 0 else np.zeros((3, 3)),
            },
        }
        for index, system in enumerate(_systems())
    ]
    converter = System2Graph(("H",), 2.5)
    write_mgl_shards(records, tmp_path, converter=converter, precomputed=precompute, shard_size=1)
    dataset = MGLDiskDataset(tmp_path, converter=None if precompute else converter)
    batch = collate_fn_pes([dataset[0], dataset[1]])
    assert batch[0].total_charge.tolist() == [1.0, 0.0]
    assert batch[0].pbc.tolist() == [[False] * 3, [True] * 3]
    module = PotentialLightningModule(_model(), stress_weight=0.1, allow_missing_labels=True)
    metrics, _ = module.step(batch)
    assert torch.isfinite(metrics["Total_Loss"])
    assert module._last_metric_counts["Stress_MAE"].item() == 9


def test_explicit_stress_weight_requires_periodic_targets(tmp_path):
    path = tmp_path / "molecule.extxyz"
    write(path, _frames()[0], format="extxyz")
    dataset = MGLDatasetLoader.from_extxyz(path, element_types=("H",), save_cache=False)
    with pytest.raises(ValueError, match="requires at least one periodic stress label"):
        MGLPotentialTrainer(_model(), stress_weight=0.1).fit(dataset)


@pytest.mark.parametrize("charge_output", [False, True])
def test_ase_qet_uses_conditioning_independently_of_charge_output(charge_output):
    atoms = _frames()[0]
    potential = Potential(_model(QET), calc_charge=charge_output, calc_stresses=False)
    atoms.calc = PESCalculator(potential, stress_unit="eV/A3")
    assert np.isfinite(atoms.get_potential_energy())
    torch.testing.assert_close(potential.model.feature_dict["charge"].sum(), torch.tensor(1.0))
    atoms.calc = PESCalculator(potential, total_charge=torch.tensor([-1.0]), stress_unit="eV/A3")
    assert np.isfinite(atoms.get_potential_energy())
    torch.testing.assert_close(potential.model.feature_dict["charge"].sum(), torch.tensor(-1.0))


def test_initial_moments_do_not_invent_spin_multiplicity():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0.8, 0, 0]])
    atoms.set_initial_magnetic_moments([0.2, 0.3])
    _, _, spin = get_system_metadata(atoms, is_atoms=True)
    assert np.isnan(spin)


@pytest.mark.parametrize("charge", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_conditioning_charge_is_rejected(charge):
    atoms = _frames()[0]
    atoms.info["total_charge"] = charge
    with pytest.raises(ValueError, match="total_charge must be finite"):
        get_system_metadata(atoms, is_atoms=True)


@pytest.mark.parametrize("spin", [0, -1, 1.5, float("inf"), -float("inf")])
def test_invalid_spin_multiplicity_is_rejected(spin):
    molecule = _systems()[0]
    molecule.properties["spin_multiplicity"] = spin
    with pytest.raises(ValueError, match="spin_multiplicity must be a positive integer"):
        get_system_metadata(molecule)


def test_system_converter_rejects_unsupported_input():
    with pytest.raises(TypeError, match="expects a pymatgen Structure or Molecule"):
        System2Graph(("H",)).get_graph(_frames()[0])


def test_periodic_stress_rejects_zero_volume():
    crystal = _systems()[1]
    graph, lattice, _ = Structure2Graph(("H",), cutoff=2.5).get_graph(crystal)
    lattice[0, 2, 2] = 0
    with pytest.raises(ValueError, match="Periodic stress requires a finite, nonzero cell volume"):
        Potential(_model(), calc_stresses=True).eval()(graph, lattice)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_periodic_stress_does_not_synchronize_on_cuda():
    # The cell-volume check runs on the device: a host synchronization in every stress call stalls
    # the pipeline and prevents capturing the stress path in a CUDA graph.
    graph, lattice, _ = Structure2Graph(("H",), cutoff=2.5).get_graph(_systems()[1])
    potential = Potential(_model(), calc_stresses=True).eval().cuda()
    graph, lattice = graph.to("cuda"), lattice.to("cuda")
    expected = potential(graph, lattice)  # the first call may fill caches from the host
    torch.cuda.set_sync_debug_mode("error")
    try:
        _, _, stresses, _ = potential(graph, lattice)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.allclose(stresses, expected[2])
    assert torch.isfinite(stresses).all()


@pytest.mark.parametrize("property_name", ["energies", "forces", "stresses"])
def test_collation_rejects_infinite_targets(property_name):
    graph, lattice, state = Molecule2Graph(("H",), cutoff=2.5).get_graph(_systems()[0])
    labels = {"energies": torch.tensor(-1.0), "forces": torch.zeros(2, 3), "stresses": torch.zeros(3, 3)}
    labels[property_name] = torch.full_like(labels[property_name], float("inf"))
    with pytest.raises(ValueError, match="PES labels cannot contain infinite values"):
        collate_fn_pes([(graph, lattice, torch.as_tensor(state), labels)])


def test_json_conditioning_overrides_and_voigt_stress(tmp_path):
    molecule = Molecule(["H", "H"], [[0, 0, 0], [0.8, 0, 0]])
    crystal = _systems()[1]
    samples = [
        {
            "structure": molecule.as_dict(),
            "total_charge": 1,
            "spin_multiplicity": 2,
            "energy": None,
            "forces": np.zeros((2, 3)).tolist(),
        },
        {"structure": crystal.as_dict(), "energy": -2.0, "forces": None, "stress": [1, 2, 3, 4, 5, 6]},
    ]
    path = tmp_path / "conditioned.json"
    path.write_text(json.dumps(samples))
    dataset = MGLDatasetLoader.from_json(path, element_types=("H",), stress_unit="GPa", save_cache=False)
    batch = collate_fn_pes([dataset[0], dataset[1]])
    assert dataset.graphs[0].total_charge.item() == 1
    assert dataset.graphs[0].spin_multiplicity.item() == 2
    assert torch.isnan(batch[3][0])
    assert torch.isnan(batch[4][2:]).all()
    assert torch.isnan(batch[5][:3]).all()
    torch.testing.assert_close(batch[5][3:], torch.tensor([[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]]))


def test_external_potential_changes_charges_without_changing_net_charge():
    graph, lattice, _ = Molecule2Graph(("H",), cutoff=2.5).get_graph(_systems()[0])
    potential = Potential(_model(QET), calc_charge=True, calc_stresses=False).eval()
    before = potential(graph, lattice)[4]
    # Conditioning values may come from a double-precision external solver.
    shifted = potential(graph, lattice, ext_pot=torch.tensor([-0.5, 0.5], dtype=torch.float64))
    assert shifted[4].dtype == potential.data_mean.dtype
    assert not torch.allclose(before, shifted[4])
    torch.testing.assert_close(shifted[4].sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-6)
    assert torch.isfinite(shifted[0]).all()
    assert torch.isfinite(shifted[1]).all()
