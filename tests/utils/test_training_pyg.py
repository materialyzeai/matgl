from __future__ import annotations

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Required for deterministic CUDA operations

import shutil
from functools import partial

# This function is used for M3GNet property dataset
import lightning as pl
import numpy as np
import pytest
import torch.backends.mps
from pymatgen.core import Lattice, Structure

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("Skipping PYG tests", allow_module_level=True)
from matgl.ext._pymatgen_pyg import Structure2Graph, get_element_list
from matgl.graph._data_pyg import MGLDataLoader, MGLDataset, collate_fn_pes, split_dataset
from matgl.models._qet_pyg import QET
from matgl.models._tensornet_pyg import TensorNet
from matgl.utils.training import (
    ModelLightningModule,
    PotentialLightningModule,
    xavier_init,
)

module_dir = os.path.dirname(os.path.abspath(__file__))

# The device can be chosen as "cpu" or "cuda". Note:"mps" is currently not available
device = "cpu"
torch.set_default_device(device)
torch.set_float32_matmul_precision("high")


class TestModelTrainer:
    def test_tensornet_training(self, LiFePO4, BaNiO3):
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
        two_body = Structure(Lattice.cubic(10.0), ["Li", "Li"], [[0, 0, 0], [0.2, 0, 0]])
        structures = [LiFePO4, BaNiO3] * 5 + [isolated_atom, two_body]
        energies = [-2.0, -3.0] * 5 + [-1.0, -1.5]
        forces = [np.zeros((len(s), 3)).tolist() for s in structures]
        stresses = [np.zeros((3, 3)).tolist()] * len(structures)
        element_types = get_element_list([LiFePO4, BaNiO3])
        converter = Structure2Graph(element_types=element_types, cutoff=5.0)
        dataset = MGLDataset(
            structures=structures,
            converter=converter,
            labels={"energies": energies, "forces": forces, "stresses": stresses},
            save_cache=False,
        )
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.8, 0.1, 0.1],
            shuffle=True,
            random_state=42,
        )
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            collate_fn=collate_fn_pes,
            batch_size=2,
            num_workers=0,
            generator=torch.Generator(device=device),
        )
        model = TensorNet(element_types=element_types, is_intensive=False, use_warp=False)
        lit_model = PotentialLightningModule(
            model=model, stress_weight=0.0001, loss="smooth_l1_loss", loss_params={"beta": 1.0}
        )
        # We will use CPU if MPS is available since there is a serious bug.
        trainer = pl.Trainer(max_epochs=10, accelerator=device, inference_mode=False)

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(lit_model, dataloaders=test_loader)

        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 10 epochs. This just tests that the energy is actually < 0.
        assert torch.allclose(pred_LFP_energy, torch.tensor([-2.8354]), atol=1e-4)
        assert torch.allclose(pred_BNO_energy, torch.tensor([-2.6534]), atol=1e-4)
        # specify customize optimizer and scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR

        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5, amsgrad=True)
        scheduler = CosineAnnealingLR(optimizer, T_max=1000 * 10, eta_min=1.0e-2 * 1.0e-3)
        lit_model = PotentialLightningModule(
            model=model,
            stress_weight=0.0001,
            loss="l1_loss",
            optimizer=optimizer,
            scheduler=scheduler,
        )
        trainer = pl.Trainer(max_epochs=10, accelerator=device, inference_mode=False)

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(lit_model, dataloaders=test_loader)

        pred_LFP_energy = model.predict_structure(LiFePO4)
        pred_BNO_energy = model.predict_structure(BaNiO3)

        # We are not expecting accuracy with 10 epochs. This just tests that the energy is actually < 0.
        assert torch.all(pred_LFP_energy < 0)
        assert torch.all(pred_BNO_energy < 0)

        self.teardown_class()

    def test_qet_training(self, LiFePO4, BaNiO3):
        torch.manual_seed(0)
        structures = [LiFePO4, BaNiO3] * 5
        energies = [-2.0, -3.0] * 5
        forces = [np.zeros((len(s), 3)).tolist() for s in structures]
        charges = [np.zeros(len(s)).tolist() for s in structures]
        stresses = [np.zeros((3, 3)).tolist()] * len(structures)
        element_types = get_element_list([LiFePO4, BaNiO3])
        converter = Structure2Graph(element_types=element_types, cutoff=5.0)
        dataset = MGLDataset(
            structures=structures,
            converter=converter,
            include_ref_charge=True,
            labels={"energies": energies, "forces": forces, "stresses": stresses, "charges": charges},
            save_cache=False,
        )
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.8, 0.1, 0.1],
            shuffle=True,
            random_state=42,
        )
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            collate_fn=partial(collate_fn_pes, include_charge=True),
            batch_size=2,
            num_workers=0,
            generator=torch.Generator(device=device),
        )
        model = QET(element_types=element_types, is_intensive=False, use_smooth=True, rbf_type="SphericalBessel")
        lit_model = PotentialLightningModule(
            model=model,
            stress_weight=0.0001,
            charge_weight=0.001,
            loss="smooth_l1_loss",
            loss_params={"beta": 1.0},
        )
        trainer = pl.Trainer(max_epochs=2, accelerator=device, inference_mode=False)

        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(lit_model, dataloaders=test_loader)

        pred_LFP_energy = model.predict_structure(LiFePO4, total_charge=torch.tensor([0.0]))
        pred_BNO_energy = model.predict_structure(BaNiO3, total_charge=torch.tensor([0.0]))

        # Loose check: energies should be finite after a 2-epoch run.
        assert torch.isfinite(pred_LFP_energy).all()
        assert torch.isfinite(pred_BNO_energy).all()

        self.teardown_class()

    @classmethod
    def teardown_class(cls):
        try:
            shutil.rmtree("lightning_logs")
        except FileNotFoundError:
            pass


def test_prediction_logger_train_and_val(LiFePO4, BaNiO3, tmp_path):
    """PredictionLogger captures per-epoch train + val preds in a stable per-sample order."""
    from matgl.utils.callbacks import PredictionLogger, add_sample_indices

    torch.manual_seed(0)
    structures = [LiFePO4, BaNiO3] * 4
    energies = [-2.0, -3.0] * 4
    forces = [np.zeros((len(s), 3)).tolist() for s in structures]
    stresses = [np.zeros((3, 3)).tolist()] * len(structures)
    element_types = get_element_list([LiFePO4, BaNiO3])
    converter = Structure2Graph(element_types=element_types, cutoff=5.0)
    dataset = MGLDataset(
        structures=structures,
        converter=converter,
        labels={"energies": energies, "forces": forces, "stresses": stresses},
        save_cache=False,
    )
    train_data, val_data, _test_data = split_dataset(
        dataset,
        frac_list=[0.5, 0.5, 0.0],
        shuffle=True,
        random_state=42,
    )
    add_sample_indices(train_data)
    add_sample_indices(val_data)

    train_loader, val_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        collate_fn=collate_fn_pes,
        batch_size=2,
        num_workers=0,
        generator=torch.Generator(device=device),
    )
    n_train = len(train_data)
    n_val = len(val_data)
    n_train_atoms = sum(train_data[i][0].num_nodes for i in range(n_train))
    n_val_atoms = sum(val_data[i][0].num_nodes for i in range(n_val))

    model = TensorNet(element_types=element_types, is_intensive=False, use_warp=False)
    lit_model = PotentialLightningModule(model=model, stress_weight=0.0, loss="mse_loss")
    log_path = tmp_path / "predictions.pt"
    logger_cb = PredictionLogger(save_path=log_path, log_train=True, log_validation=True)
    n_epochs = 3
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator=device,
        inference_mode=False,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        logger=False,
        callbacks=[logger_cb],
    )
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    log = logger_cb.predictions
    assert log["train_energy_preds"].shape == (n_epochs, n_train)
    assert log["train_energy_labels"].shape == (n_train,)
    assert log["train_force_preds"].shape == (n_epochs, n_train_atoms, 3)
    assert log["train_force_labels"].shape == (n_train_atoms, 3)
    assert log["val_energy_preds"].shape == (n_epochs, n_val)
    assert log["val_force_preds"].shape == (n_epochs, n_val_atoms, 3)
    # Errors are preds - labels.
    assert torch.allclose(
        log["train_energy_errors"],
        log["train_energy_preds"] - log["train_energy_labels"].unsqueeze(0),
    )
    # Ground-truth forces in the loaders were all zero.
    assert torch.allclose(log["train_force_labels"], torch.zeros(n_train_atoms, 3))

    assert log_path.exists()
    on_disk = torch.load(log_path, weights_only=True)
    assert torch.equal(on_disk["train_energy_preds"], log["train_energy_preds"])
    assert torch.equal(on_disk["val_energy_preds"], log["val_energy_preds"])

    try:
        shutil.rmtree("lightning_logs")
    except FileNotFoundError:
        pass


def test_prediction_logger_requires_indices(LiFePO4, BaNiO3):
    """PredictionLogger raises a clear error when add_sample_indices wasn't called."""
    from matgl.utils.callbacks import PredictionLogger

    torch.manual_seed(0)
    structures = [LiFePO4, BaNiO3] * 2
    energies = [-2.0, -3.0] * 2
    forces = [np.zeros((len(s), 3)).tolist() for s in structures]
    stresses = [np.zeros((3, 3)).tolist()] * len(structures)
    element_types = get_element_list([LiFePO4, BaNiO3])
    converter = Structure2Graph(element_types=element_types, cutoff=5.0)
    dataset = MGLDataset(
        structures=structures,
        converter=converter,
        labels={"energies": energies, "forces": forces, "stresses": stresses},
        save_cache=False,
    )
    train_data, val_data, _ = split_dataset(dataset, frac_list=[0.5, 0.5, 0.0], shuffle=True, random_state=42)
    train_loader, val_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        collate_fn=collate_fn_pes,
        batch_size=2,
        num_workers=0,
        generator=torch.Generator(device=device),
    )
    model = TensorNet(element_types=element_types, is_intensive=False, use_warp=False)
    lit_model = PotentialLightningModule(model=model, stress_weight=0.0, loss="mse_loss")
    logger_cb = PredictionLogger()
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator=device,
        inference_mode=False,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        logger=False,
        callbacks=[logger_cb],
    )
    with pytest.raises(RuntimeError, match="add_sample_indices"):
        trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    try:
        shutil.rmtree("lightning_logs")
    except FileNotFoundError:
        pass


def _make_efs_batch():
    """Tiny synthetic (preds, labels) tuple usable by PotentialLightningModule.loss_fn.

    Three structures with 2/3/4 atoms. We give torch.nn.Parameter forces so that
    grad-required inputs match what the real training loop produces.
    """
    energies = torch.tensor([-1.0, -2.0, -3.0])
    forces = torch.zeros(2 + 3 + 4, 3)
    stresses = torch.zeros(3, 3, 3)
    magmoms = torch.zeros(2 + 3 + 4, 1)
    num_atoms = torch.tensor([2, 3, 4])
    return energies, forces, stresses, magmoms, num_atoms


@pytest.mark.parametrize("loss_name", ["mse_loss", "huber_loss", "smooth_l1_loss", "l1_loss"])
def test_potential_lightning_module_loss_selection(loss_name):
    """Each supported loss-name string should resolve to a callable on `self.loss`."""
    model = TensorNet(use_warp=False)
    lit = PotentialLightningModule(model=model, loss=loss_name)
    assert callable(lit.loss)


def test_potential_lightning_loss_fn_no_num_atoms_branch():
    """num_atoms=None → loss_fn falls back to torch.ones_like(preds[0])."""
    model = TensorNet(use_warp=False)
    lit = PotentialLightningModule(model=model, stress_weight=0.0, magmom_weight=0.0)
    energies, forces, stresses, _, _ = _make_efs_batch()
    preds = (energies.clone(), forces.clone(), stresses.clone())
    labels = (energies, forces, stresses)
    out = lit.loss_fn(loss=lit.loss, labels=labels, preds=preds, num_atoms=None)
    assert "Total_Loss" in out
    # Identical preds and labels with num_atoms broadcast: total loss must be (≈) 0.
    assert torch.allclose(out["Total_Loss"], torch.tensor(0.0), atol=1e-6)


def test_potential_lightning_loss_fn_allow_missing_labels():
    """allow_missing_labels=True must drop NaN entries before computing the loss."""
    model = TensorNet(use_warp=False)
    lit = PotentialLightningModule(model=model, stress_weight=0.0, magmom_weight=0.0, allow_missing_labels=True)
    # Make the second structure's energy + matching forces NaN.
    energies, forces, stresses, _, num_atoms = _make_efs_batch()
    energies = energies.clone()
    energies[1] = float("nan")
    forces = forces.clone()
    forces[2:5] = float("nan")  # rows belonging to structure index 1
    stresses = stresses.clone()
    stresses[1] = float("nan")

    preds = (
        torch.tensor([-1.0, -2.0, -3.0]),
        torch.zeros_like(forces),
        torch.zeros_like(stresses),
    )
    labels = (energies, forces, stresses)
    out = lit.loss_fn(loss=lit.loss, labels=labels, preds=preds, num_atoms=num_atoms)
    # All survivors are exact matches → finite, near-zero loss.
    assert torch.isfinite(out["Total_Loss"])
    assert torch.allclose(out["Total_Loss"], torch.tensor(0.0), atol=1e-6)


@pytest.mark.parametrize("magmom_target", ["absolute", "symbreak"])
def test_potential_lightning_loss_fn_magmom_branches(magmom_target):
    """`magmom_weight > 0` enables calc_magmom and exercises both magmom_target branches."""
    model = TensorNet(use_warp=False)
    lit = PotentialLightningModule(
        model=model,
        stress_weight=0.0,
        magmom_weight=1.0,
        magmom_target=magmom_target,
    )
    assert lit.model.calc_magmom is True
    energies, forces, stresses, magmoms, num_atoms = _make_efs_batch()
    # Use signed magmoms so the symbreak min(loss(m), loss(-m)) path is non-trivial.
    magmom_labels = magmoms.clone()
    magmom_labels[0] = 0.5
    magmom_labels[1] = -0.5
    preds = (energies.clone(), forces.clone(), stresses.clone(), magmom_labels.clone())
    labels = (energies, forces, stresses, magmom_labels)
    out = lit.loss_fn(loss=lit.loss, labels=labels, preds=preds, num_atoms=num_atoms)
    assert "Magmom_MAE" in out
    assert "Magmom_RMSE" in out
    assert torch.isfinite(out["Total_Loss"])


def test_potential_lightning_on_load_checkpoint_fills_missing_keys():
    """on_load_checkpoint should populate keys that are missing from the checkpoint."""
    model = TensorNet(use_warp=False)
    lit = PotentialLightningModule(model=model)
    full = lit.state_dict()
    # Drop one key to simulate an upgrade-style checkpoint.
    dropped_key = next(iter(full))
    partial = {k: v for k, v in full.items() if k != dropped_key}
    checkpoint = {"state_dict": partial}
    lit.on_load_checkpoint(checkpoint)
    assert dropped_key in checkpoint["state_dict"]
    assert torch.allclose(checkpoint["state_dict"][dropped_key], full[dropped_key])


def test_model_lightning_module_init_loss_branches():
    """ModelLightningModule's __init__ should resolve each supported loss string."""
    model = TensorNet(use_warp=False, is_intensive=True, ntargets=1)
    for name in ("mse_loss", "huber_loss", "smooth_l1_loss", "anything_else"):
        lit = ModelLightningModule(model=model, loss=name, sync_dist=True)
        assert lit.model is model
        assert callable(lit.loss)
        # `sync_dist` is forwarded onto the instance so logging picks it up.
        assert lit.sync_dist is True


def test_model_lightning_module_loss_fn_scaling():
    """ModelLightningModule.loss_fn rescales preds via data_mean/data_std before comparing."""
    model = TensorNet(use_warp=False, is_intensive=True, ntargets=1)
    lit = ModelLightningModule(model=model, data_mean=2.0, data_std=3.0)
    preds = torch.tensor([[1.0], [2.0]])
    labels = torch.tensor([5.0, 8.0])  # 1*3+2=5, 2*3+2=8 → exact match
    out = lit.loss_fn(loss=lit.loss, labels=labels, preds=preds)
    assert "MAE" in out
    assert "RMSE" in out
    assert torch.allclose(out["Total_Loss"], torch.tensor(0.0), atol=1e-6)


def test_model_lightning_module_forward_and_step(LiFePO4, BaNiO3):
    """ModelLightningModule.forward should attach `pos`/`pbc_offshift` derived from
    ``frac_coords``/``pbc_offset`` and ``lat``, then return a model prediction; step
    should compose forward + loss_fn into a (results, batch_size) tuple."""
    torch.manual_seed(0)
    structures = [LiFePO4, BaNiO3]
    labels = {"energies": [-1.0, -2.0]}
    element_types = get_element_list(structures)
    converter = Structure2Graph(element_types=element_types, cutoff=5.0)
    dataset = MGLDataset(
        structures=structures,
        converter=converter,
        labels=labels,
        save_cache=False,
    )
    # Manually batch the dataset to avoid creating a Lightning Trainer.
    from matgl.graph._data_pyg import collate_fn_graph

    batch = collate_fn_graph([dataset[0], dataset[1]])
    g, lat, state_attr, _label_tensor = batch

    model = TensorNet(element_types=element_types, is_intensive=True, ntargets=1, use_warp=False)
    lit = ModelLightningModule(model=model, data_mean=0.0, data_std=1.0, sync_dist=False)

    preds = lit(g=g, lat=lat, state_attr=state_attr)
    # Two structures in the batch → preds should be a length-2 1-D tensor.
    assert preds.shape == (2,)
    assert torch.isfinite(preds).all()

    # forward must have populated ``g.pos`` and ``g.pbc_offshift`` from ``lat``.
    assert hasattr(g, "pos")
    assert hasattr(g, "pbc_offshift")
    assert g.pos.shape == (g.num_nodes, 3)
    assert g.pbc_offshift.shape == (g.edge_index.size(1), 3)

    # The full step path should also work and produce the expected logging keys.
    results, batch_size = lit.step(batch)
    assert {"Total_Loss", "MAE", "RMSE"}.issubset(results)
    assert batch_size == preds.numel()
    assert torch.isfinite(results["Total_Loss"])


@pytest.mark.parametrize("distribution", ["normal", "uniform", "fake"])
def test_xavier_init(distribution):
    model = TensorNet()
    # get a parameter
    w = model.linear.get_parameter("weight").clone()

    if distribution == "fake":
        with pytest.raises(ValueError, match=r"^Invalid distribution:."):
            xavier_init(model, distribution=distribution)
    else:
        xavier_init(model, distribution=distribution)
        print(w)
        assert not torch.allclose(w, model.linear.get_parameter("weight"))
        assert torch.allclose(torch.tensor(0.0), model.linear.get_parameter("bias"))
