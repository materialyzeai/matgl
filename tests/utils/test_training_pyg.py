from __future__ import annotations

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Required for deterministic CUDA operations

import shutil

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
from matgl.models._tensornet_pyg import TensorNet
from matgl.utils._training_pyg import PotentialLightningModule, xavier_init

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
        model = TensorNet(element_types=element_types, is_intensive=False)
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
        assert torch.allclose(pred_LFP_energy, torch.tensor([-2.0512]), atol=1e-4)
        assert torch.allclose(pred_BNO_energy, torch.tensor([-3.2459]), atol=1e-4)
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
        assert torch.allclose(pred_LFP_energy, torch.tensor([-2.0237]), atol=1e-4)
        assert torch.allclose(pred_BNO_energy, torch.tensor([-3.2062]), atol=1e-4)

        self.teardown_class()

    @classmethod
    def teardown_class(cls):
        try:
            shutil.rmtree("lightning_logs")
        except FileNotFoundError:
            pass


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
