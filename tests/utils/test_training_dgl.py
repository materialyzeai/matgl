from __future__ import annotations

import os
import shutil

# This function is used for M3GNet property dataset
from functools import partial

import lightning as pl
import numpy as np
import pytest

import matgl

if matgl.config.BACKEND != "DGL":
    pytest.skip("Skipping DGL tests", allow_module_level=True)
import torch.backends.mps
from dgl.data.utils import split_dataset
from pymatgen.core import Lattice, Structure

from matgl.ext._pymatgen_dgl import Structure2Graph, get_element_list
from matgl.graph._data_dgl import MGLDataLoader, MGLDataset, collate_fn_graph, collate_fn_pes
from matgl.models import CHGNet, M3GNet, MEGNet, SO3Net, TensorNet
from matgl.models._qet_dgl import QET
from matgl.utils.training import ModelLightningModule, PotentialLightningModule, xavier_init

module_dir = os.path.dirname(os.path.abspath(__file__))

# The device can be chosen as "cpu" or "cuda". Note:"mps" is currently not available
device = "cpu"
torch.set_default_device(device)
torch.set_float32_matmul_precision("high")


def teardown():
    try:
        shutil.rmtree("lightning_logs")
    except FileNotFoundError:
        pass


def test_megnet_training(LiFePO4, BaNiO3):
    isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
    structures = [LiFePO4] * 5 + [BaNiO3] * 5 + [isolated_atom]
    label = [-2] * 5 + [-3] * 5 + [-1]  # Artificial dataset.
    element_types = get_element_list([LiFePO4, BaNiO3])
    cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
    dataset = MGLDataset(
        structures=structures,
        converter=cry_graph,
        labels={"label": label},
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
        collate_fn=collate_fn_graph,
        batch_size=2,
        num_workers=0,
        generator=torch.Generator(device=device),
    )

    model = MEGNet(
        dim_node_embedding=16,
        dim_edge_embedding=100,
        dim_state_embedding=2,
        nblocks=3,
        hidden_layer_sizes_input=(64, 32),
        hidden_layer_sizes_conv=(64, 64, 32),
        nlayers_set2set=1,
        niters_set2set=3,
        hidden_layer_sizes_outputput=[32, 16],
        is_classification=False,
    )
    xavier_init(model)
    lit_model = ModelLightningModule(model=model)
    # We will use CPU if MPS is available since there is a serious bug.
    trainer = pl.Trainer(max_epochs=10, accelerator=device)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    results = trainer.test(dataloaders=test_loader)
    assert "test_Total_Loss" in results[0]
    model = model.to(torch.device(device))
    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
    assert pred_LFP_energy < 0
    assert pred_BNO_energy < 0

    teardown()


def test_m3gnet_training(LiFePO4, BaNiO3):
    isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
    two_body = Structure(Lattice.cubic(10.0), ["Li", "Li"], [[0, 0, 0], [0.2, 0, 0]])
    structures = [LiFePO4, BaNiO3] * 5 + [isolated_atom, two_body]
    energies = [-2.0, -3.0] * 5 + [-1.0, -1.5]
    forces = [np.zeros((len(s), 3)).tolist() for s in structures]
    stresses = [np.zeros((3, 3)).tolist()] * len(structures)
    element_types = get_element_list([LiFePO4, BaNiO3])
    converter = Structure2Graph(element_types=element_types, cutoff=5.0)
    dataset = MGLDataset(
        threebody_cutoff=4.0,
        structures=structures,
        converter=converter,
        include_line_graph=True,
        labels={"energies": energies, "forces": forces, "stresses": stresses},
        save_cache=False,
    )
    train_data, val_data, test_data = split_dataset(
        dataset,
        frac_list=[0.8, 0.1, 0.1],
        shuffle=True,
        random_state=42,
    )
    my_collate_fn = partial(collate_fn_pes, include_line_graph=True)
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=my_collate_fn,
        batch_size=2,
        num_workers=0,
        generator=torch.Generator(device=device),
    )
    model = M3GNet(element_types=element_types, is_intensive=False)
    lit_model = PotentialLightningModule(model=model, stress_weight=0.0001, include_line_graph=True)
    # We will use CPU if MPS is available since there is a serious bug.
    trainer = pl.Trainer(max_epochs=2, accelerator=device, inference_mode=False)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lit_model, dataloaders=test_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
    assert pred_LFP_energy < 0
    assert pred_BNO_energy < 0
    # specify customize optimizer and scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5, amsgrad=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000 * 10, eta_min=1.0e-2 * 1.0e-3)
    lit_model = PotentialLightningModule(
        model=model,
        stress_weight=0.0001,
        include_line_graph=True,
        loss="l1_loss",
        optimizer=optimizer,
        scheduler=scheduler,
    )
    trainer = pl.Trainer(max_epochs=2, accelerator=device, inference_mode=False)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lit_model, dataloaders=test_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
    assert pred_LFP_energy < 0
    assert pred_BNO_energy < 0

    teardown()


def test_so3net_training(LiFePO4, BaNiO3):
    isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
    two_body = Structure(Lattice.cubic(10.0), ["Li", "Li"], [[0, 0, 0], [0.2, 0, 0]])
    structures = [LiFePO4, BaNiO3] * 5 + [isolated_atom, two_body]
    energies = [-2.0, -3.0] * 5 + [-1.0, -1.5]
    forces = [np.zeros((len(s), 3)).tolist() for s in structures]
    stresses = [np.zeros((3, 3)).tolist()] * len(structures)
    element_types = get_element_list([LiFePO4, BaNiO3])
    converter = Structure2Graph(element_types=element_types, cutoff=5.0)
    dataset = MGLDataset(
        threebody_cutoff=4.0,
        structures=structures,
        converter=converter,
        include_line_graph=False,
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
    model = SO3Net(element_types=element_types, lmax=2, is_intensive=False)
    lit_model = PotentialLightningModule(
        model=model, stress_weight=0.0001, loss="huber_loss", loss_params={"delta": 1.0}
    )
    # We will use CPU if MPS is available since there is a serious bug.
    trainer = pl.Trainer(max_epochs=2, accelerator=device, inference_mode=False)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lit_model, dataloaders=test_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
    assert pred_LFP_energy < 0
    assert pred_BNO_energy < 0
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
    trainer = pl.Trainer(max_epochs=2, accelerator=device, inference_mode=False)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lit_model, dataloaders=test_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
    assert pred_LFP_energy < 0
    assert pred_BNO_energy < 0

    teardown()


def test_tensornet_training(LiFePO4, BaNiO3):
    isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
    two_body = Structure(Lattice.cubic(10.0), ["Li", "Li"], [[0, 0, 0], [0.2, 0, 0]])
    structures = [LiFePO4, BaNiO3] * 5 + [isolated_atom, two_body]
    energies = [-2.0, -3.0] * 5 + [-1.0, -1.5]
    forces = [np.zeros((len(s), 3)).tolist() for s in structures]
    stresses = [np.zeros((3, 3)).tolist()] * len(structures)
    element_types = get_element_list([LiFePO4, BaNiO3])
    converter = Structure2Graph(element_types=element_types, cutoff=5.0)
    dataset = MGLDataset(
        threebody_cutoff=4.0,
        structures=structures,
        converter=converter,
        include_line_graph=False,
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
    trainer = pl.Trainer(max_epochs=2, accelerator=device, inference_mode=False)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lit_model, dataloaders=test_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
    assert pred_LFP_energy < 0
    assert pred_BNO_energy < 0
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
    trainer = pl.Trainer(max_epochs=2, accelerator=device, inference_mode=False)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lit_model, dataloaders=test_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
    assert pred_LFP_energy < 0
    assert pred_BNO_energy < 0

    teardown()


def test_m3gnet_training_without_stress(LiFePO4, BaNiO3):
    isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
    two_body = Structure(Lattice.cubic(10.0), ["Li", "Li"], [[0, 0, 0], [0.2, 0, 0]])
    structures = [LiFePO4, BaNiO3] * 5 + [isolated_atom, two_body]
    energies = [-2.0, -3.0] * 5 + [-1.0, -1.5]
    forces = [np.zeros((len(s), 3)).tolist() for s in structures]
    element_types = get_element_list([LiFePO4, BaNiO3])
    converter = Structure2Graph(element_types=element_types, cutoff=5.0)
    dataset = MGLDataset(
        threebody_cutoff=4.0,
        structures=structures,
        converter=converter,
        include_line_graph=True,
        labels={"energies": energies, "forces": forces},
        save_cache=False,
    )
    train_data, val_data, test_data = split_dataset(
        dataset,
        frac_list=[0.8, 0.1, 0.1],
        shuffle=True,
        random_state=42,
    )
    my_collate_fn = partial(collate_fn_pes, include_stress=False, include_line_graph=True)
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=my_collate_fn,
        batch_size=2,
        num_workers=0,
        generator=torch.Generator(device=device),
    )
    model = M3GNet(element_types=element_types, is_intensive=False)
    lit_model = PotentialLightningModule(model=model, include_line_graph=True, stress_weight=0.0)
    # We will use CPU if MPS is available since there is a serious bug.
    trainer = pl.Trainer(max_epochs=5, accelerator=device, inference_mode=False)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lit_model, dataloaders=test_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
    assert pred_LFP_energy < 0
    assert pred_BNO_energy < 0
    teardown()


def test_m3gnet_property_training(LiFePO4, BaNiO3):
    isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
    structures = [LiFePO4] * 5 + [BaNiO3] * 5 + [isolated_atom]
    label = [-2] * 5 + [-3] * 5 + [-1]  # Artificial dataset.
    element_types = get_element_list([LiFePO4, BaNiO3])
    converter = Structure2Graph(element_types=element_types, cutoff=5.0)
    dataset = MGLDataset(
        threebody_cutoff=4.0,
        structures=structures,
        converter=converter,
        include_line_graph=True,
        labels={"eform": label},
        save_cache=False,
    )
    train_data, val_data, test_data = split_dataset(
        dataset,
        frac_list=[0.8, 0.1, 0.1],
        shuffle=True,
        random_state=42,
    )
    # This modification is required for M3GNet property dataset
    collate_fn_property = partial(collate_fn_graph, include_line_graph=True)
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=collate_fn_property,
        batch_size=2,
        num_workers=0,
        generator=torch.Generator(device=device),
    )
    model = M3GNet(
        element_types=element_types,
        is_intensive=True,
        readout_type="set2set",
    )
    lit_model = ModelLightningModule(
        model=model, include_line_graph=True, loss="huber_loss", loss_params={"delta": 1.0}
    )
    # We will use CPU if MPS is available since there is a serious bug.
    trainer = pl.Trainer(max_epochs=2, accelerator=device)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
    assert pred_LFP_energy < 0
    assert pred_BNO_energy < 0

    results = trainer.predict(model=lit_model, dataloaders=test_loader)

    assert "MAE" in results[0][0]

    lit_model = ModelLightningModule(model=model, include_line_graph=True, loss="l1_loss")
    # We will use CPU if MPS is available since there is a serious bug.
    trainer = pl.Trainer(max_epochs=2, accelerator=device)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
    assert pred_LFP_energy < 0
    assert pred_BNO_energy < 0

    results = trainer.predict(model=lit_model, dataloaders=test_loader)

    assert "MAE" in results[0][0]
    teardown()


def test_so3net_property_training(LiFePO4, BaNiO3):
    isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
    structures = [LiFePO4] * 5 + [BaNiO3] * 5 + [isolated_atom]
    label = [-2] * 5 + [-3] * 5 + [-1]  # Artificial dataset.
    element_types = get_element_list([LiFePO4, BaNiO3])
    converter = Structure2Graph(element_types=element_types, cutoff=5.0)
    dataset = MGLDataset(
        threebody_cutoff=4.0,
        structures=structures,
        converter=converter,
        labels={"eform": label},
        save_cache=False,
    )
    train_data, val_data, test_data = split_dataset(
        dataset,
        frac_list=[0.8, 0.1, 0.1],
        shuffle=True,
        random_state=42,
    )
    # This modification is required for M3GNet property dataset
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=collate_fn_graph,
        batch_size=2,
        num_workers=0,
        generator=torch.Generator(device=device),
    )
    model = SO3Net(
        element_types=element_types,
        is_intensive=True,
        lmax=2,
        target_property="graph",
        readout_type="set2set",
    )
    lit_model = ModelLightningModule(model=model, loss="smooth_l1_loss", loss_params={"beta": 1.0})
    # We will use CPU if MPS is available since there is a serious bug.
    trainer = pl.Trainer(max_epochs=2, accelerator=device)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
    assert pred_LFP_energy < 0
    assert pred_BNO_energy < 0

    results = trainer.predict(model=lit_model, dataloaders=test_loader)

    assert "MAE" in results[0][0]

    lit_model = ModelLightningModule(model=model, loss="l1_loss")
    # We will use CPU if MPS is available since there is a serious bug.
    trainer = pl.Trainer(max_epochs=2, accelerator=device)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
    assert pred_LFP_energy < 0
    assert pred_BNO_energy < 0

    results = trainer.predict(model=lit_model, dataloaders=test_loader)

    assert "MAE" in results[0][0]
    teardown()


def test_tensornet_property_training(LiFePO4, BaNiO3):
    isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
    structures = [LiFePO4] * 5 + [BaNiO3] * 5 + [isolated_atom]
    label = [-2] * 5 + [-3] * 5 + [-1]  # Artificial dataset.
    element_types = get_element_list([LiFePO4, BaNiO3])
    converter = Structure2Graph(element_types=element_types, cutoff=5.0)
    dataset = MGLDataset(
        threebody_cutoff=4.0,
        structures=structures,
        converter=converter,
        labels={"eform": label},
        save_cache=False,
    )
    train_data, val_data, test_data = split_dataset(
        dataset,
        frac_list=[0.8, 0.1, 0.1],
        shuffle=True,
        random_state=42,
    )
    # This modification is required for M3GNet property dataset
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=collate_fn_graph,
        batch_size=2,
        num_workers=0,
        generator=torch.Generator(device=device),
    )
    model = TensorNet(
        element_types=element_types,
        is_intensive=True,
        readout_type="set2set",
    )
    lit_model = ModelLightningModule(model=model)
    # We will use CPU if MPS is available since there is a serious bug.
    trainer = pl.Trainer(max_epochs=2, accelerator=device)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
    assert pred_LFP_energy < 0
    assert pred_BNO_energy < 0

    results = trainer.predict(model=lit_model, dataloaders=test_loader)

    assert "MAE" in results[0][0]

    lit_model = ModelLightningModule(model=model, loss="l1_loss")
    # We will use CPU if MPS is available since there is a serious bug.
    trainer = pl.Trainer(max_epochs=2, accelerator=device)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
    assert pred_LFP_energy < 0
    assert pred_BNO_energy < 0

    results = trainer.predict(model=lit_model, dataloaders=test_loader)

    assert "MAE" in results[0][0]
    teardown()


def test_m3gnet_property_trainin_multiple_values_per_target(LiFePO4, BaNiO3):
    isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
    structures = [LiFePO4] * 5 + [BaNiO3] * 5 + [isolated_atom]
    label = np.full((11, 5), -1.0).tolist()  # Artificial dataset.
    element_types = get_element_list([LiFePO4, BaNiO3])
    converter = Structure2Graph(element_types=element_types, cutoff=5.0)
    dataset = MGLDataset(
        threebody_cutoff=4.0,
        structures=structures,
        converter=converter,
        include_line_graph=True,
        labels={"multiple_values": label},
        save_cache=False,
    )
    train_data, val_data, test_data = split_dataset(
        dataset,
        frac_list=[0.8, 0.1, 0.1],
        shuffle=True,
        random_state=42,
    )
    # This modification is required for M3GNet property dataset
    collate_fn_property = partial(collate_fn_graph, include_line_graph=True, multiple_values_per_target=True)
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=collate_fn_property,
        batch_size=2,
        num_workers=0,
        generator=torch.Generator(device=device),
    )
    model = M3GNet(
        element_types=element_types,
        is_intensive=True,
        readout_type="set2set",
        ntargets=5,
    )
    lit_model = ModelLightningModule(model=model, include_line_graph=True)
    # We will use CPU if MPS is available since there is a serious bug.
    trainer = pl.Trainer(max_epochs=2, accelerator=device)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
    assert torch.any(pred_LFP_energy < 0)
    assert torch.any(pred_BNO_energy < 0)

    results = trainer.predict(model=lit_model, dataloaders=test_loader)

    assert "MAE" in results[0][0]

    lit_model = ModelLightningModule(model=model, include_line_graph=True, loss="l1_loss")
    # We will use CPU if MPS is available since there is a serious bug.
    trainer = pl.Trainer(max_epochs=2, accelerator=device)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
    assert torch.any(pred_LFP_energy < 0)
    assert torch.any(pred_BNO_energy < 0)

    results = trainer.predict(model=lit_model, dataloaders=test_loader)

    assert "MAE" in results[0][0]
    teardown()


@pytest.mark.parametrize("threebody_cutoff", [0, 3])
def test_chgnet_training(LiFePO4, BaNiO3, threebody_cutoff):
    isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
    two_body = Structure(Lattice.cubic(10.0), ["Li", "Li"], [[0, 0, 0], [0.5, 0, 0]])
    structures = [LiFePO4, BaNiO3] * 5 + [isolated_atom, two_body]
    energies = [-2.0, -3.0] * 5 + [-1.0, -1.5]
    forces = [np.zeros((len(s), 3)).tolist() for s in structures]
    stresses = [np.zeros((3, 3)).tolist()] * len(structures)
    magmoms = [np.zeros((len(s), 1)).tolist() for s in structures]
    element_types = get_element_list([LiFePO4, BaNiO3])
    converter = Structure2Graph(element_types=element_types, cutoff=6.0)
    dataset = MGLDataset(
        threebody_cutoff=threebody_cutoff,
        structures=structures,
        converter=converter,
        include_line_graph=True,
        directed_line_graph=True,
        labels={
            "energies": energies,
            "forces": forces,
            "stresses": stresses,
            "magmoms": magmoms,
        },
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
        collate_fn=partial(collate_fn_pes, include_magmom=True, include_line_graph=True),
        batch_size=2,
        num_workers=0,
        generator=torch.Generator(device=device),
    )
    model = CHGNet(element_types=element_types, threebody_cutoff=threebody_cutoff, is_intensive=False)
    lit_model = PotentialLightningModule(
        model=model, stress_weight=0.1, magmom_weight=0.1, include_line_graph=True, magmom_target="symbreak"
    )
    # We will use CPU if MPS is available since there is a serious bug.
    trainer = pl.Trainer(max_epochs=2, accelerator=device, inference_mode=False)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lit_model, dataloaders=test_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    assert torch.any(pred_LFP_energy < 0)
    assert torch.any(pred_BNO_energy < 0)
    # specify customize optimizer and scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5, amsgrad=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000 * 10, eta_min=1.0e-2 * 1.0e-3)
    lit_model = PotentialLightningModule(
        model=model,
        stress_weight=0.1,
        magmom_weight=0.1,
        include_line_graph=True,
        loss="huber_loss",
        optimizer=optimizer,
        scheduler=scheduler,
    )
    trainer = pl.Trainer(max_epochs=2, accelerator=device, inference_mode=False)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lit_model, dataloaders=test_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    assert pred_LFP_energy < 0
    assert pred_BNO_energy < 0

    teardown()


def test_chgnet_training_without_m(LiFePO4, BaNiO3):
    isolated_atom = Structure(Lattice.cubic(10.0), ["Li"], [[0, 0, 0]])
    two_body = Structure(Lattice.cubic(10.0), ["Li", "Li"], [[0, 0, 0], [0.5, 0, 0]])
    structures = [LiFePO4, BaNiO3] * 5 + [isolated_atom, two_body]
    energies = [-2.0, -3.0] * 5 + [-1.0, -1.5]
    forces = [np.zeros((len(s), 3)).tolist() for s in structures]
    stresses = [np.zeros((3, 3)).tolist()] * len(structures)
    element_types = get_element_list([LiFePO4, BaNiO3])
    converter = Structure2Graph(element_types=element_types, cutoff=6.0)
    dataset = MGLDataset(
        threebody_cutoff=3.0,
        structures=structures,
        converter=converter,
        include_line_graph=True,
        directed_line_graph=True,
        labels={
            "energies": energies,
            "forces": forces,
            "stresses": stresses,
        },
        save_cache=False,
    )
    train_data, val_data, test_data = split_dataset(
        dataset,
        frac_list=[0.8, 0.1, 0.1],
        shuffle=True,
        random_state=42,
    )
    my_collate_fn = partial(collate_fn_pes, include_line_graph=True)
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        collate_fn=my_collate_fn,
        batch_size=2,
        num_workers=0,
        generator=torch.Generator(device=device),
    )
    model = CHGNet(element_types=element_types, is_intensive=False)
    lit_model = PotentialLightningModule(model=model, include_line_graph=True)
    # We will use CPU if MPS is available since there is a serious bug.
    trainer = pl.Trainer(max_epochs=2, accelerator=device, inference_mode=False)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lit_model, dataloaders=test_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    assert torch.any(pred_LFP_energy < 0)
    assert torch.any(pred_BNO_energy < 0)

    teardown()


def test_chgnet_training_with_missing_label(LiFePO4, BaNiO3):
    structures = [LiFePO4, BaNiO3] * 5
    energies = [-2.0, -3.0] * 5
    forces = [np.ones((len(s), 3)).tolist() for s in structures]
    stresses = [np.zeros((3, 3)).tolist()] * len(structures)
    magmoms = [np.ones((len(s), 1)).tolist() for s in structures]
    # Create some missing labels
    energies[2] = np.nan
    forces[4] = (np.nan * np.ones((len(structures[4]), 3))).tolist()
    stresses[6] = (np.nan * np.ones((3, 3))).tolist()
    magmoms[8] = (np.nan * np.ones((len(structures[8]), 1))).tolist()
    element_types = get_element_list([LiFePO4, BaNiO3])
    converter = Structure2Graph(element_types=element_types, cutoff=6.0)
    dataset = MGLDataset(
        threebody_cutoff=3.0,
        structures=structures,
        converter=converter,
        include_line_graph=True,
        directed_line_graph=True,
        labels={
            "energies": energies,
            "forces": forces,
            "stresses": stresses,
            "magmoms": magmoms,
        },
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
        collate_fn=partial(collate_fn_pes, include_magmom=True, include_line_graph=True),
        batch_size=4,
        num_workers=0,
        generator=torch.Generator(device=device),
    )
    model = CHGNet(element_types=element_types, is_intensive=False)
    lit_model = PotentialLightningModule(
        model=model,
        stress_weight=0.1,
        magmom_weight=0.1,
        include_line_graph=True,
        allow_missing_labels=True,
    )
    # We will use CPU if MPS is available since there is a serious bug.
    trainer = pl.Trainer(max_epochs=2, accelerator=device, inference_mode=False)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lit_model, dataloaders=test_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4)
    pred_BNO_energy = model.predict_structure(BaNiO3)

    assert torch.any(pred_LFP_energy < 0)
    assert torch.any(pred_BNO_energy < 0)

    teardown()


def test_qet_training(LiFePO4, BaNiO3):
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
        model=model, stress_weight=0.0001, charge_weight=0.001, loss="smooth_l1_loss", loss_params={"beta": 1.0}
    )
    # We will use CPU if MPS is available since there is a serious bug.
    trainer = pl.Trainer(max_epochs=2, accelerator=device, inference_mode=False)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lit_model, dataloaders=test_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4, total_charge=torch.tensor([0.0]))
    pred_BNO_energy = model.predict_structure(BaNiO3, total_charge=torch.tensor([0.0]))

    # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
    assert pred_LFP_energy < 0
    assert pred_BNO_energy < 0
    # specify customize optimizer and scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5, amsgrad=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=1000 * 10, eta_min=1.0e-2 * 1.0e-3)
    lit_model = PotentialLightningModule(
        model=model,
        stress_weight=0.0001,
        charge_weight=0.0001,
        loss="l1_loss",
        optimizer=optimizer,
        scheduler=scheduler,
    )
    trainer = pl.Trainer(max_epochs=2, accelerator=device, inference_mode=False)

    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(lit_model, dataloaders=test_loader)

    pred_LFP_energy = model.predict_structure(LiFePO4, total_charge=torch.tensor([0.0]))
    pred_BNO_energy = model.predict_structure(BaNiO3, total_charge=torch.tensor([0.0]))

    # We are not expecting accuracy with 2 epochs. This just tests that the energy is actually < 0.
    assert pred_LFP_energy < 0
    assert pred_BNO_energy < 0

    teardown()


@pytest.mark.parametrize("distribution", ["normal", "uniform", "fake"])
def test_xavier_init(distribution):
    model = MEGNet()
    # get a parameter
    w = model.output_proj.layers[0].get_parameter("weight").clone()

    if distribution == "fake":
        with pytest.raises(ValueError, match=r"^Invalid distribution:."):
            xavier_init(model, distribution=distribution)
    else:
        xavier_init(model, distribution=distribution)
        print(w)
        assert not torch.allclose(w, model.output_proj.layers[0].get_parameter("weight"))
        assert torch.allclose(torch.tensor(0.0), model.output_proj.layers[0].get_parameter("bias"))


def test_prediction_logger_train_and_val(LiFePO4, BaNiO3, tmp_path):
    """PredictionLogger captures per-epoch train + val preds (DGL backend)."""
    from matgl.utils.callbacks import PredictionLogger, add_sample_indices

    torch.manual_seed(0)
    structures = [LiFePO4, BaNiO3] * 4
    energies = [-2.0, -3.0] * 4
    forces = [np.zeros((len(s), 3)).tolist() for s in structures]
    stresses = [np.zeros((3, 3)).tolist()] * len(structures)
    element_types = get_element_list([LiFePO4, BaNiO3])
    converter = Structure2Graph(element_types=element_types, cutoff=5.0)
    dataset = MGLDataset(
        threebody_cutoff=4.0,
        structures=structures,
        converter=converter,
        include_line_graph=False,
        labels={"energies": energies, "forces": forces, "stresses": stresses},
        save_cache=False,
    )
    train_data, val_data, _ = split_dataset(
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
    n_train_atoms = sum(train_data[i][0].num_nodes() for i in range(n_train))
    n_val_atoms = sum(val_data[i][0].num_nodes() for i in range(n_val))

    model = TensorNet(element_types=element_types, is_intensive=False)
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
    assert log["train_force_preds"].shape == (n_epochs, n_train_atoms, 3)
    assert log["val_energy_preds"].shape == (n_epochs, n_val)
    assert log["val_force_preds"].shape == (n_epochs, n_val_atoms, 3)
    assert torch.allclose(
        log["train_energy_errors"],
        log["train_energy_preds"] - log["train_energy_labels"].unsqueeze(0),
    )

    assert log_path.exists()
    on_disk = torch.load(log_path, weights_only=True)
    assert torch.equal(on_disk["train_energy_preds"], log["train_energy_preds"])

    teardown()


def test_prediction_logger_with_stress_and_charge(LiFePO4, BaNiO3, tmp_path):
    """PredictionLogger captures per-epoch stress and per-atom charge predictions (DGL)."""
    from matgl.utils.callbacks import PredictionLogger, add_sample_indices

    torch.manual_seed(0)
    structures = [LiFePO4, BaNiO3] * 4
    energies = [-2.0, -3.0] * 4
    forces = [np.zeros((len(s), 3)).tolist() for s in structures]
    stresses = [np.zeros((3, 3)).tolist()] * len(structures)
    charges = [np.zeros(len(s)).tolist() for s in structures]
    element_types = get_element_list([LiFePO4, BaNiO3])
    converter = Structure2Graph(element_types=element_types, cutoff=5.0)
    dataset = MGLDataset(
        threebody_cutoff=4.0,
        structures=structures,
        converter=converter,
        include_line_graph=False,
        include_ref_charge=True,
        labels={"energies": energies, "forces": forces, "stresses": stresses, "charges": charges},
        save_cache=False,
    )
    train_data, val_data, _ = split_dataset(
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
        collate_fn=partial(collate_fn_pes, include_charge=True),
        batch_size=2,
        num_workers=0,
        generator=torch.Generator(device=device),
    )
    n_train = len(train_data)
    n_val = len(val_data)
    n_train_atoms = sum(train_data[i][0].num_nodes() for i in range(n_train))
    n_val_atoms = sum(val_data[i][0].num_nodes() for i in range(n_val))

    model = QET(element_types=element_types, is_intensive=False, use_smooth=True, rbf_type="SphericalBessel")
    lit_model = PotentialLightningModule(
        model=model,
        stress_weight=0.0001,
        charge_weight=0.001,
        loss="mse_loss",
    )
    log_path = tmp_path / "predictions.pt"
    logger_cb = PredictionLogger(save_path=log_path, log_train=True, log_validation=True)
    n_epochs = 2
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
    assert log["train_stress_preds"].shape == (n_epochs, n_train, 3, 3)
    assert log["train_stress_labels"].shape == (n_train, 3, 3)
    assert torch.allclose(
        log["train_stress_errors"],
        log["train_stress_preds"] - log["train_stress_labels"].unsqueeze(0),
    )
    assert log["train_charge_preds"].shape == (n_epochs, n_train_atoms)
    assert log["train_charge_labels"].shape == (n_train_atoms,)
    assert log["val_stress_preds"].shape == (n_epochs, n_val, 3, 3)
    assert log["val_charge_preds"].shape == (n_epochs, n_val_atoms)

    on_disk = torch.load(log_path, weights_only=True)
    assert torch.equal(on_disk["train_stress_preds"], log["train_stress_preds"])
    assert torch.equal(on_disk["train_charge_preds"], log["train_charge_preds"])

    teardown()
