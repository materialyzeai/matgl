from __future__ import annotations

import os
import shutil

# This function is used for M3GNet property dataset
from functools import partial

import numpy as np
import pytest
from pymatgen.core import Molecule

import matgl

if matgl.config.BACKEND != "PYG":
    pytest.skip("Skipping PYG tests", allow_module_level=True)
from matgl.ext.pymatgen import Molecule2Graph, Structure2Graph, get_element_list
from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_graph, collate_fn_pes, split_dataset

module_dir = os.path.dirname(os.path.abspath(__file__))


class TestDataset:
    def test_mgl_dataset(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3]
        energies = [-1.0, 2.0]
        forces = [np.zeros((28, 3)).tolist(), np.zeros((10, 3)).tolist()]
        stresses = [np.zeros((3, 3)).tolist(), np.zeros((3, 3)).tolist()]
        element_types = get_element_list(structures)
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = MGLDataset(
            structures=structures,
            converter=cry_graph,
            labels={"energies": energies, "forces": forces, "stresses": stresses},
            clear_processed=True,
            root="MGLDataset_pes",
        )
        g1, lat1, _, pes1 = dataset[0]
        g2, lat2, _, pes2 = dataset[1]
        assert pes1["energies"] == energies[0]
        assert g1.num_edges == cry_graph.get_graph(LiFePO4)[0].num_edges
        assert g1.num_nodes == cry_graph.get_graph(LiFePO4)[0].num_nodes
        assert g2.num_edges == cry_graph.get_graph(BaNiO3)[0].num_edges
        assert g2.num_nodes == cry_graph.get_graph(BaNiO3)[0].num_nodes
        assert np.shape(pes1["forces"])[0] == 28
        assert np.shape(pes2["forces"])[0] == 10
        assert np.allclose(lat1.detach().cpu().numpy(), structures[0].lattice.matrix)
        assert np.allclose(lat2.detach().cpu().numpy(), structures[1].lattice.matrix)
        # Check that structures are indeed cleared.
        assert len(dataset.structures) == 0

    def test_load_mgl_dataset(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3]
        element_types = get_element_list(structures)
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = MGLDataset(
            root="MGLDataset_pes",
        )
        dataset.load()
        g1, lat1, _, pes1 = dataset[0]
        g2, lat2, _, pes2 = dataset[1]
        assert pes1["energies"] == -1.0
        assert g1.num_edges == cry_graph.get_graph(LiFePO4)[0].num_edges
        assert g1.num_nodes == cry_graph.get_graph(LiFePO4)[0].num_nodes
        assert g2.num_edges == cry_graph.get_graph(BaNiO3)[0].num_edges
        assert g2.num_nodes == cry_graph.get_graph(BaNiO3)[0].num_nodes
        assert np.shape(pes1["forces"])[0] == 28
        assert np.shape(pes2["forces"])[0] == 10
        assert np.allclose(lat1.detach().cpu().numpy(), structures[0].lattice.matrix)
        assert np.allclose(lat2.detach().cpu().numpy(), structures[1].lattice.matrix)
        shutil.rmtree(f"{dataset.root}")

    def test_mgl_property_dataset(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3]
        labels = [1.0, -2.0]
        element_types = get_element_list(structures)
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = MGLDataset(
            filename_labels="eform.json",
            structures=structures,
            converter=cry_graph,
            labels={"Eform_per_atom": labels},
        )
        g1, lat1, _, label1 = dataset[0]
        g2, lat2, _, _ = dataset[1]
        assert label1["Eform_per_atom"] == labels[0]
        assert g1.num_edges == cry_graph.get_graph(LiFePO4)[0].num_edges
        assert g1.num_nodes == cry_graph.get_graph(LiFePO4)[0].num_nodes
        assert g2.num_edges == cry_graph.get_graph(BaNiO3)[0].num_edges
        assert g2.num_nodes == cry_graph.get_graph(BaNiO3)[0].num_nodes
        assert np.allclose(lat1.detach().numpy(), structures[0].lattice.matrix)
        assert np.allclose(lat2.detach().numpy(), structures[1].lattice.matrix)

        dataset = MGLDataset(
            filename_labels="eform.json",
            include_line_graph=True,
        )
        g1, lat1, _, label1 = dataset[0]
        g2, lat2, _, _ = dataset[1]
        assert label1["Eform_per_atom"] == labels[0]
        assert g1.num_edges == cry_graph.get_graph(LiFePO4)[0].num_edges
        assert g1.num_nodes == cry_graph.get_graph(LiFePO4)[0].num_nodes
        assert g2.num_edges == cry_graph.get_graph(BaNiO3)[0].num_edges
        assert g2.num_nodes == cry_graph.get_graph(BaNiO3)[0].num_nodes
        shutil.rmtree(f"{dataset.root}")

    def test_mgl_property_dataset_with_graph_label(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3]
        labels = [1.0, -2.0]
        graph_label = [0, 1]
        element_types = get_element_list(structures)
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = MGLDataset(
            filename_labels="eform.json",
            structures=structures,
            converter=cry_graph,
            labels={"Eform_per_atom": labels},
            graph_labels=graph_label,
            save_cache=False,
        )
        _, _, state1, _ = dataset[0]
        _, _, state2, _ = dataset[1]
        assert state1.detach().numpy() == graph_label[0]
        assert state2.detach().numpy() == graph_label[1]

    def test_megnet_dataloader(self, LiFePO4, BaNiO3):
        structures = [LiFePO4] * 10 + [BaNiO3] * 10
        label = np.zeros(20).tolist()
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
        )
        assert len(train_loader) == 8
        assert len(val_loader) == 1
        assert len(test_loader) == 1

        train_loader_new, val_loader_new = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            collate_fn=collate_fn_graph,
            batch_size=2,
            num_workers=0,
        )
        assert len(train_loader_new) == 8
        assert len(val_loader_new) == 1

    def test_megnet_dataloader_for_mol(self):
        coords = [
            [0.000000, 0.000000, 0.000000],
            [0.000000, 0.000000, 1.089000],
            [1.026719, 0.000000, -0.363000],
            [-0.513360, -0.889165, -0.363000],
            [-0.513360, 0.889165, -0.363000],
        ]
        m1 = Molecule(["C", "H", "H", "H", "H"], coords)
        structures = [m1, m1, m1, m1, m1, m1, m1, m1, m1, m1]
        label = np.zeros(10).tolist()
        element_types = get_element_list([m1])
        mol_graph = Molecule2Graph(element_types=element_types, cutoff=1.5)
        dataset = MGLDataset(
            structures=structures,
            converter=mol_graph,
            labels={"label": label},
            save_cache=False,
        )
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.6, 0.2, 0.2],
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
        )
        assert len(train_loader) == 3
        assert len(val_loader) == 1
        assert len(test_loader) == 1

    def test_mgl_dataloader(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3] * 10
        energies = np.zeros(20).tolist()
        f1 = np.zeros((28, 3)).tolist()
        f2 = np.zeros((10, 3)).tolist()
        s = np.zeros((3, 3)).tolist()
        forces = [f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2]
        stresses = [s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s]
        element_types = get_element_list([LiFePO4, BaNiO3])
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = MGLDataset(
            structures=structures,
            converter=cry_graph,
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
            collate_fn=collate_fn_graph,
            batch_size=2,
            num_workers=0,
        )
        assert len(train_loader) == 8
        assert len(val_loader) == 1
        assert len(test_loader) == 1

    def test_mgl_dataloader_without_stresses(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3] * 10
        energies = np.zeros(20).tolist()
        f1 = np.zeros((28, 3)).tolist()
        f2 = np.zeros((10, 3)).tolist()
        np.zeros((3, 3)).tolist()
        forces = [f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2]
        element_types = get_element_list([LiFePO4, BaNiO3])
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = MGLDataset(
            structures=structures,
            converter=cry_graph,
            labels={"energies": energies, "forces": forces},
            save_cache=False,
        )
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.8, 0.1, 0.1],
            shuffle=True,
            random_state=42,
        )
        my_collate_fn = partial(collate_fn_pes, include_stress=False)
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            collate_fn=my_collate_fn,
            batch_size=2,
            num_workers=0,
        )
        assert len(train_loader) == 8
        assert len(val_loader) == 1
        assert len(test_loader) == 1

    def test_mgl_property_dataloader(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3] * 10
        e_form = np.zeros(20)
        element_types = get_element_list([LiFePO4, BaNiO3])
        cry_graph = Structure2Graph(element_types=element_types, cutoff=4.0)
        dataset = MGLDataset(
            structures=structures,
            converter=cry_graph,
            labels={"EForm": e_form},
            save_cache=False,
        )
        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.8, 0.1, 0.1],
            shuffle=True,
            random_state=42,
        )
        # This modification is required for M3GNet property dataset
        my_collate_fn = partial(collate_fn_graph)
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            collate_fn=my_collate_fn,
            batch_size=2,
            num_workers=1,
        )
        assert len(train_loader) == 8
        assert len(val_loader) == 1
        assert len(test_loader) == 1

    def test_mgl_dataset_with_magmom(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3]
        energies = [0 for s in structures]
        forces = [np.zeros((len(s), 3)).tolist() for s in structures]
        stresses = [np.zeros((3, 3)).tolist() for s in structures]
        magmoms = [[1] * len(LiFePO4), [2] * len(BaNiO3)]
        labels = {
            "energies": energies,
            "forces": forces,
            "stresses": stresses,
            "magmoms": magmoms,
        }
        element_types = get_element_list(structures)
        cry_graph = Structure2Graph(element_types=element_types, cutoff=5.0)
        dataset = MGLDataset(
            structures=structures,
            converter=cry_graph,
            labels=labels,
            clear_processed=True,
            save_cache=False,
        )
        _, _, _, label1 = dataset[0]
        _, _, _, label2 = dataset[1]
        assert np.allclose(label1["magmoms"].detach().numpy(), [1] * len(LiFePO4))
        assert np.allclose(label2["magmoms"].detach().numpy(), [2] * len(BaNiO3))

    def test_mgl_dataloader_with_magmom(self, LiFePO4, BaNiO3):
        structures = [LiFePO4, BaNiO3] * 10
        energies = np.zeros(20).tolist()
        f1 = np.zeros((28, 3)).tolist()
        f2 = np.zeros((10, 3)).tolist()
        s = np.zeros((3, 3)).tolist()
        m1 = np.zeros(28).tolist()
        m2 = np.zeros(10).tolist()
        np.zeros((3, 3)).tolist()
        forces = [f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2, f1, f2]
        stresses = [s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s, s]
        magmoms = [m1, m2, m1, m2, m1, m2, m1, m2, m1, m2, m1, m2, m1, m2, m1, m2, m1, m2, m1, m2]

        labels = {
            "energies": energies,
            "forces": forces,
            "stresses": stresses,
            "magmoms": magmoms,
        }
        element_types = get_element_list(structures)
        cry_graph = Structure2Graph(element_types=element_types, cutoff=5.0)
        dataset = MGLDataset(
            structures=structures,
            converter=cry_graph,
            labels=labels,
            clear_processed=True,
            save_cache=True,
        )

        train_data, val_data, test_data = split_dataset(
            dataset,
            frac_list=[0.8, 0.1, 0.1],
            shuffle=True,
            random_state=42,
        )
        # This modification is required for M3GNet property dataset
        my_collate_fn = partial(collate_fn_pes, include_magmom=True)
        train_loader, val_loader, test_loader = MGLDataLoader(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            collate_fn=my_collate_fn,
            batch_size=2,
            num_workers=1,
        )
        assert len(train_loader) == 8
        assert len(val_loader) == 1
        assert len(test_loader) == 1
        shutil.rmtree(f"{dataset.root}")
