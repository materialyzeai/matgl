"""Interface with pymatgen objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from pymatgen.core import Element

from matgl.ext._alchmtk import neighbor_list_from_molecule, neighbor_list_from_structure
from matgl.graph._converters_pyg import GraphConverter

if TYPE_CHECKING:
    from pymatgen.core.structure import Molecule, Structure


def get_element_list(train_structures: list[Structure | Molecule]) -> tuple[str, ...]:
    """Get the tuple of elements in the training set for atomic features.

    Args:
        train_structures: pymatgen Molecule/Structure object

    Returns:
        Tuple of elements covered in training set
    """
    elements: set[str] = set()
    for s in train_structures:
        elements.update(s.composition.get_el_amt_dict().keys())
    return tuple(sorted(elements, key=lambda el: Element(el).Z))


class Molecule2Graph(GraphConverter):
    """Construct a DGL graph from Pymatgen Molecules."""

    def __init__(
        self,
        element_types: tuple[str, ...],
        cutoff: float = 5.0,
    ):
        """Parameters
        ----------
        element_types: List of elements present in dataset for graph conversion. This ensures all graphs are
            constructed with the same dimensionality of features.
        cutoff: Cutoff radius for graph representation
        """
        self.element_types = tuple(element_types)
        self.cutoff = cutoff

    def get_graph(self, mol: Molecule):
        """Get a DGL graph from an input molecule.

        :param mol: pymatgen molecule object
        :return:
            g: DGL graph
            lat: default lattice for molecular systems (np.ones)
            state_attr: state features
        """
        src_id, dst_id, _, positions = neighbor_list_from_molecule(
            molecule=mol,
            cutoff=self.cutoff,
            compute_distances=False,
        )
        natoms = len(mol)
        element_types = self.element_types
        weight = mol.composition.weight / len(mol)
        nbonds = len(src_id) / (2 * natoms)
        lattice_matrix = torch.eye(3, dtype=torch.float32, device=src_id.device).unsqueeze(0)
        images = torch.zeros(len(src_id), 3, dtype=torch.float32, device=src_id.device)
        g, lat, _ = super().get_graph_from_processed_structure(
            mol,
            src_id,
            dst_id,
            images,
            lattice_matrix,
            element_types,
            positions,
        )
        state_attr = [weight, nbonds]
        return g, lat, state_attr


class Structure2Graph(GraphConverter):
    """Construct a DGL graph from Pymatgen Structure."""

    def __init__(
        self,
        element_types: tuple[str, ...],
        cutoff: float = 5.0,
    ):
        """Parameters
        ----------
        element_types: List of elements present in dataset for graph conversion. This ensures all graphs are
            constructed with the same dimensionality of features.
        cutoff: Cutoff radius for graph representation
        """
        self.element_types = tuple(element_types)
        self.cutoff = cutoff

    def get_graph(self, structure: Structure):
        """Get a DGL graph from an input Structure.

        :param structure: pymatgen structure object
        :return:
            g: DGL graph
            lat: lattice for periodic systems
            state_attr: state features
        """
        src_id, dst_id, _, images, _ = neighbor_list_from_structure(
            structure=structure,
            cutoff=self.cutoff,
            compute_distances=False,
        )
        element_types = self.element_types
        lattice_matrix = torch.as_tensor(
            structure.lattice.matrix.copy(), dtype=torch.float32, device=src_id.device
        ).unsqueeze(0)
        frac_coords = torch.as_tensor(structure.frac_coords, dtype=torch.float32, device=src_id.device)
        g, lat, state_attr = super().get_graph_from_processed_structure(
            structure,
            src_id,
            dst_id,
            images,
            lattice_matrix,
            element_types,
            frac_coords,
        )
        return g, lat, state_attr
