"""Interface with pymatgen objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from pymatgen.core import Element

from matgl.graph._converters import GraphConverter

try:
    from matgl.ext._alchmtk import neighbor_list_from_molecule, neighbor_list_from_structure

    _alchmtk_available = True
except ImportError:
    _alchmtk_available = False

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
    """Construct a PyG graph from Pymatgen Molecules."""

    def __init__(
        self,
        element_types: tuple[str, ...],
        cutoff: float = 5.0,
    ):
        """Initialize the Molecule2Graph converter.

        Args:
            element_types: List of elements present in dataset for graph conversion. This ensures all graphs are
                constructed with the same dimensionality of features.
            cutoff: Cutoff radius for graph representation
        """
        self.element_types = tuple(element_types)
        self.cutoff = cutoff

    def get_graph(self, mol: Molecule):
        """Get a PyG graph from an input molecule.

        Args:
            mol: pymatgen molecule object

        Returns:
            g: PyG graph
            lat: default lattice for molecular systems (np.ones)
            state_attr: state features
        """
        natoms = len(mol)
        element_types = self.element_types
        weight = mol.composition.weight / len(mol)

        if _alchmtk_available:
            src_id, dst_id, _, positions = neighbor_list_from_molecule(
                molecule=mol,
                cutoff=self.cutoff,
                compute_distances=False,
            )
            nbonds = len(src_id) / (2 * natoms)
            lattice_matrix = torch.eye(3, dtype=torch.float32, device=src_id.device).unsqueeze(0)
            images = torch.zeros(len(src_id), 3, dtype=torch.float32, device=src_id.device)
        else:
            cart_coords = mol.cart_coords
            dist = np.linalg.norm(cart_coords[:, None, :] - cart_coords[None, :, :], axis=-1)
            dists = mol.distance_matrix.flatten()
            nbonds = (np.count_nonzero(dists <= self.cutoff) - natoms) / 2
            nbonds /= natoms
            import scipy.sparse as sp

            adj = sp.csr_matrix(dist <= self.cutoff) - sp.eye(natoms, dtype=np.bool_)
            adj = adj.tocoo()
            src_id, dst_id = adj.row, adj.col
            images = np.zeros((len(src_id), 3))  # type: ignore[assignment]
            lattice_matrix = np.expand_dims(np.identity(3), axis=0)  # type: ignore[assignment]
            positions = cart_coords  # type: ignore[assignment]

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
    """Construct a PyG graph from Pymatgen Structure."""

    def __init__(
        self,
        element_types: tuple[str, ...],
        cutoff: float = 5.0,
    ):
        """Initialize the Structure2Graph converter.

        Args:
            element_types: List of elements present in dataset for graph conversion. This ensures all graphs are
                constructed with the same dimensionality of features.
            cutoff: Cutoff radius for graph representation
        """
        self.element_types = tuple(element_types)
        self.cutoff = cutoff

    def get_graph(self, structure: Structure):
        """Get a PyG graph from an input Structure.

        Args:
            structure: pymatgen structure object

        Returns:
            g: PyG graph
            lat: lattice for periodic systems
            state_attr: state features
        """
        element_types = self.element_types

        if _alchmtk_available:
            src_id, dst_id, _, images, _ = neighbor_list_from_structure(
                structure=structure,
                cutoff=self.cutoff,
                compute_distances=False,
            )
            lattice_matrix = torch.as_tensor(
                structure.lattice.matrix.copy(), dtype=torch.float32, device=src_id.device
            ).unsqueeze(0)
            frac_coords = torch.as_tensor(structure.frac_coords, dtype=torch.float32, device=src_id.device)
        else:
            from pymatgen.optimization.neighbors import find_points_in_spheres

            numerical_tol = 1.0e-8
            pbc = np.array([1, 1, 1], dtype=np.int64)
            lattice_matrix = structure.lattice.matrix  # type: ignore[assignment]
            cart_coords = structure.cart_coords
            src_id, dst_id, images, bond_dist = find_points_in_spheres(
                cart_coords,
                cart_coords,
                r=self.cutoff,
                pbc=pbc,
                lattice=lattice_matrix,
                tol=numerical_tol,
            )
            exclude_self = (src_id != dst_id) | (bond_dist > numerical_tol)
            src_id, dst_id, images = src_id[exclude_self], dst_id[exclude_self], images[exclude_self]
            lattice_matrix = np.expand_dims(lattice_matrix, axis=0)  # type: ignore[assignment]
            frac_coords = structure.frac_coords

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
