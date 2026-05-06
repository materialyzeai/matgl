"""Interface with pymatgen objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
from pymatgen.core.periodic_table import Element
from pymatgen.optimization.neighbors import find_points_in_spheres

from matgl.graph._converters_dgl import GraphConverter

if TYPE_CHECKING:
    import dgl
    import torch
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
        """Initialize the Molecule2Graph converter.

        Args:
            element_types: List of elements present in dataset for graph conversion. This ensures all graphs are
                constructed with the same dimensionality of features.
            cutoff: Cutoff radius for graph representation
        """
        self.element_types = tuple(element_types)
        self.cutoff = cutoff

    def get_graph(self, mol: Molecule) -> tuple[dgl.DGLGraph, torch.Tensor, list]:
        """Get a DGL graph from an input molecule.

        Args:
            mol: pymatgen molecule object

        Returns:
            g: DGL graph
            lat: default lattice for molecular systems (np.ones)
            state_attr: state features
        """
        natoms = len(mol)
        R = mol.cart_coords
        element_types = self.element_types
        weight = mol.composition.weight / len(mol)
        dist = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
        dists = mol.distance_matrix.flatten()
        nbonds = (np.count_nonzero(dists <= self.cutoff) - natoms) / 2
        nbonds /= natoms
        adj = sp.csr_matrix(dist <= self.cutoff) - sp.eye(natoms, dtype=np.bool_)
        adj = adj.tocoo()
        g, lat, _ = super().get_graph_from_processed_structure(
            mol,
            adj.row,
            adj.col,
            np.zeros((len(adj.row), 3)),
            np.expand_dims(np.identity(3), axis=0),
            element_types,
            R,
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
        """Initialize the Structure2Graph converter.

        Args:
            element_types: List of elements present in dataset for graph conversion. This ensures all graphs are
                constructed with the same dimensionality of features.
            cutoff: Cutoff radius for graph representation
        """
        self.element_types = tuple(element_types)
        self.cutoff = cutoff

    def get_graph(self, structure: Structure) -> tuple[dgl.DGLGraph, torch.Tensor, list | np.ndarray]:
        """Get a DGL graph from an input Structure.

        Args:
            structure: pymatgen structure object

        Returns:
            g: DGL graph
            lat: lattice for periodic systems
            state_attr: state features
        """
        numerical_tol = 1.0e-8
        pbc = np.array([1, 1, 1], dtype=np.int64)
        element_types = self.element_types
        lattice_matrix = structure.lattice.matrix
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
        src_id, dst_id, images, bond_dist = (
            src_id[exclude_self],
            dst_id[exclude_self],
            images[exclude_self],
            bond_dist[exclude_self],
        )
        g, lat, state_attr = super().get_graph_from_processed_structure(
            structure,
            src_id,
            dst_id,
            images,
            [lattice_matrix],
            element_types,
            structure.frac_coords,
        )
        return g, lat, state_attr
