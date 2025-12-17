"""Tools to convert materials representations from Pymatgen and other codes to DGLGraphs."""

from __future__ import annotations

import abc

import numpy as np
import torch
from torch_geometric.data import Data

import matgl


class GraphConverter(metaclass=abc.ABCMeta):
    """Abstract base class for converters from input crystals/molecules to graphs."""

    @abc.abstractmethod
    def get_graph(self, structure) -> tuple[Data, torch.Tensor, list | np.ndarray]:
        """
        Args:
            structure: Input crystal or molecule (e.g., Pymatgen structure or molecule).

        Returns:
            Tuple containing:
            - Data: PyTorch Geometric Data object with edge_index, node features, and edge attributes.
            - torch.Tensor: Lattice matrix.
            - Union[List, np.ndarray]: State attributes.
        """

    def get_graph_from_processed_structure(
        self,
        structure,
        src_id: list[int] | np.ndarray | torch.Tensor,
        dst_id: list[int] | np.ndarray | torch.Tensor,
        images: np.ndarray | torch.Tensor,
        lattice_matrix: np.ndarray | torch.Tensor,
        element_types: tuple[str, ...],
        frac_coords: np.ndarray | torch.Tensor,
        is_atoms: bool = False,
    ) -> tuple[Data, torch.Tensor, np.ndarray]:
        """
        Construct a PyTorch Geometric Data object from processed structure and bond information.

        Args:
            structure: Input crystal or molecule (Pymatgen structure, molecule, or ASE atoms).
            src_id: Site indices for starting point of bonds.
            dst_id: Site indices for destination point of bonds.
            images: Periodic image offsets for the bonds.
            lattice_matrix: Lattice information of the structure.
            element_types: Element symbols of all atoms in the structure.
            frac_coords: Fractional coordinates of all atoms (or Cartesian for molecules).
            is_atoms: Whether the input structure is an ASE Atoms object.

        Returns:
            Tuple containing:
            - Data: PyTorch Geometric Data object with edge_index, node features, and edge attributes.
            - torch.Tensor: Lattice matrix.
            - np.ndarray: State attributes.
        """
        # Create edge_index from src_id and dst_id
        src_id = torch.as_tensor(src_id, dtype=matgl.int_th)
        dst_id = torch.as_tensor(dst_id, dtype=matgl.int_th)
        edge_index = torch.stack([src_id, dst_id], dim=0)

        device = edge_index.device

        # Create Data object
        graph = Data(num_nodes=len(structure), edge_index=edge_index)

        # Add periodic boundary condition (PBC) offset as edge attribute
        pbc_offset = torch.as_tensor(images, dtype=matgl.float_th, device=device)
        graph.pbc_offset = pbc_offset  # Store as edge_attr instead of separate pbc_offset

        # Convert lattice matrix to tensor
        lattice = torch.as_tensor(lattice_matrix, dtype=matgl.float_th, device=device)

        # Create node features (node_type based on element indices)
        if is_atoms:
            node_type = np.array([element_types.index(elem) for elem in structure.get_chemical_symbols()])
        else:
            node_type = np.array([element_types.index(site.specie.symbol) for site in structure])
        graph.node_type = torch.tensor(node_type, dtype=torch.long, device=device)  # Node features

        # Add fractional coordinates as node attribute
        graph.frac_coords = torch.as_tensor(frac_coords, dtype=matgl.float_th, device=device)

        # Default state attributes
        state_attr = np.array([0.0, 0.0], dtype=matgl.float_np)

        return graph, lattice, state_attr
