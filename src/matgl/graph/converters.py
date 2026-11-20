"""Tools to convert materials representations from Pymatgen and other codes to PyG graphs."""

from __future__ import annotations

import abc

import numpy as np
import torch
from torch_geometric.data import Data

import matgl


def get_system_metadata(structure, is_atoms: bool = False) -> tuple[np.ndarray, float, float]:
    """Return PBC, net charge and spin multiplicity without using target labels.

    ASE inputs may declare ``total_charge`` (or ``charge``) and
    ``spin_multiplicity`` in ``Atoms.info``. Unknown spin multiplicities are NaN;
    a periodic structure does not implicitly have a molecular singlet state.
    """
    if is_atoms:
        pbc = np.asarray(structure.pbc, dtype=bool)
        metadata = structure.info
        charge = getattr(structure, "charge", np.sum(structure.get_initial_charges()))
        spin = getattr(structure, "spin_multiplicity", float("nan"))
    else:
        pbc = np.asarray(structure.lattice.pbc if hasattr(structure, "lattice") else [False] * 3, dtype=bool)
        metadata = getattr(structure, "properties", {})
        charge = getattr(structure, "charge", 0.0)
        spin = getattr(structure, "spin_multiplicity", float("nan"))
    charge = float(metadata.get("total_charge", metadata.get("charge", charge)))
    spin = float(metadata.get("spin_multiplicity", spin))
    if not np.isfinite(charge):
        raise ValueError("total_charge must be finite.")
    if not np.isnan(spin) and (not np.isfinite(spin) or spin < 1 or not spin.is_integer()):
        raise ValueError("spin_multiplicity must be a positive integer when supplied.")
    return pbc, charge, spin


class GraphConverter(metaclass=abc.ABCMeta):
    """Abstract base class for converters from input crystals/molecules to graphs."""

    @abc.abstractmethod
    def get_graph(self, structure) -> tuple[Data, torch.Tensor, list | np.ndarray]:
        """Get a graph from a structure.

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
        """Construct a PyTorch Geometric Data object from processed structure and bond information.

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

        # Convert lattice matrix to tensor. Callers occasionally pass a list of
        # numpy arrays (e.g. ``[lattice_matrix]`` for a single periodic cell);
        # routing those through ``np.asarray`` first avoids the
        # "Creating a tensor from a list of numpy.ndarrays is extremely slow"
        # warning from ``torch.as_tensor``. We also force a copy when the array
        # is not writable (``pymatgen``'s ``Lattice.matrix`` returns a
        # read-only array, and ``np.expand_dims`` propagates that flag) since
        # PyTorch warns when constructing a tensor from a non-writable array.
        if not isinstance(lattice_matrix, torch.Tensor):
            lattice_matrix = np.asarray(lattice_matrix)
            if not lattice_matrix.flags.writeable:
                lattice_matrix = lattice_matrix.copy()
        lattice = torch.as_tensor(lattice_matrix, dtype=matgl.float_th, device=device)

        # Create node features (node_type based on element indices). Use a dict
        # lookup instead of list.index() so the cost is O(N_atoms) rather than
        # O(N_atoms * len(element_types)) — meaningful on chemically diverse
        # datasets (alloys, MatPES) where element_types can be 80+ entries long.
        element_to_index = {elem: idx for idx, elem in enumerate(element_types)}
        if is_atoms:
            node_type = np.fromiter(
                (element_to_index[elem] for elem in structure.get_chemical_symbols()),
                dtype=np.int64,
                count=len(structure),
            )
        else:
            node_type = np.fromiter(
                (element_to_index[site.specie.symbol] for site in structure),
                dtype=np.int64,
                count=len(structure),
            )
        graph.node_type = torch.tensor(node_type, dtype=torch.long, device=device)  # Node features

        # Add fractional coordinates as node attribute
        graph.frac_coords = torch.as_tensor(frac_coords, dtype=matgl.float_th, device=device)

        # Keep one metadata row per graph so PyG concatenates mixed systems correctly.
        pbc, charge, spin = get_system_metadata(structure, is_atoms=is_atoms)
        graph.pbc = torch.as_tensor(pbc, dtype=torch.bool, device=device).reshape(1, 3)
        graph.total_charge = torch.tensor([charge], dtype=matgl.float_th, device=device)
        graph.spin_multiplicity = torch.tensor([spin], dtype=matgl.float_th, device=device)

        # Default state attributes
        state_attr = np.array([0.0, 0.0], dtype=matgl.float_np)

        return graph, lattice, state_attr
