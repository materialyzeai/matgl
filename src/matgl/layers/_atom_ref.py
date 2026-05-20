""":class:`AtomRef` per-element offset layer.

``AtomRef`` adds a per-element constant offset to a model's prediction --
the standard isolated-atom (or "elemental reference") correction used when
training PES models on cohesive energies or absolute DFT energies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import global_add_pool

if TYPE_CHECKING:
    from torch_geometric.data import Data


class AtomRef(nn.Module):
    """Get total property offset for a system."""

    def __init__(self, property_offset: torch.Tensor | None = None, max_z: int = 89) -> None:
        """Initialize the AtomRef.

        Args:
            property_offset (Tensor): a tensor containing the property offset for each element
                if given, max_z is ignored, and the size of the tensor is used instead
            max_z (int): maximum atomic number.
        """
        super().__init__()
        if property_offset is None:
            property_offset = torch.zeros(max_z, dtype=torch.float32)
        elif isinstance(property_offset, np.ndarray | list):  # for backward compatibility of saved models
            property_offset = torch.tensor(property_offset, dtype=torch.float32)

        self.max_z = property_offset.shape[-1]
        self.register_buffer("property_offset", property_offset)

    def get_feature_matrix(self, graphs: list[Data]) -> np.ndarray:
        """Get the number of atoms for different elements in the structure.

        Args:
            graphs (list): a list of PyG Data objects

        Returns:
            features (np.ndarray): a matrix (num_structures, num_elements)
        """
        features = torch.zeros(len(graphs), self.max_z, dtype=torch.float32)
        for i, graph in enumerate(graphs):
            node_types = graph.node_type  # Node types stored in graph.x
            features[i] = torch.bincount(node_types, minlength=self.max_z)
        return features.cpu().numpy()

    def fit(self, graphs: list[Data], properties: torch.Tensor) -> None:
        """Fit the elemental reference values for the properties.

        Args:
            graphs: PyG Data objects
            properties (torch.Tensor): tensor of extensive properties
        """
        features = self.get_feature_matrix(graphs)
        self.property_offset = torch.tensor(
            np.linalg.pinv(features.T @ features) @ features.T @ np.array(properties), dtype=torch.float32
        )

    def forward(self, g: Data, state_attr: torch.Tensor | None = None):
        """Get the total property offset for a system.

        Args:
            g: a batch of PyG graphs (torch_geometric.data.Data)
            state_attr: state attributes

        Returns:
            offset_per_graph
        """
        # Gather per-atom offsets directly: shape (N,) or (S, N) for multi-state refs.
        # This replaces the previous (N, max_z) one-hot * repeat * multiply * sum pipeline,
        # which allocated max_z times the memory and FLOPs for the same result.
        batch = getattr(g, "batch", None)
        if self.property_offset.ndim > 1:
            # Multi-state: (S, max_z) -> per-atom (S, N) -> per-graph (S, B)
            atomic_offset = self.property_offset[:, g.node_type]  # (S, N)
            offset_per_graph = global_add_pool(atomic_offset.T, batch).T  # (S, B)
            return offset_per_graph[state_attr] if state_attr is not None else offset_per_graph

        atomic_offset = self.property_offset[g.node_type]  # (N,)
        return global_add_pool(atomic_offset, batch)  # (B,)
