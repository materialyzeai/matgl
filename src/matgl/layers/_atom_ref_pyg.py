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
        self.register_buffer("onehot", torch.eye(self.max_z))

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
        one_hot = torch.as_tensor(self.onehot)[g.node_type]  # type: ignore[index]

        if self.property_offset.ndim > 1:
            offset_batched_with_state_list: list[torch.Tensor] = []
            for i in range(self.property_offset.size(dim=0)):
                property_offset_batched = self.property_offset[i].repeat(g.num_nodes, 1).to(one_hot.device)
                offset = property_offset_batched * one_hot
                atomic_offset = torch.sum(offset, dim=1)
                offset_batched = global_add_pool(atomic_offset, g.batch)
                offset_batched_with_state_list.append(offset_batched)
            offset_batched_with_state: torch.Tensor = torch.stack(offset_batched_with_state_list)
            return offset_batched_with_state[state_attr] if state_attr is not None else offset_batched_with_state
        property_offset_batched = self.property_offset.repeat(g.num_nodes, 1).to(one_hot.device)
        offset = property_offset_batched * one_hot
        atomic_offset = torch.sum(offset, dim=1)
        offset_batched = global_add_pool(atomic_offset, g.batch)
        return offset_batched
