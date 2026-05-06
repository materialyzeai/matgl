from __future__ import annotations

import dgl
import numpy as np
import torch
from torch import nn

import matgl


class AtomRef(nn.Module):
    """Get total property offset for a system."""

    def __init__(self, property_offset: torch.Tensor | None = None, max_z: int = 89) -> None:
        """Initialize the AtomRef.

        Args:
            property_offset (Tensor): a tensor containing the property offset for each element
                if given max_z is ignored, and the size of the tensor is used instead
            max_z (int): maximum atomic number.
        """
        super().__init__()
        if property_offset is None:
            property_offset = torch.zeros(max_z, dtype=matgl.float_th)
        elif isinstance(property_offset, np.ndarray | list):  # for backward compatibility of saved models
            property_offset = torch.tensor(property_offset, dtype=matgl.float_th)

        self.max_z = property_offset.shape[-1]
        self.register_buffer("property_offset", property_offset)
        self.register_buffer("onehot", torch.eye(self.max_z))

    def get_feature_matrix(self, graphs: list[dgl.DGLGraph]) -> np.ndarray:
        """Get the number of atoms for different elements in the structure.

        Args:
            graphs (list): a list of dgl graph

        Returns:
            features (np.ndarray): a matrix (num_structures, num_elements)
        """
        features = torch.zeros(len(graphs), self.max_z, dtype=matgl.float_th)
        for i, graph in enumerate(graphs):
            atomic_numbers = graph.ndata["node_type"]
            features[i] = torch.bincount(atomic_numbers, minlength=self.max_z)
        return features.cpu().numpy()

    def fit(self, graphs: list[dgl.DGLGraph], properties: torch.Tensor) -> None:
        """Fit the elemental reference values for the properties.

        Args:
            graphs: dgl graphs
            properties (torch.Tensor): tensor of extensive properties
        """
        features = self.get_feature_matrix(graphs)
        self.property_offset = torch.tensor(
            np.linalg.pinv(features.T @ features) @ features.T @ np.array(properties), dtype=matgl.float_th
        )

    def forward(self, g: dgl.DGLGraph, state_attr: torch.Tensor | None = None):
        """Get the total property offset for a system.

        Args:
            g: a batch of dgl graphs
            state_attr: state attributes

        Returns:
            offset_per_graph
        """
        one_hot = torch.as_tensor(self.onehot)[g.ndata["node_type"]]  # type: ignore[index]
        if self.property_offset.ndim > 1:
            offset_batched_with_state = []
            for i in range(self.property_offset.size(dim=0)):
                property_offset_batched = self.property_offset[i].repeat(g.num_nodes(), 1)
                offset = property_offset_batched * one_hot
                g.ndata["atomic_offset"] = torch.sum(offset, 1)
                offset_batched = dgl.readout_nodes(g, "atomic_offset")
                offset_batched_with_state.append(offset_batched)
            offset_batched_with_state = torch.stack(offset_batched_with_state)  # type: ignore
            return offset_batched_with_state[state_attr]  # type: ignore
        property_offset_batched = self.property_offset.repeat(g.num_nodes(), 1).to(one_hot.device)
        offset = property_offset_batched * one_hot
        g.ndata["atomic_offset"] = torch.sum(offset, 1)
        offset_batched = dgl.readout_nodes(g, "atomic_offset")
        return offset_batched
