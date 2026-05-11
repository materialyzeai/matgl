"""DGL implementation of :class:`AtomRef`.

``AtomRef`` adds a per-element constant offset to a model's prediction --
the standard isolated-atom (or "elemental reference") correction used when
training PES models on cohesive energies or absolute DFT energies. The
backend-agnostic public name is re-exported from :mod:`matgl.layers`; the
PyG counterpart lives in :mod:`matgl.layers._atom_ref_pyg`.
"""

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
        # Gather per-atom offsets directly: shape (N,) or (S, N) for multi-state refs.
        # This replaces the previous (N, max_z) one-hot * repeat * multiply * sum pipeline,
        # which allocated max_z times the memory and FLOPs for the same result.
        node_type = g.ndata["node_type"]
        if self.property_offset.ndim > 1:
            # Multi-state: (S, max_z) -> per-atom (S, N) -> per-graph (S, B)
            atomic_offset = self.property_offset[:, node_type]  # (S, N)
            offset_batched_with_state = []
            for i in range(atomic_offset.size(0)):
                g.ndata["atomic_offset"] = atomic_offset[i]
                offset_batched_with_state.append(dgl.readout_nodes(g, "atomic_offset"))
            stacked = torch.stack(offset_batched_with_state)  # (S, B)
            return stacked[state_attr] if state_attr is not None else stacked

        g.ndata["atomic_offset"] = self.property_offset[node_type]
        return dgl.readout_nodes(g, "atomic_offset")
