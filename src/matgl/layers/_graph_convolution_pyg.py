from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from matgl.utils.cutoff import cosine_cutoff
from matgl.utils.maths import (
    decompose_tensor,
    new_radial_tensor,
    scatter_add,
    tensor_norm,
)

if TYPE_CHECKING:
    from torch_geometric.data import Data


class TensorNetInteraction(nn.Module):
    """A Graph Convolution block for TensorNet adapted for PyTorch Geometric."""

    def __init__(
        self,
        num_rbf: int,
        units: int,
        activation: nn.Module,
        cutoff: float,
        equivariance_invariance_group: str,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            num_rbf: Number of radial basis functions.
            units: Number of hidden neurons.
            activation: Activation function.
            cutoff: Cutoff radius for graph construction.
            equivariance_invariance_group: Group action on geometric tensor representations, either O(3) or SO(3).
            dtype: Data type for all variables.
        """
        super().__init__()
        self.num_rbf = num_rbf
        self.units = units
        self.cutoff = cutoff
        self.equivariance_invariance_group = equivariance_invariance_group

        # Scalar linear layers
        self.linears_scalar = nn.ModuleList(
            [
                nn.Linear(num_rbf, units, bias=True, dtype=dtype),
                nn.Linear(units, 2 * units, bias=True, dtype=dtype),
                nn.Linear(2 * units, 3 * units, bias=True, dtype=dtype),
            ]
        )

        # Tensor linear layers (6 layers for scalar, skew, and traceless components)
        self.linears_tensor = nn.ModuleList([nn.Linear(units, units, bias=False, dtype=dtype) for _ in range(6)])

        self.act = activation
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize the parameters."""
        for linear in self.linears_scalar:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        for linear in self.linears_tensor:
            nn.init.xavier_uniform_(linear.weight)

    def forward(self, graph: Data, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            graph: PyTorch Geometric Data object containing graph structure.
            X: Node tensor representations.

        Returns:
            X: Updated node tensor representations after message passing.
        """
        edge_index = graph.edge_index
        edge_weight = graph.bond_dist  # Assuming bond_dist is stored in graph
        edge_attr = graph.edge_attr  # Assuming edge_attr is stored in graph

        # Process edge attributes
        C = cosine_cutoff(edge_weight, self.cutoff)
        for linear_scalar in self.linears_scalar:
            edge_attr = self.act(linear_scalar(edge_attr))
        edge_attr = (edge_attr * C.view(-1, 1)).reshape(edge_attr.shape[0], self.units, 3)

        # Normalize input tensor
        X = X / (tensor_norm(X) + 1)[..., None, None]

        # Decompose input tensor
        scalars, skew_metrices, traceless_tensors = decompose_tensor(X)

        # Apply tensor linear transformations
        scalars = self.linears_tensor[0](scalars.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        skew_metrices = self.linears_tensor[1](skew_metrices.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        traceless_tensors = self.linears_tensor[2](traceless_tensors.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        Y = scalars + skew_metrices + traceless_tensors

        # Message passing
        graph.x_I = scalars
        graph.x_A = skew_metrices
        graph.x_S = traceless_tensors
        graph.edge_attr_processed = edge_attr

        messages = self.message(edge_index, graph.x_I, graph.x_A, graph.x_S, graph.edge_attr_processed)
        Im, Am, Sm = self.aggregate(messages, edge_index[0], X.size(0))
        # Combine messages
        msg = Im + Am + Sm

        # Apply group action
        if self.equivariance_invariance_group == "O(3)":
            A = torch.matmul(msg, Y)
            B = torch.matmul(Y, msg)
            scalars, skew_metrices, traceless_tensors = decompose_tensor(A + B)
        elif self.equivariance_invariance_group == "SO(3)":
            B = torch.matmul(Y, msg)
            scalars, skew_metrices, traceless_tensors = decompose_tensor(2 * B)
        else:
            raise ValueError("equivariance_invariance_group must be 'O(3)' or 'SO(3)'")

        # Normalize and apply final tensor transformations
        normp1 = (tensor_norm(scalars + skew_metrices + traceless_tensors) + 1)[..., None, None]
        scalars = scalars / normp1
        skew_metrices = skew_metrices / normp1
        traceless_tensors = traceless_tensors / normp1

        scalars = self.linears_tensor[3](scalars.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        skew_metrices = self.linears_tensor[4](skew_metrices.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        traceless_tensors = self.linears_tensor[5](traceless_tensors.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Compute update
        dX = scalars + skew_metrices + traceless_tensors
        X = X + dX + torch.matmul(dX, dX)

        return X

    def message(self, edge_index, x_I: torch.Tensor, x_A: torch.Tensor, x_S: torch.Tensor, edge_attr: torch.Tensor):
        """Compute messages for each edge."""
        _, dst = edge_index
        x_I_j = x_I[dst]
        x_A_j = x_A[dst]
        x_S_j = x_S[dst]
        scalars, skew_metrices, traceless_tensors = new_radial_tensor(
            x_I_j, x_A_j, x_S_j, edge_attr[..., 0], edge_attr[..., 1], edge_attr[..., 2]
        )
        return scalars, skew_metrices, traceless_tensors

    def aggregate(self, inputs, index, dim_size):
        """Aggregate messages for node updates."""
        scalars, skew_matrices, traceless_tensors = inputs
        scalars_agg = scatter_add(scalars, index, dim_size=dim_size)
        skew_matrices_agg = scatter_add(skew_matrices, index, dim_size=dim_size)
        traceless_tensors_agg = scatter_add(traceless_tensors, index, dim_size=dim_size)
        return scalars_agg, skew_matrices_agg, traceless_tensors_agg
