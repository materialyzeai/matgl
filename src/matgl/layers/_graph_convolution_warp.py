"""NVIDIA Warp-accelerated TensorNet interaction.

Drop-in replacement for the PyTorch ``TensorNetInteraction`` in
:mod:`matgl.layers._graph_convolution_pyg` that dispatches the message-
passing gather + tensor-product + scatter step to a custom Warp kernel
(:mod:`matgl.kernels` / :mod:`matgl.ops`), avoiding the Python-side
loops over edges that dominate the PyTorch version's runtime. Requires
the ``nvalchemiops`` optional dependency; selected automatically by
TensorNet on supported devices.
"""

from __future__ import annotations

import torch
from torch import nn

from matgl.ops import (
    fn_compose_tensor,
    fn_decompose_tensor,
    fn_message_passing,
    fn_tensor_matmul_o3_3x3,
    fn_tensor_matmul_so3_3x3,
)
from matgl.utils.cutoff import cosine_cutoff


def _tensor_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Frobenius norm over the two spatial (3x3) dims of warp tensors shaped (N, 3, 3, units)."""
    return (tensor * tensor).sum((-3, -2))


class TensorNetInteraction(nn.Module):
    """TensorNet interaction layer using warp-accelerated message passing from ``nvalchemi-toolkit-ops``."""

    def __init__(
        self,
        num_rbf: int,
        units: int,
        activation: nn.Module,
        cutoff: float,
        equivariance_invariance_group: str,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.num_rbf = num_rbf
        self.units = units
        self.cutoff = cutoff
        self.equivariance_invariance_group = equivariance_invariance_group

        self.linears_scalar = nn.ModuleList(
            [
                nn.Linear(num_rbf, units, bias=True, dtype=dtype),
                nn.Linear(units, 2 * units, bias=True, dtype=dtype),
                nn.Linear(2 * units, 3 * units, bias=True, dtype=dtype),
            ]
        )
        self.linears_tensor = nn.ModuleList([nn.Linear(units, units, bias=False, dtype=dtype) for _ in range(6)])
        self.act = activation
        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.linears_scalar:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        for linear in self.linears_tensor:
            nn.init.xavier_uniform_(linear.weight)

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
        row_data: torch.Tensor,
        row_indices: torch.Tensor,
        row_indptr: torch.Tensor,
        col_data: torch.Tensor,
        col_indices: torch.Tensor,
        col_indptr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            X: Node tensor representations, shape (num_nodes, 3, 3, units)
            edge_index: Edge indices, shape (2, num_edges)
            edge_weight: Edge weights (distances), shape (num_edges,)
            edge_attr: Edge attributes (RBF), shape (num_edges, num_rbf)
            row_data: CSR row data indices for message passing.
            row_indices: CSR row indices for message passing.
            row_indptr: CSR row pointers for message passing.
            col_data: CSC column data indices for message passing.
            col_indices: CSC column indices for message passing.
            col_indptr: CSC column pointers for message passing.

        Returns:
            X: Updated tensor representations, shape (num_nodes, 3, 3, units)
        """
        C = cosine_cutoff(edge_weight, self.cutoff)
        edge_attr_processed = edge_attr
        for linear_scalar in self.linears_scalar:
            edge_attr_processed = self.act(linear_scalar(edge_attr_processed))
        edge_attr_processed = (
            (edge_attr_processed * C.view(-1, 1)).view(edge_attr.shape[0], self.units, 3).mT.contiguous()
        )  # (num_edges, 3, units)

        norm_X = (X * X).sum((-3, -2)) + 1  # (num_nodes, units)
        X = X / norm_X.view(-1, 1, 1, X.shape[-1])

        I, A, S = fn_decompose_tensor(X)  # noqa: E741

        I = self.linears_tensor[0](I)  # noqa: E741
        A = self.linears_tensor[1](A)
        S = self.linears_tensor[2](S)

        Y = fn_compose_tensor(I, A, S)

        Im, Am, Sm = fn_message_passing(
            I,
            A,
            S,
            edge_attr_processed,
            row_data,
            row_indices,
            row_indptr,
            col_data,
            col_indices,
            col_indptr,
        )
        msg = fn_compose_tensor(Im, Am, Sm)

        if self.equivariance_invariance_group == "O(3)":
            C = fn_tensor_matmul_o3_3x3(Y, msg)
        else:  # SO(3)
            C = 2 * fn_tensor_matmul_so3_3x3(Y, msg)
        I, A, S = fn_decompose_tensor(C)  # noqa: E741

        normp1 = (_tensor_norm(C) + 1).unsqueeze(-2)
        I, A, S = I / normp1, A / normp1, S / normp1  # noqa: E741

        I = self.linears_tensor[3](I)  # noqa: E741
        A = self.linears_tensor[4](A)
        S = self.linears_tensor[5](S)
        dX = fn_compose_tensor(I, A, S)
        return X + dX + fn_tensor_matmul_so3_3x3(dX, dX)
