"""NVIDIA Warp-accelerated TensorNet embedding.

Drop-in replacement for the PyTorch ``TensorEmbedding`` in
:mod:`matgl.layers._embedding_pyg` that dispatches the most expensive
gather + outer-product step to a custom Warp kernel
(:mod:`matgl.kernels` / :mod:`matgl.ops`). Only the embedding kernel is
replaced; the upstream and downstream PyTorch ops remain the same so
training and autograd behave identically. Requires the
``nvalchemiops`` optional dependency and is selected automatically by
TensorNet when running on a CUDA device with Warp installed.
"""

from __future__ import annotations

import torch
from torch import nn

import matgl
from matgl.ops import fn_compose_tensor, fn_radial_message_passing
from matgl.utils.cutoff import cosine_cutoff


def _tensor_norm(tensor: torch.Tensor) -> torch.Tensor:
    """Frobenius norm over the two spatial (3x3) dims of warp tensors shaped (N, 3, 3, units)."""
    return (tensor * tensor).sum((-3, -2))


class TensorEmbedding(nn.Module):
    """TensorNet embedding layer using warp-accelerated message passing from ``nvalchemi-toolkit-ops``."""

    def __init__(
        self,
        units: int,
        degree_rbf: int,
        activation: nn.Module,
        ntypes_node: int,
        cutoff: float,
        dtype: torch.dtype = matgl.float_th,
    ):
        super().__init__()
        self.units = units
        self.cutoff = cutoff

        # Create unified distance_proj from 3 temp layers (matches reference RNG pattern).
        self.distance_proj = self._create_distance_proj(degree_rbf, units, dtype=dtype)

        self.emb = nn.Embedding(ntypes_node, units, dtype=dtype)
        self.emb2 = nn.Linear(2 * units, units, dtype=dtype)
        self.act = activation
        self.linears_tensor = nn.ModuleList([nn.Linear(units, units, bias=False, dtype=dtype) for _ in range(3)])
        self.linears_scalar = nn.ModuleList(
            [
                nn.Linear(units, 2 * units, bias=True, dtype=dtype),
                nn.Linear(2 * units, 3 * units, bias=True, dtype=dtype),
            ]
        )
        self.init_norm = nn.LayerNorm(units, dtype=dtype)

        self.reset_parameters()

    def _create_distance_proj(
        self,
        in_features: int,
        units: int,
        dtype: torch.dtype = matgl.float_th,
    ) -> nn.Linear:
        """Create unified distance_proj from 3 separate layers to match reference RNG pattern."""
        d_proj1 = nn.Linear(in_features, units, bias=True, dtype=dtype)
        d_proj2 = nn.Linear(in_features, units, bias=True, dtype=dtype)
        d_proj3 = nn.Linear(in_features, units, bias=True, dtype=dtype)

        layer = torch.nn.utils.skip_init(nn.Linear, in_features, 3 * units, bias=True, dtype=dtype)
        with torch.no_grad():
            layer.weight.copy_(torch.cat([d_proj1.weight, d_proj2.weight, d_proj3.weight], dim=0))
            layer.bias.copy_(torch.cat([d_proj1.bias, d_proj2.bias, d_proj3.bias], dim=0))
        return layer

    def _reset_distance_proj(self) -> None:
        """Reset distance_proj weights using 3 temp layers to match reference RNG pattern."""
        dtype = self.distance_proj.weight.dtype
        d_proj1 = torch.nn.utils.skip_init(
            nn.Linear, self.distance_proj.in_features, self.units, bias=True, dtype=dtype
        )
        d_proj2 = torch.nn.utils.skip_init(
            nn.Linear, self.distance_proj.in_features, self.units, bias=True, dtype=dtype
        )
        d_proj3 = torch.nn.utils.skip_init(
            nn.Linear, self.distance_proj.in_features, self.units, bias=True, dtype=dtype
        )
        d_proj1.reset_parameters()
        d_proj2.reset_parameters()
        d_proj3.reset_parameters()
        with torch.no_grad():
            self.distance_proj.weight.copy_(torch.cat([d_proj1.weight, d_proj2.weight, d_proj3.weight], dim=0))
            self.distance_proj.bias.copy_(torch.cat([d_proj1.bias, d_proj2.bias, d_proj3.bias], dim=0))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Handle legacy checkpoints with separate distance_proj1/2/3 layers."""
        w_keys = [f"{prefix}distance_proj{i}.weight" for i in (1, 2, 3)]
        b_keys = [f"{prefix}distance_proj{i}.bias" for i in (1, 2, 3)]
        new_w = f"{prefix}distance_proj.weight"
        new_b = f"{prefix}distance_proj.bias"

        if all(k in state_dict for k in w_keys + b_keys):
            state_dict[new_w] = torch.cat([state_dict.pop(k) for k in w_keys], dim=0)
            state_dict[new_b] = torch.cat([state_dict.pop(k) for k in b_keys], dim=0)

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def reset_parameters(self):
        """Reinitialize parameters with RNG pattern matching reference implementation."""
        self._reset_distance_proj()
        self.emb.reset_parameters()
        self.emb2.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()
        for linear in self.linears_scalar:
            linear.reset_parameters()
        self.init_norm.reset_parameters()

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_attr: torch.Tensor,
        row_data: torch.Tensor,
        row_indptr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Node types, shape (num_nodes,)
            edge_index: Edge indices, shape (2, num_edges)
            edge_weight: Edge weights (distances), shape (num_edges,)
            edge_vec: Edge vectors, shape (num_edges, 3)
            edge_attr: Edge attributes (RBF), shape (num_edges, num_rbf)
            row_data: CSR row data for source aggregation, shape (num_edges,)
            row_indptr: CSR row indptr for source aggregation, shape (num_nodes+1,)

        Returns:
            X: Tensor representation, shape (num_nodes, 3, 3, units)
        """
        x = self.emb(z)  # (num_nodes, units)

        C = cosine_cutoff(edge_weight, self.cutoff)
        edge_attr = self.distance_proj(edge_attr).view(-1, 3, self.units)

        zij = x.index_select(0, edge_index.t().reshape(-1)).view(-1, self.units * 2)
        Zij = self.emb2(zij)  # (num_edges, units)

        edge_attr_processed = edge_attr.view(-1, 3, self.units) * C.view(-1, 1, 1) * Zij.view(-1, 1, self.units)

        edge_vec_norm = edge_vec / torch.norm(edge_vec, dim=1, keepdim=True).clamp(min=1e-6)
        I, A, S = fn_radial_message_passing(edge_vec_norm, edge_attr_processed, row_data, row_indptr)  # noqa: E741

        X = fn_compose_tensor(I, A, S)  # (num_nodes, 3, 3, units)

        norm = _tensor_norm(X)  # (num_nodes, units)
        norm = self.init_norm(norm)

        for linear_scalar in self.linears_scalar:
            norm = self.act(linear_scalar(norm))

        norm = norm.view(-1, self.units, 3)
        norm_I, norm_A, norm_S = norm.unbind(dim=-1)

        I = self.linears_tensor[0](I) * norm_I.unsqueeze(-2)  # noqa: E741
        A = self.linears_tensor[1](A) * norm_A.unsqueeze(-2)
        S = self.linears_tensor[2](S) * norm_S.unsqueeze(-2)

        return fn_compose_tensor(I, A, S)
