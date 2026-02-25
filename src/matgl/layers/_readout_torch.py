"""Pure PyTorch readout layers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from ._core import MLP, GatedMLP

if TYPE_CHECKING:
    from collections.abc import Sequence


class WeightedAtomReadOut(nn.Module):
    """Weighted atom readout for graph properties using pure PyTorch tensors."""

    def __init__(self, in_feats: int, dims: Sequence[int], activation: nn.Module):
        super().__init__()
        self.dims = [in_feats, *dims]
        self.activation = activation
        self.mlp = MLP(dims=self.dims, activation=self.activation, activate_last=True)
        self.weight = nn.Sequential(nn.Linear(in_feats, 1), nn.Sigmoid())

    def forward(self, node_feat: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        """Aggregate weighted node features into graph-level representations."""
        h = self.mlp(node_feat)
        w = self.weight(node_feat)
        weighted_h = h * w

        if batch is not None:
            num_graphs = int(batch.max().item()) + 1
            out = torch.zeros(num_graphs, weighted_h.size(1), device=weighted_h.device, dtype=weighted_h.dtype)
            out.index_add_(0, batch.to(torch.long), weighted_h)
        else:
            out = weighted_h.sum(dim=0, keepdim=True)

        return out


class ReduceReadOut(nn.Module):
    """Reduce node features into graph-level representations."""

    def __init__(self, op: str = "mean", field: str = "node_feat"):
        super().__init__()
        self.op = op
        self.field = field
        if op not in ["mean", "sum", "max"]:
            raise ValueError("op must be 'mean', 'sum', or 'max'")

    def forward(self, node_feat: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        if batch is not None:
            num_graphs = int(batch.max().item()) + 1
            if self.op == "sum":
                out = torch.zeros(num_graphs, node_feat.size(1), device=node_feat.device, dtype=node_feat.dtype)
                out.index_add_(0, batch.to(torch.long), node_feat)
            elif self.op == "mean":
                out = torch.zeros(num_graphs, node_feat.size(1), device=node_feat.device, dtype=node_feat.dtype)
                out.index_add_(0, batch.to(torch.long), node_feat)
                counts = torch.zeros(num_graphs, device=node_feat.device, dtype=torch.long)
                counts.index_add_(0, batch.to(torch.long), torch.ones_like(batch, dtype=torch.long))
                out = out / counts.unsqueeze(1).clamp(min=1)
            else:  # max
                out = torch.full(
                    (num_graphs, node_feat.size(1)),
                    float("-inf"),
                    device=node_feat.device,
                    dtype=node_feat.dtype,
                )
                out.index_reduce_(0, batch.to(torch.long), node_feat, "amax", include_self=False)
        else:
            if self.op == "sum":
                out = node_feat.sum(dim=0, keepdim=True)
            elif self.op == "mean":
                out = node_feat.mean(dim=0, keepdim=True)
            else:  # max
                out = node_feat.max(dim=0, keepdim=True)[0]

        return out


class WeightedReadOut(nn.Module):
    """Per-node gated readout for atomic properties."""

    def __init__(self, in_feats: int, dims: Sequence[int], num_targets: int):
        super().__init__()
        self.in_feats = in_feats
        self.dims = [in_feats, *dims, num_targets]
        self.gated = GatedMLP(in_feats=in_feats, dims=self.dims, activate_last=False)

    def forward(self, node_feat: torch.Tensor) -> torch.Tensor:
        return self.gated(node_feat)
