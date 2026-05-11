"""Normalization layers for graph-structured features.

Two layers are exposed:

* :class:`GraphNorm` -- per-graph normalisation following
  https://proceedings.mlr.press/v139/cai21e.html. Subtracts a learnable
  fraction of the per-graph mean and divides by the per-graph standard
  deviation, then applies a learnable affine. Operates over a DGL graph
  with a ``batched_field`` selector for node- or edge-level features.
* :class:`LayerNorm` -- a thin wrapper around ``nn.LayerNorm`` whose
  ``forward`` accepts and silently ignores trailing positional / keyword
  arguments, so it can be used in the same call-site as :class:`GraphNorm`.

The PyG models currently rely on PyG's built-in normalisers; this module
is exercised mainly from the DGL graph convolutions in
:mod:`matgl.layers._graph_convolution_dgl`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

if TYPE_CHECKING:
    import dgl


class GraphNorm(nn.Module):
    """Graph normalization layer.

    Following the following paper:
        https://proceedings.mlr.press/v139/cai21e.html
    """

    def __init__(self, input_dim: int, eps: float = 1e-5, batched_field: Literal["node", "edge"] = "node"):
        """Init GraphNorm layer.

        Args:
            input_dim: dimension of input features
            eps: value added to denominator for numerical stability
            batched_field: batched attribute, the attributed from which to determine batches.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.mean_scale = nn.Parameter(torch.ones(input_dim))
        self.batched_field = batched_field

    def forward(self, features: torch.Tensor, graph: dgl.DGLGraph):
        """Forward pass.

        Args:
            features (torch.Tensor): features
            graph (dgl.DGLGraph): g

        Returns:
            torch.Tensor: normalized features
        """
        batch_list = graph.batch_num_nodes() if self.batched_field == "node" else graph.batch_num_edges()
        batch_index = torch.arange(graph.batch_size, dtype=torch.long, device=graph.device).repeat_interleave(
            batch_list
        )
        batch_index = batch_index.view((-1,) + (1,) * (features.dim() - 1)).expand_as(features)
        mean = torch.zeros(graph.batch_size, *features.shape[1:]).to(features)
        mean = mean.scatter_add_(0, batch_index, features)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)
        out = features - mean * self.mean_scale
        std = torch.zeros(graph.batch_size, *features.shape[1:]).to(features)
        std = std.scatter_add_(0, batch_index, out.pow(2))
        std = ((std.T / batch_list).T + self.eps).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)

        return self.weight * out / std + self.bias


class LayerNorm(nn.LayerNorm):
    """Sames as nn.LayerNorm but allows arbitrary arguments to forward."""

    def forward(self, inputs: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # type: ignore[override]
        return super().forward(inputs)
