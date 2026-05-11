"""DGL-specific neural-network building blocks.

Counterparts of the primitives in :mod:`matgl.layers._core` that need to
interact with a DGL graph (e.g. for per-graph normalisation inside the
MLP, or for the Set2Set edge readout that depends on DGL message-passing
primitives). Public exports re-exported from :mod:`matgl.layers` when
``MATGL_BACKEND=DGL`` include :class:`EdgeSet2Set`, :class:`MLPNorm`,
and :class:`GatedMLPNorm`.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Literal, cast

import dgl
import torch
from dgl import DGLGraph, broadcast_edges, softmax_edges, sum_edges
from torch import nn
from torch.nn import LSTM, Linear, Module

from matgl.layers import GraphNorm
from matgl.layers._norm import LayerNorm

if TYPE_CHECKING:
    from collections.abc import Sequence


class MLPNorm(nn.Module):
    """Multi-layer perceptron with optional normalization."""

    def __init__(
        self,
        dims: list[int],
        activation: nn.Module | None = None,
        activate_last: bool = False,
        use_bias: bool = True,
        bias_last: bool = True,
        normalization: Literal["graph", "layer"] | None = None,
        normalize_hidden: bool = False,
        norm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize an MLP with optional normalization layers.

        Args:
            dims: Dimensions of each layer of the MLP.
            activation: Activation function applied after each hidden layer.
            activate_last: Whether to apply the activation function to the final layer.
            use_bias: Whether to include biases in intermediate linear layers.
            bias_last: Whether to include a bias term in the final linear layer.
            normalization: Name of the normalization strategy, either ``"graph"`` or ``"layer"``.
            normalize_hidden: Whether to normalize the outputs of hidden layers.
            norm_kwargs: Additional keyword arguments forwarded to the normalization modules.
        """
        super().__init__()
        self._depth = len(dims) - 1
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList() if normalization in ("graph", "layer") else None
        self.activation = activation if activation is not None else nn.Identity()
        self.activate_last = activate_last
        self.normalize_hidden = normalize_hidden
        norm_kwargs = norm_kwargs or {}
        norm_kwargs = cast("dict", norm_kwargs)

        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            if i < self._depth - 1:
                self.layers.append(Linear(in_dim, out_dim, bias=use_bias))
                if normalize_hidden and self.norm_layers is not None:
                    if normalization == "graph":
                        self.norm_layers.append(GraphNorm(out_dim, **norm_kwargs))
                    elif normalization == "layer":
                        self.norm_layers.append(LayerNorm(out_dim, **norm_kwargs))
            else:
                self.layers.append(Linear(in_dim, out_dim, bias=use_bias and bias_last))
                if self.norm_layers is not None:
                    if normalization == "graph":
                        self.norm_layers.append(GraphNorm(out_dim, **norm_kwargs))
                    elif normalization == "layer":
                        self.norm_layers.append(LayerNorm(out_dim, **norm_kwargs))

    def forward(self, inputs: torch.Tensor, g: dgl.Graph | None = None) -> torch.Tensor:
        """Run the stacked linear, activation, and normalization layers."""
        x = inputs
        for i in range(self._depth - 1):
            x = self.layers[i](x)
            if self.norm_layers is not None and self.normalize_hidden:
                x = self.norm_layers[i](x, g)
            x = self.activation(x)

        x = self.layers[-1](x)
        if self.norm_layers is not None:
            x = self.norm_layers[-1](x, g)
        if self.activate_last:
            x = self.activation(x)
        return x


class GatedMLPNorm(nn.Module):
    """Gated multi-layer perceptron constructed with `MLPNorm`."""

    def __init__(
        self,
        in_feats: int,
        dims: Sequence[int],
        activation: nn.Module | None = None,
        activate_last: bool = True,
        use_bias: bool = True,
        bias_last: bool = True,
        normalization: Literal["graph", "layer"] | None = None,
        normalize_hidden: bool = False,
        norm_kwargs: dict[str, Any] | None = None,
    ):
        """Initialize a gated MLP with normalization-aware branches.

        Args:
            in_feats: Dimensionality of the input features.
            dims: Hidden-layer sizes for the gating and value branches.
            activation: Non-linear activation to use between layers.
            activate_last: Whether to apply the activation to the final value layer.
            use_bias: Whether to include biases in intermediate linear layers.
            bias_last: Whether to include a bias term in the final layer.
            normalization: Normalization strategy name, ``"graph"`` or ``"layer"``.
            normalize_hidden: Whether to normalize outputs of hidden layers.
            norm_kwargs: Additional keyword arguments forwarded to normalization layers.
        """
        super().__init__()
        self.in_feats = in_feats
        self.dims = [in_feats, *dims]
        self._depth = len(dims)
        self.use_bias = use_bias
        self.activate_last = activate_last

        activation = activation if activation is not None else nn.SiLU()
        self.layers = MLPNorm(
            self.dims,
            activation=activation,
            activate_last=True,
            use_bias=use_bias,
            bias_last=bias_last,
            normalization=normalization,
            normalize_hidden=normalize_hidden,
            norm_kwargs=norm_kwargs,
        )
        self.gates = MLPNorm(
            self.dims,
            activation,
            activate_last=False,
            use_bias=use_bias,
            bias_last=bias_last,
            normalization=normalization,
            normalize_hidden=normalize_hidden,
            norm_kwargs=norm_kwargs,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor, graph: dgl.Graph | None = None) -> torch.Tensor:
        return self.layers(inputs, graph) * self.sigmoid(self.gates(inputs, graph))


class EdgeSet2Set(Module):
    """Implementation of Set2Set."""

    def __init__(self, input_dim: int, n_iters: int, n_layers: int) -> None:
        """Create a Set2Set-style readout over edges.

        Args:
            input_dim: Size of each input sample.
            n_iters: Number of iterative refinement steps.
            n_layers: Number of recurrent layers in the internal LSTM.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = LSTM(self.output_dim, self.input_dim, n_layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialize learnable parameters."""
        self.lstm.reset_parameters()

    def forward(self, g: DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        """Aggregate edge features with the Set2Set mechanism."""
        with g.local_scope():
            batch_size = g.batch_size

            h = (
                feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
                feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
            )

            q_star = feat.new_zeros(batch_size, self.output_dim)

            for _ in range(self.n_iters):
                q, h = self.lstm(q_star.unsqueeze(0), h)
                q = q.view(batch_size, self.input_dim)
                e = (feat * broadcast_edges(g, q)).sum(dim=-1, keepdim=True)
                g.edata["e"] = e
                alpha = softmax_edges(g, "e")
                g.edata["r"] = feat * alpha
                readout = sum_edges(g, "r")
                q_star = torch.cat([q, readout], dim=-1)

            return q_star
