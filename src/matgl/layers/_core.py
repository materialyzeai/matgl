"""Neural-network primitives used across matgl.

Collects the small reusable PyTorch building blocks that the graph
convolutions and readouts compose:

* :class:`MLP` -- a configurable multi-layer perceptron with a single
  activation and optional final-layer activation;
* :class:`GatedMLP` -- the gated variant (sigmoid-gate * value-branch)
  used by M3GNet, CHGNet and MEGNet edge/node updates;
* :class:`GatedEquivariantBlock` and
  :func:`build_gated_equivariant_mlp` -- vector-aware MLPs that preserve
  rotational equivariance (used by TensorNet readouts).
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn
from torch.nn import Linear, ModuleList

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class MLP(nn.Module):
    """An implementation of a multi-layer perceptron."""

    def __init__(
        self,
        dims: Sequence[int],
        activation: Callable[[Tensor], Tensor] | None = None,
        activate_last: bool = False,
        bias_last: bool = True,
    ) -> None:
        """Initialize the MLP.

        Args:
            dims: Dimensions of each layer of MLP.
            activation: Activation function.
            activate_last: Whether to apply activation to last layer.
            bias_last: Whether to apply bias to last layer.
        """
        super().__init__()
        self._depth = len(dims) - 1
        self.layers = ModuleList()

        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            if i < self._depth - 1:
                self.layers.append(Linear(in_dim, out_dim, bias=True))

                if activation is not None:
                    self.layers.append(activation)  # type: ignore
            else:
                self.layers.append(Linear(in_dim, out_dim, bias=bias_last))

                if activation is not None and activate_last:
                    self.layers.append(activation)  # type: ignore

    def __repr__(self) -> str:
        dims = []

        for layer in self.layers:
            if isinstance(layer, Linear):
                dims.append(f"{layer.in_features} \u2192 {layer.out_features}")
            else:
                dims.append(layer.__class__.__name__)

        return f"MLP({', '.join(dims)})"

    @property
    def last_linear(self) -> Linear:
        """Return the last linear layer in the network."""
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                return layer
        msg = "MLP must contain at least one Linear layer."
        raise RuntimeError(msg)

    @property
    def depth(self) -> int:
        """Returns depth of MLP."""
        return self._depth

    @property
    def in_features(self) -> int:
        """Return input features of MLP."""
        first_layer = self.layers[0]
        assert isinstance(first_layer, Linear), "First layer must be Linear"
        return first_layer.in_features

    @property
    def out_features(self) -> int:
        """Returns output features of MLP."""
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                return layer.out_features
        raise RuntimeError

    def forward(self, inputs: Tensor) -> Tensor:
        """Apply each layer in turn."""
        x = inputs
        for layer in self.layers:
            x = layer(x)

        return x


class GatedMLP(nn.Module):
    """An implementation of a Gated multi-layer perceptron."""

    def __init__(self, in_feats: int, dims: Sequence[int], activate_last: bool = True, use_bias: bool = True):
        """Initialize the GatedMLP.

        Args:
            in_feats: Dimension of input features.
            dims: Architecture of neural networks.
            activate_last: Whether applying activation to last layer or not.
            use_bias: Whether applying bias in MLP.
        """
        super().__init__()
        self.in_feats = in_feats
        self.dims = [in_feats, *dims]
        self._depth = len(dims)
        self.layers = nn.Sequential()
        self.gates = nn.Sequential()
        self.use_bias = use_bias
        self.activate_last = activate_last
        for i, (in_dim, out_dim) in enumerate(zip(self.dims[:-1], self.dims[1:], strict=False)):
            if i < self._depth - 1:
                self.layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
                self.gates.append(nn.Linear(in_dim, out_dim, bias=use_bias))
                self.layers.append(nn.SiLU())
                self.gates.append(nn.SiLU())
            else:
                self.layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
                if self.activate_last:
                    self.layers.append(nn.SiLU())
                self.gates.append(nn.Linear(in_dim, out_dim, bias=use_bias))
                self.gates.append(nn.Sigmoid())

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs) * self.gates(inputs)


class GatedEquivariantBlock(nn.Module):
    r"""Gated equivariant block as used for the prediction of tensorial properties by PaiNN.

    Transforms scalar and vector representations using gated nonlinearities. The reference
    implementation is available at
    https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/equivariant.py.

    References:
    ----------
    .. [Schuett2021] Schütt, K. T., Unke, O. T., & Gastegger, M. *Equivariant message passing for the
       prediction of tensorial properties and molecular spectra.* Proceedings of the 38th International
       Conference on Machine Learning (2021).
    """

    def __init__(
        self,
        n_sin: int,
        n_vin: int,
        n_sout: int,
        n_vout: int,
        n_hidden: int,
        activation: nn.Module,
        sactivation: nn.Module | None = None,
    ):
        """Initialize the GatedEquivariantBlock.

        Args:
            n_sin: number of input scalar features.
            n_vin: number of input vector features.
            n_sout: number of output scalar features.
            n_vout: number of output vector features.
            n_hidden: number of hidden units.
            activation: internal activation function.
            sactivation: activation function for scalar outputs.
        """
        super().__init__()
        self.n_sin = n_sin
        self.n_vin = n_vin
        self.n_sout = n_sout
        self.n_vout = n_vout
        self.n_hidden = n_hidden
        self.mix_vectors = MLP(
            dims=[n_vin, 2 * n_vout],
            activation=None,
            activate_last=False,  # No activation in the last layer
            bias_last=False,
        )
        self.scalar_net = MLP(
            dims=[n_sin + n_vout, n_hidden, n_sout + n_vout], activation=activation, activate_last=False, bias_last=True
        )
        self.sactivation = sactivation

    def forward(self, inputs: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
        scalars, vectors = inputs
        vmix = self.mix_vectors(vectors)
        vectors_V, vectors_W = torch.split(vmix, self.n_vout, dim=-1)
        vectors_Vn = torch.norm(vectors_V, dim=-2)

        ctx = torch.cat([scalars, vectors_Vn], dim=-1)
        x = self.scalar_net(ctx)
        s_out, x = torch.split(x, [self.n_sout, self.n_vout], dim=-1)
        v_out = x.unsqueeze(-2) * vectors_W

        if self.sactivation:
            s_out = self.sactivation(s_out)

        return s_out, v_out


def build_gated_equivariant_mlp(
    n_in: int,
    n_out: int,
    activation: nn.Module,
    sactivation: nn.Module | None = None,
    n_hidden: int | Sequence[int] | None = None,
    n_gating_hidden: int | Sequence[int] | None = None,
    n_layers: int = 2,
):
    """Construct a stacked `GatedEquivariantBlock` network.

    The official implementation can be found in
    https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/blocks.py.

    Args:
        n_in: Number of input nodes.
        n_out: Number of output nodes.
        activation: Activation function applied inside each block.
        sactivation: Optional activation function for scalar outputs.
        n_hidden: Hidden-layer sizes. If an integer, use the same size for all hidden layers.
            If ``None``, construct a pyramidal network by halving after each layer.
        n_gating_hidden: Hidden sizes for the gating network. Supports the same conventions
            as ``n_hidden``.
        n_layers: Total number of `GatedEquivariantBlock` layers.

    Returns:
        A sequential module stacking `n_layers` `GatedEquivariantBlock` instances.
    """
    # get list of number of nodes in input, hidden & output layers
    if n_hidden is None:
        c_neurons = n_in
        n_neurons = []
        for _ in range(n_layers):
            n_neurons.append(c_neurons)
            c_neurons = max(n_out, c_neurons // 2)
        n_neurons.append(n_out)
    else:
        # get list of number of nodes hidden layers
        n_hidden = [n_hidden] * (n_layers - 1) if isinstance(n_hidden, int) else list(n_hidden)
        n_neurons = [n_in, *n_hidden, n_out]
    if n_gating_hidden is None:
        n_gating_hidden = n_neurons[:-1]
    elif isinstance(n_gating_hidden, int):
        n_gating_hidden = [n_gating_hidden] * n_layers
    else:
        n_gating_hidden = list(n_gating_hidden)
    # assign a GatedEquivariantBlock (with activation function) to each hidden layer
    layers = [
        GatedEquivariantBlock(
            n_sin=n_neurons[i],
            n_vin=n_neurons[i],
            n_sout=n_neurons[i + 1],
            n_vout=n_neurons[i + 1],
            n_hidden=n_gating_hidden[i],
            activation=activation,
            sactivation=sactivation,
        )
        for i in range(n_layers - 1)
    ]
    # assign a GatedEquivariantBlock (without scalar activation function)
    # to the output layer
    layers.append(
        GatedEquivariantBlock(
            n_sin=n_neurons[-2],
            n_vin=n_neurons[-2],
            n_sout=n_neurons[-1],
            n_vout=n_neurons[-1],
            n_hidden=n_gating_hidden[-1],
            activation=activation,
            sactivation=None,
        )
    )
    # put all layers together to make the network
    return nn.Sequential(*layers)
