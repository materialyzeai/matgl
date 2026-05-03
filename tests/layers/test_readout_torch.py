"""Direct unit tests for the backend-agnostic readout layers.

`matgl.layers._readout_torch` is the pure-PyTorch implementation that
``_tensornet_pyg`` re-exports from. Most of its branches are unreachable
through the existing PyG tests because the high-level model flow only ever
exercises a subset, so we test it directly here to lift coverage of the
batched and edge-case paths.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from matgl.layers._readout_torch import ReduceReadOut, WeightedAtomReadOut, WeightedReadOut


@pytest.mark.parametrize("op", ["sum", "mean", "max"])
def test_reduce_readout_unbatched(op):
    """The non-batched path collapses (N, F) -> (1, F) for every supported op."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = ReduceReadOut(op=op).forward(x)
    assert out.shape == (1, 3)
    if op == "sum":
        assert torch.allclose(out, torch.tensor([[5.0, 7.0, 9.0]]))
    elif op == "mean":
        assert torch.allclose(out, torch.tensor([[2.5, 3.5, 4.5]]))
    else:  # max
        assert torch.allclose(out, torch.tensor([[4.0, 5.0, 6.0]]))


@pytest.mark.parametrize("op", ["sum", "mean", "max"])
def test_reduce_readout_batched(op):
    """The batched path scatters into per-graph rows using the `batch` index."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    batch = torch.tensor([0, 0, 1, 1])
    out = ReduceReadOut(op=op).forward(x, batch=batch)
    assert out.shape == (2, 2)
    if op == "sum":
        assert torch.allclose(out, torch.tensor([[4.0, 6.0], [12.0, 14.0]]))
    elif op == "mean":
        assert torch.allclose(out, torch.tensor([[2.0, 3.0], [6.0, 7.0]]))
    else:  # max
        assert torch.allclose(out, torch.tensor([[3.0, 4.0], [7.0, 8.0]]))


def test_reduce_readout_invalid_op_raises():
    with pytest.raises(ValueError, match="op must be"):
        ReduceReadOut(op="bogus")


def test_weighted_atom_readout_unbatched():
    """Unbatched WeightedAtomReadOut collapses to a single (1, dim) row."""
    war = WeightedAtomReadOut(in_feats=4, dims=[8, 8], activation=nn.SiLU())
    x = torch.randn(6, 4)
    out = war(x)
    assert out.shape == (1, 8)


def test_weighted_atom_readout_batched():
    """Batched WeightedAtomReadOut produces one row per graph in the batch."""
    war = WeightedAtomReadOut(in_feats=4, dims=[8, 8], activation=nn.SiLU())
    x = torch.randn(6, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    out = war(x, batch=batch)
    assert out.shape == (2, 8)


def test_weighted_readout_shape():
    """WeightedReadOut maps per-node features to (N, num_targets)."""
    wr = WeightedReadOut(in_feats=4, dims=[8, 8], num_targets=3)
    x = torch.randn(5, 4)
    out = wr(x)
    assert out.shape == (5, 3)
