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
from torch import nn

from matgl.layers._readout_torch import ReduceReadOut, WeightedAtomReadOut, WeightedReadOut

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_close_to_expected(output: torch.Tensor, expected_values, *, rtol=1e-5, atol=1e-6):
    expected = torch.tensor(expected_values, dtype=output.dtype, device=output.device)
    assert output.shape == expected.shape
    assert torch.isfinite(output).all()
    assert torch.allclose(output, expected, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# ReduceReadOut
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["sum", "mean", "max"])
def test_reduce_readout_unbatched(op):
    """The non-batched path collapses (N, F) -> (1, F) for every supported op."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = ReduceReadOut(op=op).forward(x)
    assert out.shape == (1, 3)
    assert torch.isfinite(out).all()
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
    assert torch.isfinite(out).all()
    if op == "sum":
        assert torch.allclose(out, torch.tensor([[4.0, 6.0], [12.0, 14.0]]))
    elif op == "mean":
        assert torch.allclose(out, torch.tensor([[2.0, 3.0], [6.0, 7.0]]))
    else:  # max
        assert torch.allclose(out, torch.tensor([[3.0, 4.0], [7.0, 8.0]]))


def test_reduce_readout_invalid_op_raises():
    with pytest.raises(ValueError, match="op must be"):
        ReduceReadOut(op="bogus")


# ---------------------------------------------------------------------------
# WeightedReadOut
# ---------------------------------------------------------------------------


def test_weighted_readout_numbers():
    """WeightedReadOut maps per-node features to (N, num_targets) with fixed seed."""
    torch.manual_seed(42)
    x = torch.randn(5, 4)
    wr = WeightedReadOut(in_feats=4, dims=[8, 8], num_targets=3)
    out = wr(x)

    expected_values = [
        [-0.07476293295621872, -0.13569995760917664, -0.0838092565536499],
        [-0.0751706138253212, -0.13388681411743164, -0.0648883655667305],
        [-0.07388679683208466, -0.11860352009534836, -0.06600876152515411],
        [-0.0732363611459732, -0.12554557621479034, -0.06692913919687271],
        [-0.06959028542041779, -0.12596376240253448, -0.07409191131591797],
    ]

    _assert_close_to_expected(out, expected_values)


# ---------------------------------------------------------------------------
# WeightedAtomReadOut
# ---------------------------------------------------------------------------


def test_weighted_atom_readout_unbatched_numbers():
    """Unbatched WeightedAtomReadOut collapses all nodes to a single (1, dim) row."""
    torch.manual_seed(42)
    x = torch.randn(6, 4)
    war = WeightedAtomReadOut(in_feats=4, dims=[8, 8], activation=nn.SiLU())
    out = war(x)

    expected_values = [
        [
            0.019682496786117554,
            0.012260183691978455,
            0.1443532407283783,
            0.2507040798664093,
            0.1520005464553833,
            -0.14217662811279297,
            0.04660123586654663,
            -0.04173845052719116,
        ]
    ]

    _assert_close_to_expected(out, expected_values)


def test_weighted_atom_readout_batched_numbers():
    """Batched WeightedAtomReadOut produces one row per graph, graphs produce different outputs."""
    torch.manual_seed(42)
    x = torch.randn(6, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    war = WeightedAtomReadOut(in_feats=4, dims=[8, 8], activation=nn.SiLU())
    out = war(x, batch=batch)

    expected_values = [
        [
            0.08378858864307404,
            0.030509421601891518,
            0.13405263423919678,
            0.24054312705993652,
            0.11397531628608704,
            -0.16801601648330688,
            0.02209596149623394,
            0.0015756934881210327,
        ],
        [
            -0.05510534346103668,
            -0.00902985967695713,
            0.15637019276618958,
            0.2625580430030823,
            0.1963617205619812,
            -0.11203167587518692,
            0.07518972456455231,
            -0.09226985275745392,
        ],
    ]

    _assert_close_to_expected(out, expected_values)
    assert not torch.allclose(out[0], out[1]), "two distinct graphs must produce different readouts"
