"""Implementations of math functions."""

from __future__ import annotations

from functools import lru_cache
from math import pi, sqrt
from pathlib import Path

import numpy as np
import sympy
import torch
from scipy.optimize import brentq
from scipy.special import spherical_jn

import matgl

# Precomputed Spherical Bessel function roots in a 2D array with dimension [128, 128]. The n-th (0-based index) root of
# order l Spherical Bessel function is the `[l, n]` entry.
SPHERICAL_BESSEL_ROOTS = torch.tensor(np.load(Path(__file__).parent / "sb_roots.npy"), dtype=matgl.float_th)


@lru_cache(maxsize=128)
def spherical_bessel_roots(max_l: int, max_n: int):
    """Calculate the spherical Bessel roots.

    The n-th root of the l-th spherical Bessel function is the `[l, n]` entry of the
    return matrix. The calculation is based on the fact that the n-root for l-th
    spherical Bessel function `j_l`, i.e., `z_{j, n}`, is in the range
    `[z_{j-1,n}, z_{j-1, n+1}]`. On the other hand we know precisely the
    roots for j0, i.e., sinc(x).

    Args:
        max_l: max order of spherical bessel function
        max_n: max number of roots

    Returns:
        root matrix of size [max_l, max_n]
    """
    temp_zeros = np.arange(1, max_l + max_n + 1) * pi  # j0
    roots = [temp_zeros[:max_n]]
    for i in range(1, max_l):
        roots_temp = []
        for j in range(max_n + max_l - i):
            low = temp_zeros[j]
            high = temp_zeros[j + 1]
            root = brentq(lambda x, v: spherical_jn(v, x), low, high, (i,))
            roots_temp.append(root)
        temp_zeros = np.array(roots_temp)
        roots.append(temp_zeros[:max_n])
    return np.array(roots)


def _block_repeat(array, block_size, repeats):
    col_index = torch.arange(array.size()[1])
    indices = []
    start = 0

    for i, b in enumerate(block_size):
        indices.append(torch.tile(col_index[start : start + b], [repeats[i]]).to(array.device))
        start += b
    indices = torch.cat(indices, axis=0)
    return torch.index_select(array, 1, indices)


@lru_cache(maxsize=128)
def _get_lambda_func(max_n, cutoff: float = 5.0):
    r = sympy.symbols("r")
    en = [i**2 * (i + 2) ** 2 / (4 * (i + 1) ** 4 + 1) for i in range(max_n)]

    dn = [1.0]
    for i in range(1, max_n):
        dn_value = 1 - en[i] / dn[-1]
        dn.append(dn_value)

    fnr = [
        (-1) ** i
        * sqrt(2.0)
        * pi
        / cutoff**1.5
        * (i + 1)
        * (i + 2)
        / sympy.sqrt(1.0 * (i + 1) ** 2 + (i + 2) ** 2)
        * (
            sympy.sin(r * (i + 1) * pi / cutoff) / (r * (i + 1) * pi / cutoff)
            + sympy.sin(r * (i + 2) * pi / cutoff) / (r * (i + 2) * pi / cutoff)
        )
        for i in range(max_n)
    ]

    gnr = [fnr[0]]
    for i in range(1, max_n):
        gnr_value = 1 / sympy.sqrt(dn[i]) * (fnr[i] + sympy.sqrt(en[i] / dn[i - 1]) * gnr[-1])
        gnr.append(gnr_value)
    return [sympy.lambdify([r], sympy.simplify(i), torch) for i in gnr]


def get_segment_indices_from_n(ns):
    """Get segment indices from a number array.

    For example, if ``ns = [2, 3]``, the function returns ``[0, 0, 1, 1, 1]``.

    Args:
        ns: torch.Tensor, the number of atoms/bonds array

    Returns:
        torch.Tensor: segment indices tensor
    """
    segments = torch.zeros(ns.sum(), dtype=matgl.int_th)
    segments[ns.cumsum(0)[:-1]] = 1
    return segments.cumsum(0)


def get_range_indices_from_n(ns: torch.Tensor):
    """Give ``ns = [2, 3]``, return ``[0, 1, 0, 1, 2]``.

    Args:
        ns: torch.Tensor, the number of atoms/bonds array

    Returns:
        range indices
    """
    max_n = int(torch.max(ns))
    n = ns.size(dim=0)
    n_range = torch.arange(max_n)
    matrix = n_range.tile(
        [n, 1],
    )
    mask = torch.arange(max_n)[None, :] < ns[:, None]

    #    return matrix[mask]
    return torch.masked_select(matrix, mask)


def repeat_with_n(ns, n):
    """Repeat the first dimension according to n array.

    Args:
        ns (torch.tensor): tensor
        n (torch.tensor): a list of replications

    Returns:
        repeated tensor
    """
    return torch.repeat_interleave(ns, n, dim=0)


def scatter_sum(input_tensor: torch.Tensor, segment_ids: torch.Tensor, num_segments: int, dim: int) -> torch.Tensor:
    """Scatter sum operation along the specified dimension.

    Modified from the torch_scatter library
    (https://github.com/rusty1s/pytorch_scatter).

    Args:
        input_tensor (torch.Tensor): The input tensor to be scattered.
        segment_ids (torch.Tensor): Segment ID for each element in the input tensor.
        num_segments (int): The number of segments.
        dim (int): The dimension along which the scatter sum operation is performed (default: -1).

    Returns:
        resulting tensor
    """
    segment_ids = broadcast(segment_ids, input_tensor, dim)
    size = list(input_tensor.size())
    if segment_ids.numel() == 0:
        size[dim] = 0
    else:
        size[dim] = num_segments
    output = torch.zeros(size, dtype=input_tensor.dtype)
    output = output.to(input_tensor.device)
    segment_ids = segment_ids.to(input_tensor.device)
    return output.scatter_add_(dim, segment_ids, input_tensor)


def scatter_add(x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0) -> torch.Tensor:
    """Sum over values with the same indices.

    Implementation taken from Schnetpack2.0
    (https://github.com/atomistic-machine-learning/schnetpack in
    schnetpack/src/schnetpack/nn/scatter.py).

    Args:
        x: input values
        idx_i: index of central atom i
        dim_size: size of the dimension after reduction
        dim: the dimension to reduce

    Returns:
        resulting output from _scatter_add
    """
    return _scatter_add(x, idx_i, dim_size, dim)


@torch.jit.script
def _scatter_add(x: torch.Tensor, idx_i: torch.Tensor, dim_size: int, dim: int = 0) -> torch.Tensor:
    """Sum over values with the same indices (jit-compiled implementation).

    Implementation taken from Schnetpack2.0
    (https://github.com/atomistic-machine-learning/schnetpack in
    schnetpack/src/schnetpack/nn/scatter.py).

    Args:
        x: input values
        idx_i: index of central atom i
        dim_size: size of the dimension after reduction
        dim: the dimension to reduce

    Returns:
        y: resulting tensors
    """
    shape = list(x.shape)
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return tmp.index_add(dim, idx_i, x)


def scatter_mean(
    x: torch.Tensor,
    idx_i: torch.Tensor,
    dim_size: int,
    dim: int = 0,
) -> torch.Tensor:
    """Average over values with the same indices.

    Args:
        x: input values
        idx_i: index of central atom i
        dim_size: size of the dimension after reduction
        dim: the dimension to reduce

    Returns:
        resulting output from _scatter_mean
    """
    return _scatter_mean(x, idx_i, dim_size, dim)


@torch.jit.script
def _scatter_mean(
    x: torch.Tensor,
    idx_i: torch.Tensor,
    dim_size: int,
    dim: int = 0,
) -> torch.Tensor:
    """Average over values with the same indices.

    Args:
        x: input values
        idx_i: index of central atom i
        dim_size: size of the dimension after reduction
        dim: the dimension to reduce

    Returns:
        y: resulting tensors
    """
    shape = list(x.shape)
    shape[dim] = dim_size

    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    y = tmp.index_add(dim, idx_i, x)

    count_shape = [1 for _ in range(x.dim())]
    count_shape[dim] = dim_size

    count_tmp = torch.zeros(count_shape, dtype=x.dtype, device=x.device)
    ones_shape = [1 for _ in range(x.dim())]
    ones_shape[dim] = x.shape[dim]
    ones = torch.ones(ones_shape, dtype=x.dtype, device=x.device)

    count = count_tmp.index_add(dim, idx_i, ones)
    count = torch.clamp(count, min=1.0)

    return y / count


def unsorted_segment_fraction(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int):
    """Compute the segment fraction of each element.

    Args:
        data (torch.tensor): original data
        segment_ids (torch.tensor): segment ids
        num_segments (int): number of segments

    Returns:
        data (torch.tensor): data after fraction.
    """
    segment_sum = scatter_sum(input_tensor=data, segment_ids=segment_ids, dim=0, num_segments=num_segments)
    sums = torch.gather(segment_sum, 0, segment_ids)
    return torch.div(data, sums)


def broadcast(input_tensor: torch.Tensor, target_tensor: torch.Tensor, dim: int):
    """Broadcast input tensor along a given dimension to match the shape of the target tensor.

    Modified from the torch_scatter library
    (https://github.com/rusty1s/pytorch_scatter).

    Args:
        input_tensor: The tensor to broadcast.
        target_tensor: The tensor whose shape to match.
        dim: The dimension along which to broadcast.

    Returns:
        resulting input tensor after broadcasting
    """
    if input_tensor.dim() == 1:
        for _ in range(dim):
            input_tensor = input_tensor.unsqueeze(0)
    for _ in range(input_tensor.dim(), target_tensor.dim()):
        input_tensor = input_tensor.unsqueeze(-1)
    target_shape = list(target_tensor.shape)
    target_shape[dim] = input_tensor.shape[dim]
    return input_tensor.expand(target_shape)


def binom(n: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Compute binomial coefficients ``(n k)``.

    Taken from Schnetpack2.0
    (https://github.com/atomistic-machine-learning/schnetpack in
    schnetpack/src/schnetpack/nn/ops/math.py).
    """
    return torch.exp(torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1))


# following functions are token from TensorNet
def vector_to_skewtensor(vector: torch.Tensor):
    """Create a skew-symmetric tensor from a vector.

    Args:
        vector: input vectors.

    Returns:
        resulting skew tensors
    """
    batch_size = vector.size(0)
    zero = torch.zeros(batch_size, device=vector.device, dtype=vector.dtype)
    tensor = torch.stack(
        (
            zero,
            -vector[:, 2],
            vector[:, 1],
            vector[:, 2],
            zero,
            -vector[:, 0],
            -vector[:, 1],
            vector[:, 0],
            zero,
        ),
        dim=1,
    )
    tensor = tensor.view(-1, 3, 3)
    return tensor.squeeze(0)


def vector_to_symtensor(vector: torch.Tensor):
    """Create a symmetric traceless tensor from the outer product of a vector with itself.

    Args:
        vector: input vectors.

    Returns:
        resulting symmetric traceless tensors
    """
    tensor = torch.matmul(vector.unsqueeze(-1), vector.unsqueeze(-2))
    scalars = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[..., None, None] * torch.eye(
        3, 3, device=tensor.device, dtype=tensor.dtype
    )
    return 0.5 * (tensor + tensor.transpose(-2, -1)) - scalars


# Full tensor decomposition into irreducible components
def decompose_tensor(tensor: torch.Tensor):
    """Create a symmetric traceless tensor from the outer product of a vector with itself.

    Args:
        tensor: input tensors.

    Returns:
        resulting symmetric traceless tensors.
    """
    scalars = (tensor.diagonal(offset=0, dim1=-1, dim2=-2)).mean(-1)[..., None, None] * torch.eye(
        3, 3, device=tensor.device, dtype=tensor.dtype
    )
    skew_metrices = 0.5 * (tensor - tensor.transpose(-2, -1))
    traceless_tensors = 0.5 * (tensor + tensor.transpose(-2, -1)) - scalars
    return scalars, skew_metrices, traceless_tensors


# Modifies tensor by multiplying invariant features to irreducible components
def new_radial_tensor(
    scalars: torch.Tensor,
    skew_metrices: torch.Tensor,
    traceless_tensors: torch.Tensor,
    f_I: torch.Tensor,
    f_A: torch.Tensor,
    f_S: torch.Tensor,
):
    """Modifies tensor by multiplying invariant features to irreducible components.

    Args:
        scalars: input scalars.
        skew_metrices: input skew tensors.
        traceless_tensors: input traceless tensors.
        f_I: pair distance dependent features for scalars.
        f_A: pair distance dependent features for input skew tensors.
        f_S: pair distance dependent features for input traceless tensors.

    Returns:
        resulting new radial tensors
    """
    scalars = f_I[..., None, None] * scalars
    skew_metrices = f_A[..., None, None] * skew_metrices
    traceless_tensors = f_S[..., None, None] * traceless_tensors
    return scalars, skew_metrices, traceless_tensors


def tensor_norm(tensor: torch.Tensor):
    """Computes Frobenius norm.

    Args:
        tensor: input tensor.

    Returns:
        resulting Frobenius norm of tensors
    """
    return (tensor**2).sum((-2, -1))
