from __future__ import annotations

import numpy as np
import pytest
import torch

from matgl.utils.maths import (
    SPHERICAL_BESSEL_ROOTS,
    binom,
    decompose_tensor,
    get_range_indices_from_n,
    get_segment_indices_from_n,
    new_radial_tensor,
    repeat_with_n,
    scatter_add,
    scatter_mean,
    scatter_sum,
    spherical_bessel_roots,
    tensor_norm,
    unsorted_segment_fraction,
    vector_to_skewtensor,
    vector_to_symtensor,
)


def test_spherical_bessel_roots():
    roots = spherical_bessel_roots(max_l=1, max_n=5)
    roots2 = SPHERICAL_BESSEL_ROOTS
    assert np.allclose(roots2[0, :5], roots.ravel())
    roots = spherical_bessel_roots(max_l=3, max_n=5)
    assert roots[0][0], pytest.approx(np.pi)


def test_torch_operations():
    ns = torch.tensor([2, 3])
    assert get_segment_indices_from_n(ns).tolist() == [0, 0, 1, 1, 1]
    ns = torch.tensor([2, 3])
    assert get_range_indices_from_n(ns).tolist() == [0, 1, 0, 1, 2]
    assert repeat_with_n(torch.tensor([[0, 0], [1, 1], [2, 2]]), torch.tensor([1, 2, 3])).tolist() == [
        [0, 0],
        [1, 1],
        [1, 1],
        [2, 2],
        [2, 2],
        [2, 2],
    ]


def test_segments():
    x = torch.tensor([1.0, 1.0, 2.0, 3.0])
    res = unsorted_segment_fraction(x, torch.tensor([0, 0, 1, 1]), 2)
    assert np.allclose(res.tolist(), [0.5, 0.5, 0.4, 0.6])


def test_scatter_sum():
    input_tensor = torch.tensor([1, 2, 3, 4, 5, 6])
    segment_ids = torch.tensor([0, 0, 1, 1, 2, 2])
    num_segments = 3
    # Perform the scatter sum
    result = scatter_sum(input_tensor, segment_ids, num_segments, dim=0)

    # Check the correctness of the result
    expected_result = torch.tensor([3, 7, 11], dtype=input_tensor.dtype)
    assert torch.equal(result, expected_result)


def test_scatter_add():
    x = torch.tensor([1, 2, 3, 4, 5, 6])
    idx_i = torch.tensor([0, 0, 1, 1, 2, 2])
    dim_size = 3

    # Perform the scatter add
    result = scatter_add(x, idx_i, dim_size, dim=0)

    # Check the correctness of the result
    expected_result = torch.tensor([3, 7, 11], dtype=x.dtype)
    assert torch.equal(result, expected_result)


def test_scatter_mean():
    """Test scatter_mean for a simple 2D tensor reduced along dim=0."""
    x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )
    idx_i = torch.tensor([0, 0, 1])

    out = scatter_mean(x, idx_i, dim_size=3, dim=0)

    expected = torch.tensor(
        [
            [2.0, 3.0],
            [5.0, 6.0],
            [0.0, 0.0],
        ]
    )

    assert torch.allclose(out, expected)


def test_binom():
    n = torch.tensor([4, 5, 6])
    k = torch.tensor([2, 3, 1])

    # Perform the binomial coefficient calculation
    result = binom(n, k)

    # Check the correctness of the result
    expected_result = torch.tensor([6, 10, 6], dtype=n.dtype)
    for i in range(len(result)):
        assert float(result[i]) == pytest.approx(float(expected_result[i]))


def test_vector_to_skewtensor():
    vector = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # Perform the skew-symmetric tensor calculation
    result = vector_to_skewtensor(vector)
    # Check the correctness of the result
    expected_result = torch.tensor(
        [
            [[0.0, -3.0, 2.0], [3.0, 0.0, -1.0], [-2.0, 1.0, 0.0]],
            [[0.0, -6.0, 5.0], [6.0, 0.0, -4.0], [-5.0, 4.0, 0.0]],
            [[0.0, -9.0, 8.0], [9.0, 0.0, -7.0], [-8.0, 7.0, 0.0]],
        ]
    )
    assert torch.equal(result, expected_result)


def test_vector_to_symtensor():
    vector = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

    # Perform the symmetric traceless tensor calculation
    result = vector_to_symtensor(vector)
    # Check the correctness of the result
    expected_result = torch.tensor(
        [
            [[-3.6667, 2.0000, 3.0000], [2.0000, -0.6667, 6.0000], [3.0000, 6.0000, 4.3333]],
            [[-9.6667, 20.0000, 24.0000], [20.0000, -0.6667, 30.0000], [24.0000, 30.0000, 10.3333]],
            [[-15.6667, 56.0000, 63.0000], [56.0000, -0.6667, 72.0000], [63.0000, 72.0000, 16.3333]],
        ]
    )
    assert torch.allclose(result, expected_result, atol=1e-4)


def test_decompose_tensor():
    tensor = torch.tensor(
        [
            [[-4.0, 5.0, -4.0], [5.0, -6.0, 5.0], [-4.0, 5.0, -4.0]],
            [[-28.0, 32.0, -28.0], [32.0, -36.0, 32.0], [-28.0, 32.0, -28.0]],
            [[-76.0, 80.0, -76.0], [80.0, -84.0, 80.0], [-76.0, 80.0, -76.0]],
        ]
    )

    # Perform the decomposition of the tensor
    scalars, skew_metrices, traceless_tensors = decompose_tensor(tensor)

    # Check the correctness of the result
    expected_scalars = torch.tensor(
        [
            [[-4.6667, -0.0000, -0.0000], [-0.0000, -4.6667, -0.0000], [-0.0000, -0.0000, -4.6667]],
            [[-30.6667, -0.0000, -0.0000], [-0.0000, -30.6667, -0.0000], [-0.0000, -0.0000, -30.6667]],
            [[-78.6667, -0.0000, -0.0000], [-0.0000, -78.6667, -0.0000], [-0.0000, -0.0000, -78.6667]],
        ]
    )
    expected_skew_metrices = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ]
    )
    expected_traceless_tensors = torch.tensor(
        [
            [[0.6667, 5.0000, -4.0000], [5.0000, -1.3333, 5.0000], [-4.0000, 5.0000, 0.6667]],
            [[2.6667, 32.0000, -28.0000], [32.0000, -5.3333, 32.0000], [-28.0000, 32.0000, 2.6667]],
            [[2.6667, 80.0000, -76.0000], [80.0000, -5.3333, 80.0000], [-76.0000, 80.0000, 2.6667]],
        ]
    )
    assert torch.allclose(scalars, expected_scalars, atol=1e-4)
    assert torch.allclose(skew_metrices, expected_skew_metrices, atol=1e-4)
    assert torch.allclose(traceless_tensors, expected_traceless_tensors, atol=1e-4)


def test_new_radial_tensor():
    # Input tensors
    scalars = torch.tensor([1.0, 2.0, 3.0])
    skew_metrices = torch.tensor([[-1.0, 2.0, -3.0], [4.0, -5.0, 6.0], [7.0, -8.0, 9.0]])
    traceless_tensors = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [2.0, -4.0, 5.0], [3.0, 5.0, -6.0]],
            [[-4.0, 8.0, -9.0], [8.0, -12.0, 10.0], [-9.0, 10.0, -14.0]],
            [[-7.6, 8.0, -3.8], [8.0, -8.4, 7.6], [-3.8, 7.6, -7.6]],
        ]
    )

    # Pair distance dependent features
    f_I = torch.tensor([0.5, 0.6, 0.7])
    f_A = torch.tensor([1.0, 1.2, 1.4])
    f_S = torch.tensor([2.0, 2.2, 2.4])

    # Call the function
    scalars, skew_metrices, traceless_tensors = new_radial_tensor(
        scalars, skew_metrices, traceless_tensors, f_I, f_A, f_S
    )

    # Expected output tensors
    expected_scalars = torch.tensor(
        [[[0.5000, 1.0000, 1.5000]], [[0.6000, 1.2000, 1.8000]], [[0.7000, 1.4000, 2.1000]]]
    )
    expected_skew_metrices = torch.tensor(
        [
            [[-1.0000, 2.0000, -3.0000], [4.0000, -5.0000, 6.0000], [7.0000, -8.0000, 9.0000]],
            [[-1.2000, 2.4000, -3.6000], [4.8000, -6.0000, 7.2000], [8.4000, -9.6000, 10.8000]],
            [[-1.4000, 2.8000, -4.2000], [5.6000, -7.0000, 8.4000], [9.8000, -11.2000, 12.6000]],
        ]
    )
    expected_traceless_tensors = torch.tensor(
        [
            [[2.0000, 4.0000, 6.0000], [4.0000, -8.0000, 10.0000], [6.0000, 10.0000, -12.0000]],
            [[-8.8000, 17.6000, -19.8000], [17.6000, -26.4000, 22.0000], [-19.8000, 22.0000, -30.8000]],
            [[-18.2400, 19.2000, -9.1200], [19.2000, -20.1600, 18.2400], [-9.1200, 18.2400, -18.2400]],
        ]
    )

    assert torch.allclose(scalars, expected_scalars, atol=1e-4)
    assert torch.allclose(skew_metrices, expected_skew_metrices, atol=1e-4)
    assert torch.allclose(traceless_tensors, expected_traceless_tensors, atol=1e-4)


def test_tensor_norm():
    # Input tensor
    input_tensor = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0]],
            [[0.5, 1.5, 2.5], [-0.5, -1.5, -2.5]],
        ]
    )

    # Call the function
    norm = tensor_norm(input_tensor)

    # Expected output
    expected_norm = torch.tensor([91.0000, 91.0000, 17.5000])

    assert torch.equal(norm, expected_norm)
