# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Warp kernels for SO(3)-equivariant 3x3 tensor matrix multiplication."""

import warp as wp

from .utils import add_module, get_wp_fp_dtype


def generate_tensor_matmul_so3_3x3(dtype: str):
    """Generate Warp kernels for SO(3)-equivariant 3x3 matrix multiplication: C = AB."""
    dtype_wp = get_wp_fp_dtype(dtype)

    class mat3x3(wp.types.matrix(shape=(3, 3), dtype=dtype_wp)):
        pass

    def tensor_matmul_so3_3x3_fwd(
        A: wp.array(ndim=4, dtype=dtype_wp),
        B: wp.array(ndim=4, dtype=dtype_wp),
        C: wp.array(ndim=4, dtype=dtype_wp),
    ):
        b, h = wp.tid()

        a_reg = mat3x3()
        b_reg = mat3x3()
        c_reg = mat3x3()

        for i in range(3):
            for j in range(3):
                a_reg[i, j] = A[b, i, j, h]
                b_reg[i, j] = B[b, i, j, h]

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    c_reg[i, j] += a_reg[i, k] * b_reg[k, j]

        for i in range(3):
            for j in range(3):
                C[b, i, j, h] = c_reg[i, j]

    def tensor_matmul_so3_3x3_bwd(
        A: wp.array(ndim=4, dtype=dtype_wp),
        B: wp.array(ndim=4, dtype=dtype_wp),
        dC: wp.array(ndim=4, dtype=dtype_wp),
        dA: wp.array(ndim=4, dtype=dtype_wp),
        dB: wp.array(ndim=4, dtype=dtype_wp),
    ):
        b, h = wp.tid()

        a_reg = mat3x3()
        b_reg = mat3x3()

        da_reg = mat3x3()
        db_reg = mat3x3()

        dc_reg = mat3x3()

        for i in range(3):
            for j in range(3):
                a_reg[i, j] = A[b, i, j, h]
                b_reg[i, j] = B[b, i, j, h]
                dc_reg[i, j] = dC[b, i, j, h]

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    da_reg[i, k] += dc_reg[i, j] * b_reg[k, j]
                    db_reg[k, j] += dc_reg[i, j] * a_reg[i, k]

        for i in range(3):
            for j in range(3):
                dA[b, i, j, h] = da_reg[i, j]
                dB[b, i, j, h] = db_reg[i, j]

    def tensor_matmul_so3_3x3_bwd_bwd(
        A: wp.array(ndim=4, dtype=dtype_wp),
        B: wp.array(ndim=4, dtype=dtype_wp),
        dA: wp.array(ndim=4, dtype=dtype_wp),
        dB: wp.array(ndim=4, dtype=dtype_wp),
        dC: wp.array(ndim=4, dtype=dtype_wp),
        d2A: wp.array(ndim=4, dtype=dtype_wp),
        d2B: wp.array(ndim=4, dtype=dtype_wp),
        d2C: wp.array(ndim=4, dtype=dtype_wp),
    ):
        b, h = wp.tid()

        a_reg = mat3x3()
        b_reg = mat3x3()

        da_reg = mat3x3()
        db_reg = mat3x3()

        dc_reg = mat3x3()

        d2a_reg = mat3x3()
        d2b_reg = mat3x3()

        d2c_reg = mat3x3()

        for i in range(3):
            for j in range(3):
                a_reg[i, j] = A[b, i, j, h]
                b_reg[i, j] = B[b, i, j, h]

                da_reg[i, j] = dA[b, i, j, h]
                db_reg[i, j] = dB[b, i, j, h]

                dc_reg[i, j] = dC[b, i, j, h]

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    d2a_reg[i, k] += dc_reg[i, j] * db_reg[k, j]
                    d2b_reg[k, j] += dc_reg[i, j] * da_reg[i, k]

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    d2c_reg[i, j] += da_reg[i, k] * b_reg[k, j]
                    d2c_reg[i, j] += a_reg[i, k] * db_reg[k, j]

        for i in range(3):
            for j in range(3):
                d2A[b, i, j, h] = d2a_reg[i, j]
                d2B[b, i, j, h] = d2b_reg[i, j]
                d2C[b, i, j, h] = d2c_reg[i, j]

    return (
        wp.Kernel(
            tensor_matmul_so3_3x3_fwd,
            key=f"tensor_matmul_so3_3x3_{dtype}",
            module=wp.get_module(f"tensor_matmul_so3_3x3_{dtype}"),
        ),
        wp.Kernel(
            tensor_matmul_so3_3x3_bwd,
            key=f"tensor_matmul_so3_3x3_bwd_{dtype}",
            module=wp.get_module(f"tensor_matmul_so3_3x3_bwd_{dtype}"),
        ),
        wp.Kernel(
            tensor_matmul_so3_3x3_bwd_bwd,
            key=f"tensor_matmul_so3_3x3_bwd_bwd_{dtype}",
            module=wp.get_module(f"tensor_matmul_so3_3x3_bwd_bwd_{dtype}"),
        ),
    )


(
    tensor_matmul_so3_3x3_fwd_fp64,
    tensor_matmul_so3_3x3_bwd_fp64,
    tensor_matmul_so3_3x3_bwd_bwd_fp64,
) = generate_tensor_matmul_so3_3x3("float64")
(
    tensor_matmul_so3_3x3_fwd_fp32,
    tensor_matmul_so3_3x3_bwd_fp32,
    tensor_matmul_so3_3x3_bwd_bwd_fp32,
) = generate_tensor_matmul_so3_3x3("float32")
(
    tensor_matmul_so3_3x3_fwd_fp16,
    tensor_matmul_so3_3x3_bwd_fp16,
    tensor_matmul_so3_3x3_bwd_bwd_fp16,
) = generate_tensor_matmul_so3_3x3("float16")

add_module("tensor_matmul_so3_3x3_fwd", ["float64"], tensor_matmul_so3_3x3_fwd_fp64)
add_module("tensor_matmul_so3_3x3_bwd", ["float64"], tensor_matmul_so3_3x3_bwd_fp64)
add_module("tensor_matmul_so3_3x3_bwd_bwd", ["float64"], tensor_matmul_so3_3x3_bwd_bwd_fp64)

add_module("tensor_matmul_so3_3x3_fwd", ["float32"], tensor_matmul_so3_3x3_fwd_fp32)
add_module("tensor_matmul_so3_3x3_bwd", ["float32"], tensor_matmul_so3_3x3_bwd_fp32)
add_module("tensor_matmul_so3_3x3_bwd_bwd", ["float32"], tensor_matmul_so3_3x3_bwd_bwd_fp32)

add_module("tensor_matmul_so3_3x3_fwd", ["float16"], tensor_matmul_so3_3x3_fwd_fp16)
add_module("tensor_matmul_so3_3x3_bwd", ["float16"], tensor_matmul_so3_3x3_bwd_fp16)
add_module("tensor_matmul_so3_3x3_bwd_bwd", ["float16"], tensor_matmul_so3_3x3_bwd_bwd_fp16)
