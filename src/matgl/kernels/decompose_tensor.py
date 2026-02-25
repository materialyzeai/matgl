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
"""Warp kernels for decomposing 3x3 tensors into I, A, S components."""

import warp as wp

from .utils import add_module, get_wp_fp_dtype


def generate_decompose_tensor(dtype: str, h_last: bool = True, use_irmem: bool = True):
    """Generate Warp kernels for decomposing a 3x3 tensor into I, A, S components."""
    dtype_wp = get_wp_fp_dtype(dtype)

    if not use_irmem:
        raise ValueError(f"only supporting use_irmem True, but got {use_irmem}")

    if not h_last:
        raise ValueError(f"only supporting h_last True but got {h_last}")

    class mat3x3(wp.types.matrix(shape=(3, 3), dtype=dtype_wp)):
        pass

    class vec3(wp.types.vector(length=3, dtype=dtype_wp)):
        pass

    class vec5(wp.types.vector(length=5, dtype=dtype_wp)):
        pass

    dim = 3 if use_irmem else 4

    def decompose_tensor_fwd(
        X: wp.array(ndim=4, dtype=dtype_wp),
        I: wp.array(ndim=dim, dtype=dtype_wp),
        A: wp.array(ndim=dim, dtype=dtype_wp),
        S: wp.array(ndim=dim, dtype=dtype_wp),
    ):
        b, h = wp.tid()

        X_reg = mat3x3()
        for i in range(3):
            for j in range(3):
                X_reg[i, j] = X[b, i, j, h]

        res = X.dtype(0)
        for i in range(3):
            res += X_reg[i, i]
        res = res / X.dtype(3.0)

        I[b, 0, h] = res

        denom = X.dtype(2.0)
        cnt = wp.int32(0)
        for i in range(2):
            for j in range(i + 1, 3):
                A[b, cnt, h] = (X_reg[i, j] - X_reg[j, i]) / denom
                cnt += 1

        cnt = wp.int32(0)
        for i in range(2):
            S[b, cnt, h] = X_reg[i, i] - res
            cnt += 1

            for j in range(i + 1, 3):
                S[b, cnt, h] = (X_reg[i, j] + X_reg[j, i]) / denom
                cnt += 1

    def decompose_tensor_bwd(
        dI: wp.array(ndim=dim, dtype=dtype_wp),
        dA: wp.array(ndim=dim, dtype=dtype_wp),
        dS: wp.array(ndim=dim, dtype=dtype_wp),
        dX: wp.array(ndim=4, dtype=dtype_wp),
    ):
        b, h = wp.tid()

        dX_reg = mat3x3(dX.dtype(0))

        dI_reg = dI[b, 0, h]
        dA_reg = vec3(dX.dtype(0))
        dS_reg = vec5(dX.dtype(0))

        for i in range(3):
            dA_reg[i] = dA[b, i, h]

        for i in range(5):
            dS_reg[i] = dS[b, i, h]

        for i in range(3):
            dX_reg[i, i] = dI_reg / dI.dtype(3.0)

        denom = dX.dtype(2.0)

        cnt = wp.int32(0)

        for i in range(3):
            for j in range(i + 1, 3):
                dX_reg[i, j] += dA_reg[cnt] / denom
                dX_reg[j, i] -= dA_reg[cnt] / denom
                cnt += 1

        cnt = wp.int32(0)
        for i in range(2):
            dX_reg[i, i] += dS_reg[cnt]
            for j in range(3):
                dX_reg[j, j] -= dS_reg[cnt] / dI.dtype(3.0)

            cnt += 1

            for j in range(i + 1, 3):
                dX_reg[i, j] += dS_reg[cnt] / denom
                dX_reg[j, i] += dS_reg[cnt] / denom
                cnt += 1

        for i in range(3):
            for j in range(3):
                dX[b, i, j, h] = dX_reg[i, j]

    def decompose_tensor_bwd_bwd(
        dX: wp.array(ndim=4, dtype=dtype_wp),
        d2I: wp.array(ndim=dim, dtype=dtype_wp),
        d2A: wp.array(ndim=dim, dtype=dtype_wp),
        d2S: wp.array(ndim=dim, dtype=dtype_wp),
    ):
        b, h = wp.tid()

        dX_reg = mat3x3(dX.dtype(0))
        d2I_reg = dX.dtype(0)
        d2A_reg = vec3(dX.dtype(0))
        d2S_reg = vec5(dX.dtype(0))

        for i in range(3):
            for j in range(3):
                dX_reg[i, j] = dX[b, i, j, h]

        for i in range(3):
            d2I_reg += dX_reg[i, i] / d2I.dtype(3.0)

        denom = dX.dtype(2.0)

        cnt = wp.int32(0)
        for i in range(3):
            for j in range(i + 1, 3):
                d2A_reg[cnt] += dX_reg[i, j] / denom
                d2A_reg[cnt] -= dX_reg[j, i] / denom
                cnt += 1

        cnt = wp.int32(0)
        for i in range(2):
            d2S_reg[cnt] += dX_reg[i, i]
            for j in range(3):
                d2S_reg[cnt] -= dX_reg[j, j] / d2I.dtype(3.0)
            cnt += 1

            for j in range(i + 1, 3):
                d2S_reg[cnt] += dX_reg[i, j] / denom
                d2S_reg[cnt] += dX_reg[j, i] / denom
                cnt += 1

        d2I[b, 0, h] = d2I_reg
        for i in range(3):
            d2A[b, i, h] = d2A_reg[i]

        for i in range(5):
            d2S[b, i, h] = d2S_reg[i]

    return (
        wp.Kernel(
            decompose_tensor_fwd,
            key=f"decompose_tensor_{dtype}",
            module=wp.get_module(f"decompose_tensor_{dtype}"),
        ),
        wp.Kernel(
            decompose_tensor_bwd,
            key=f"decompose_tensor_bwd_{dtype}",
            module=wp.get_module(f"decompose_tensor_bwd_{dtype}"),
        ),
        wp.Kernel(
            decompose_tensor_bwd_bwd,
            key=f"decompose_tensor_bwd_bwd_{dtype}",
            module=wp.get_module(f"decompose_tensor_bwd_bwd_{dtype}"),
        ),
    )


decompose_tensor_fwd_fp64, decompose_tensor_bwd_fp64, decompose_tensor_bwd_bwd_fp64 = generate_decompose_tensor(
    "float64"
)
decompose_tensor_fwd_fp32, decompose_tensor_bwd_fp32, decompose_tensor_bwd_bwd_fp32 = generate_decompose_tensor(
    "float32"
)
decompose_tensor_fwd_fp16, decompose_tensor_bwd_fp16, decompose_tensor_bwd_bwd_fp16 = generate_decompose_tensor(
    "float16"
)

add_module("decompose_tensor_fwd", ["float64"], decompose_tensor_fwd_fp64)
add_module("decompose_tensor_bwd", ["float64"], decompose_tensor_bwd_fp64)
add_module("decompose_tensor_bwd_bwd", ["float64"], decompose_tensor_bwd_bwd_fp64)

add_module("decompose_tensor_fwd", ["float32"], decompose_tensor_fwd_fp32)
add_module("decompose_tensor_bwd", ["float32"], decompose_tensor_bwd_fp32)
add_module("decompose_tensor_bwd_bwd", ["float32"], decompose_tensor_bwd_bwd_fp32)

add_module("decompose_tensor_fwd", ["float16"], decompose_tensor_fwd_fp16)
add_module("decompose_tensor_bwd", ["float16"], decompose_tensor_bwd_fp16)
add_module("decompose_tensor_bwd_bwd", ["float16"], decompose_tensor_bwd_bwd_fp16)
