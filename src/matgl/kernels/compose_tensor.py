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
"""Warp kernels for composing 3x3 tensors from I, A, S components."""

import warp as wp

from .utils import add_module, get_wp_fp_dtype


def generate_compose_tensor(dtype: str, h_last: bool = True, use_irmem: bool = True):
    """Generate Warp kernels for composing a 3x3 tensor from I, A, S components."""
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

    def compose_tensor_fwd(
        I: wp.array(ndim=dim, dtype=dtype_wp),
        A: wp.array(ndim=dim, dtype=dtype_wp),
        S: wp.array(ndim=dim, dtype=dtype_wp),
        X: wp.array(ndim=4, dtype=dtype_wp),
    ):
        b, h = wp.tid()

        X_reg = mat3x3()

        I_reg = I[b, 0, h]
        A_reg = vec3()
        S_reg = vec5()

        for i in range(3):
            A_reg[i] = A[b, i, h]

        for i in range(5):
            S_reg[i] = S[b, i, h]

        for i in range(3):
            X_reg[i, i] += I_reg

        cnt = wp.int32(0)
        for i in range(3):
            for j in range(i + 1, 3):
                X_reg[i, j] += A_reg[cnt]
                X_reg[j, i] -= A_reg[cnt]
                cnt += 1

        trace_S = -(S_reg[0] + S_reg[3])
        cnt = wp.int32(0)
        for i in range(2):
            X_reg[i, i] += S_reg[cnt]
            cnt += 1
            for j in range(i + 1, 3):
                X_reg[i, j] += S_reg[cnt]
                X_reg[j, i] += S_reg[cnt]
                cnt += 1

        X_reg[2, 2] += trace_S

        for i in range(3):
            for j in range(3):
                X[b, i, j, h] = X_reg[i, j]

    def compose_tensor_bwd(
        dX: wp.array(ndim=4, dtype=dtype_wp),
        dI: wp.array(ndim=dim, dtype=dtype_wp),
        dA: wp.array(ndim=dim, dtype=dtype_wp),
        dS: wp.array(ndim=dim, dtype=dtype_wp),
    ):
        b, h = wp.tid()

        dX_reg = mat3x3()
        for i in range(3):
            for j in range(3):
                dX_reg[i, j] = dX[b, i, j, h]

        dI_reg = dI.dtype(0)
        dA_reg = vec3(dX.dtype(0))
        dS_reg = vec5(dX.dtype(0))

        for i in range(3):
            dI_reg += dX_reg[i, i]

        cnt = wp.int32(0)
        for i in range(3):
            for j in range(i + 1, 3):
                dA_reg[cnt] += dX_reg[i, j]
                dA_reg[cnt] -= dX_reg[j, i]
                cnt += 1

        dS_reg[0] += dX_reg[0, 0]
        dS_reg[0] -= dX_reg[2, 2]

        dS_reg[1] += dX_reg[0, 1]
        dS_reg[1] += dX_reg[1, 0]

        dS_reg[2] += dX_reg[0, 2]
        dS_reg[2] += dX_reg[2, 0]

        dS_reg[3] += dX_reg[1, 1]
        dS_reg[3] -= dX_reg[2, 2]

        dS_reg[4] += dX_reg[1, 2]
        dS_reg[4] += dX_reg[2, 1]

        dI[b, 0, h] = dI_reg

        for i in range(3):
            dA[b, i, h] = dA_reg[i]

        for i in range(5):
            dS[b, i, h] = dS_reg[i]

    def compose_tensor_bwd_bwd(
        dI: wp.array(ndim=dim, dtype=dtype_wp),
        dA: wp.array(ndim=dim, dtype=dtype_wp),
        dS: wp.array(ndim=dim, dtype=dtype_wp),
        d2X: wp.array(ndim=4, dtype=dtype_wp),
    ):
        b, h = wp.tid()
        d2X_reg = mat3x3()

        dI_reg = dI[b, 0, h]
        dA_reg = vec3(dI.dtype(0))
        dS_reg = vec5(dI.dtype(0))

        for i in range(3):
            dA_reg[i] = dA[b, i, h]

        for i in range(5):
            dS_reg[i] = dS[b, i, h]

        for i in range(3):
            d2X_reg[i, i] += dI_reg

        cnt = wp.int32(0)
        for i in range(3):
            for j in range(i + 1, 3):
                d2X_reg[i, j] += dA_reg[cnt]
                d2X_reg[j, i] -= dA_reg[cnt]
                cnt += 1

        cnt = wp.int32(0)
        for i in range(2):
            d2X_reg[i, i] += dS_reg[cnt]
            cnt += 1

            for j in range(i + 1, 3):
                d2X_reg[i, j] += dS_reg[cnt]
                d2X_reg[j, i] += dS_reg[cnt]
                cnt += 1

        d2X_reg[2, 2] -= dS_reg[0]
        d2X_reg[2, 2] -= dS_reg[3]

        for i in range(3):
            for j in range(3):
                d2X[b, i, j, h] = d2X_reg[i, j]

    return (
        wp.Kernel(
            compose_tensor_fwd,
            key=f"compose_tensor_{dtype}",
            module=wp.get_module(f"compose_tensor_{dtype}"),
        ),
        wp.Kernel(
            compose_tensor_bwd,
            key=f"compose_tensor_bwd_{dtype}",
            module=wp.get_module(f"compose_tensor_bwd_{dtype}"),
        ),
        wp.Kernel(
            compose_tensor_bwd_bwd,
            key=f"compose_tensor_bwd_bwd_{dtype}",
            module=wp.get_module(f"compose_tensor_bwd_bwd_{dtype}"),
        ),
    )


(
    compose_tensor_fwd_fp64,
    compose_tensor_bwd_fp64,
    compose_tensor_bwd_bwd_fp64,
) = generate_compose_tensor("float64")
(
    compose_tensor_fwd_fp32,
    compose_tensor_bwd_fp32,
    compose_tensor_bwd_bwd_fp32,
) = generate_compose_tensor("float32")
(
    compose_tensor_fwd_fp16,
    compose_tensor_bwd_fp16,
    compose_tensor_bwd_bwd_fp16,
) = generate_compose_tensor("float16")

add_module("compose_tensor_fwd", ["float64"], compose_tensor_fwd_fp64)
add_module("compose_tensor_bwd", ["float64"], compose_tensor_bwd_fp64)
add_module("compose_tensor_bwd_bwd", ["float64"], compose_tensor_bwd_bwd_fp64)

add_module("compose_tensor_fwd", ["float32"], compose_tensor_fwd_fp32)
add_module("compose_tensor_bwd", ["float32"], compose_tensor_bwd_fp32)
add_module("compose_tensor_bwd_bwd", ["float32"], compose_tensor_bwd_bwd_fp32)

add_module("compose_tensor_fwd", ["float16"], compose_tensor_fwd_fp16)
add_module("compose_tensor_bwd", ["float16"], compose_tensor_bwd_fp16)
add_module("compose_tensor_bwd_bwd", ["float16"], compose_tensor_bwd_bwd_fp16)
