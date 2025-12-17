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

import warp as wp

from .utils import add_module, get_wp_fp_dtype


def generate_tensor_norm3(dtype: str, h_last: bool = True, use_irmem: bool = True):
    dtype_wp = get_wp_fp_dtype(dtype)

    if not use_irmem:
        raise ValueError(f"only supporting use_irmem True, but got {use_irmem}")

    if not h_last:
        raise ValueError(f"only supporting h_last True but got {h_last}")

    class mat3x3(wp.types.matrix(shape=(3, 3), dtype=dtype_wp)):
        pass

    def tensor_norm3_fwd(
        X: wp.array(ndim=4, dtype=dtype_wp),
        output: wp.array(ndim=2, dtype=dtype_wp),
    ):
        b, h = wp.tid()

        x00 = X[b, 0, 0, h]
        x01 = X[b, 0, 1, h]
        x02 = X[b, 0, 2, h]
        x10 = X[b, 1, 0, h]
        x11 = X[b, 1, 1, h]
        x12 = X[b, 1, 2, h]
        x20 = X[b, 2, 0, h]
        x21 = X[b, 2, 1, h]
        x22 = X[b, 2, 2, h]

        one_half = X.dtype(0.5)
        one_third = X.dtype(1.0 / 3.0)

        trace = x00 + x11 + x22
        trace_third = trace / X.dtype(3.0)
        norm2_i = one_third * trace * trace
        norm2_a = one_half * ((x01 - x10) * (x01 - x10) + (x02 - x20) * (x02 - x20) + (x12 - x21) * (x12 - x21))
        norm2_s = one_half * (
            (x01 + x10) * (x01 + x10)
            + (x02 + x20) * (x02 + x20)
            + (x12 + x21) * (x12 + x21)
            ) + (x00 - trace_third) * (x00 - trace_third) + (x11 - trace_third) * (x11 - trace_third) + (x22 - trace_third) * (x22 - trace_third)
        
        output[b, h] = norm2_i
        output[b, h + X.shape[3]] = norm2_a
        output[b, h + 2 * X.shape[3]] = norm2_s

    def tensor_norm3_bwd(
        grad_output: wp.array(ndim=2, dtype=dtype_wp),
        X: wp.array(ndim=4, dtype=dtype_wp),
        grad_X: wp.array(ndim=4, dtype=dtype_wp),
    ):
        b, h = wp.tid()

        grad_i = grad_output[b, h]
        grad_a = grad_output[b, h + X.shape[3]]
        grad_s = grad_output[b, h + 2 * X.shape[3]]

        x00 = X[b, 0, 0, h]
        x01 = X[b, 0, 1, h]
        x02 = X[b, 0, 2, h]
        x10 = X[b, 1, 0, h]
        x11 = X[b, 1, 1, h]
        x12 = X[b, 1, 2, h]
        x20 = X[b, 2, 0, h]
        x21 = X[b, 2, 1, h]
        x22 = X[b, 2, 2, h]

        trace = x00 + x11 + x22
        trace_third = trace / X.dtype(3.0)

        diag_grad_i = X.dtype(2.0 / 3.0) * trace * grad_i

        dev00 = x00 - trace_third
        dev11 = x11 - trace_third
        dev22 = x22 - trace_third

        c4_3 = X.dtype(4.0) / X.dtype(3.0)
        c2_3 = X.dtype(2.0) / X.dtype(3.0)

        grad_s_term_00 = c4_3 * dev00 - c2_3 * dev11 - c2_3 * dev22
        grad_s_term_11 = c4_3 * dev11 - c2_3 * dev00 - c2_3 * dev22
        grad_s_term_22 = c4_3 * dev22 - c2_3 * dev00 - c2_3 * dev11

        grad_X[b, 0, 0, h] = diag_grad_i + grad_s * grad_s_term_00
        grad_X[b, 1, 1, h] = diag_grad_i + grad_s * grad_s_term_11
        grad_X[b, 2, 2, h] = diag_grad_i + grad_s * grad_s_term_22

        diff01 = x01 - x10
        sum01 = x01 + x10
        grad_X[b, 0, 1, h] = grad_a * diff01 + grad_s * sum01
        grad_X[b, 1, 0, h] = -grad_a * diff01 + grad_s * sum01

        diff02 = x02 - x20
        sum02 = x02 + x20
        grad_X[b, 0, 2, h] = grad_a * diff02 + grad_s * sum02
        grad_X[b, 2, 0, h] = -grad_a * diff02 + grad_s * sum02

        diff12 = x12 - x21
        sum12 = x12 + x21
        grad_X[b, 1, 2, h] = grad_a * diff12 + grad_s * sum12
        grad_X[b, 2, 1, h] = -grad_a * diff12 + grad_s * sum12

    def tensor_norm3_bwd_bwd(
        grad_grad_X: wp.array(ndim=4, dtype=dtype_wp),
        grad_grad_output: wp.array(ndim=2, dtype=dtype_wp),
    ):
        b, h = wp.tid()

        gg00 = grad_grad_X[b, 0, 0, h]
        gg01 = grad_grad_X[b, 0, 1, h]
        gg02 = grad_grad_X[b, 0, 2, h]
        gg10 = grad_grad_X[b, 1, 0, h]
        gg11 = grad_grad_X[b, 1, 1, h]
        gg12 = grad_grad_X[b, 1, 2, h]
        gg20 = grad_grad_X[b, 2, 0, h]
        gg21 = grad_grad_X[b, 2, 1, h]
        gg22 = grad_grad_X[b, 2, 2, h]

        trace_gg = gg00 + gg11 + gg22
        trace_third_gg = trace_gg / grad_grad_X.dtype(3.0)

        one_half = grad_grad_X.dtype(0.5)
        one_third = grad_grad_X.dtype(1.0 / 3.0)
        
        norm2_i_gg = one_third * trace_gg * trace_gg
        grad_grad_output[b, h] = norm2_i_gg

        diff01 = gg01 - gg10
        diff02 = gg02 - gg20
        diff12 = gg12 - gg21
        norm2_a_gg = one_half * (diff01 * diff01 + diff02 * diff02 + diff12 * diff12)
        grad_grad_output[b, h + grad_grad_X.shape[3]] = norm2_a_gg

        sum01 = gg01 + gg10
        sum02 = gg02 + gg20
        sum12 = gg12 + gg21
        
        dev00 = gg00 - trace_third_gg
        dev11 = gg11 - trace_third_gg
        dev22 = gg22 - trace_third_gg
        
        norm2_s_gg = one_half * (sum01 * sum01 + sum02 * sum02 + sum12 * sum12)
        norm2_s_gg += dev00 * dev00 + dev11 * dev11 + dev22 * dev22
        grad_grad_output[b, h + 2 * grad_grad_X.shape[3]] = norm2_s_gg

    return (
        wp.Kernel(
            tensor_norm3_fwd,
            key=f"tensor_norm3_fwd_{dtype}",
            module=wp.get_module(f"tensor_norm3_fwd_{dtype}"),
        ),
        wp.Kernel(
            tensor_norm3_bwd,
            key=f"tensor_norm3_bwd_{dtype}",
            module=wp.get_module(f"tensor_norm3_bwd_{dtype}"),
        ),
        wp.Kernel(
            tensor_norm3_bwd_bwd,
            key=f"tensor_norm3_bwd_bwd_{dtype}",
            module=wp.get_module(f"tensor_norm3_bwd_bwd_{dtype}"),
        ),
    )


tensor_norm3_fwd_fp64, tensor_norm3_bwd_fp64, tensor_norm3_bwd_bwd_fp64 = (
    generate_tensor_norm3("float64")
)
tensor_norm3_fwd_fp32, tensor_norm3_bwd_fp32, tensor_norm3_bwd_bwd_fp32 = (
    generate_tensor_norm3("float32")
)
tensor_norm3_fwd_fp16, tensor_norm3_bwd_fp16, tensor_norm3_bwd_bwd_fp16 = (
    generate_tensor_norm3("float16")
)

add_module("tensor_norm3_fwd", ["float64"], tensor_norm3_fwd_fp64)
add_module("tensor_norm3_bwd", ["float64"], tensor_norm3_bwd_fp64)
add_module("tensor_norm3_bwd_bwd", ["float64"], tensor_norm3_bwd_bwd_fp64)

add_module("tensor_norm3_fwd", ["float32"], tensor_norm3_fwd_fp32)
add_module("tensor_norm3_bwd", ["float32"], tensor_norm3_bwd_fp32)
add_module("tensor_norm3_bwd_bwd", ["float32"], tensor_norm3_bwd_bwd_fp32)

add_module("tensor_norm3_fwd", ["float16"], tensor_norm3_fwd_fp16)
add_module("tensor_norm3_bwd", ["float16"], tensor_norm3_bwd_fp16)
add_module("tensor_norm3_bwd_bwd", ["float16"], tensor_norm3_bwd_bwd_fp16)
