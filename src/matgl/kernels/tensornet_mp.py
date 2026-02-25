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


def generate_message_passing(dtype: str):
    dtype_wp = get_wp_fp_dtype(dtype)

    class vec3(wp.types.vector(length=3, dtype=dtype_wp)):
        pass

    class vec5(wp.types.vector(length=5, dtype=dtype_wp)):
        pass

    def message_passing_fwd(
        I: wp.array(ndim=3, dtype=dtype_wp),
        A: wp.array(ndim=3, dtype=dtype_wp),
        S: wp.array(ndim=3, dtype=dtype_wp),
        edge_attr: wp.array(ndim=3, dtype=dtype_wp),
        row_data: wp.array(ndim=1, dtype=wp.int32),
        row_indices: wp.array(ndim=1, dtype=wp.int32),
        row_indptr: wp.array(ndim=1, dtype=wp.int32),
        output_I: wp.array(ndim=3, dtype=dtype_wp),
        output_A: wp.array(ndim=3, dtype=dtype_wp),
        output_S: wp.array(ndim=3, dtype=dtype_wp),
    ):
        b, h = wp.tid()

        output_I_reg = I.dtype(0)
        output_A_reg = vec3(I.dtype(0))
        output_S_reg = vec5(I.dtype(0))

        for i in range(row_indptr[b], row_indptr[b + 1]):
            idx_j = row_indices[i]
            idx_w = row_data[i]
            wI = edge_attr[idx_w, 0, h]
            wA = edge_attr[idx_w, 1, h]
            wS = edge_attr[idx_w, 2, h]

            output_I_reg += I[idx_j, 0, h] * wI
            for j in range(3):
                output_A_reg[j] += A[idx_j, j, h] * wA
            for j in range(5):
                output_S_reg[j] += S[idx_j, j, h] * wS

        output_I[b, 0, h] = output_I_reg
        for j in range(3):
            output_A[b, j, h] = output_A_reg[j]

        for j in range(5):
            output_S[b, j, h] = output_S_reg[j]

    def message_passing_bwd(
        I: wp.array(ndim=3, dtype=dtype_wp),
        A: wp.array(ndim=3, dtype=dtype_wp),
        S: wp.array(ndim=3, dtype=dtype_wp),
        edge_attr: wp.array(ndim=3, dtype=dtype_wp),
        doutput_I: wp.array(ndim=3, dtype=dtype_wp),
        doutput_A: wp.array(ndim=3, dtype=dtype_wp),
        doutput_S: wp.array(ndim=3, dtype=dtype_wp),
        col_data: wp.array(ndim=1, dtype=wp.int32),
        col_indices: wp.array(ndim=1, dtype=wp.int32),
        col_indptr: wp.array(ndim=1, dtype=wp.int32),
        dI: wp.array(ndim=3, dtype=dtype_wp),
        dA: wp.array(ndim=3, dtype=dtype_wp),
        dS: wp.array(ndim=3, dtype=dtype_wp),
        dedge_attr: wp.array(ndim=3, dtype=dtype_wp),
    ):
        b, h = wp.tid()

        dI_reg = I.dtype(0.0)
        dA_reg = vec3(I.dtype(0.0))
        dS_reg = vec5(I.dtype(0.0))

        for i in range(col_indptr[b], col_indptr[b + 1]):
            idx_j = col_indices[i]
            idx_w = col_data[i]

            wI = edge_attr[idx_w, 0, h]
            doutput_I_j = doutput_I[idx_j, 0, h]
            dI_reg += doutput_I_j * wI
            dedge_attr[idx_w, 0, h] = doutput_I_j * I[b, 0, h]

            # A
            wA = edge_attr[idx_w, 1, h]
            dweight_A = I.dtype(0.0)
            for j in range(3):
                dA_reg[j] += doutput_A[idx_j, j, h] * wA
                dweight_A += doutput_A[idx_j, j, h] * A[b, j, h]
            dedge_attr[idx_w, 1, h] = dweight_A

            # S
            wS = edge_attr[idx_w, 2, h]
            dweight_S = I.dtype(0.0)
            for j in range(5):
                dS_reg[j] += doutput_S[idx_j, j, h] * wS
                dweight_S += doutput_S[idx_j, j, h] * S[b, j, h]
            dedge_attr[idx_w, 2, h] = dweight_S

        dI[b, 0, h] = dI_reg
        for j in range(3):
            dA[b, j, h] = dA_reg[j]
        for j in range(5):
            dS[b, j, h] = dS_reg[j]

    def message_passing_edge_bwd_bwd(
        I: wp.array(ndim=3, dtype=dtype_wp),
        A: wp.array(ndim=3, dtype=dtype_wp),
        S: wp.array(ndim=3, dtype=dtype_wp),
        dI: wp.array(ndim=3, dtype=dtype_wp),
        dA: wp.array(ndim=3, dtype=dtype_wp),
        dS: wp.array(ndim=3, dtype=dtype_wp),
        dedge_attr: wp.array(ndim=3, dtype=dtype_wp),
        doutput_I: wp.array(ndim=3, dtype=dtype_wp),
        doutput_A: wp.array(ndim=3, dtype=dtype_wp),
        doutput_S: wp.array(ndim=3, dtype=dtype_wp),
        col_data: wp.array(ndim=1, dtype=wp.int32),
        col_indices: wp.array(ndim=1, dtype=wp.int32),
        col_indptr: wp.array(ndim=1, dtype=wp.int32),
        d2I: wp.array(ndim=3, dtype=dtype_wp),
        d2A: wp.array(ndim=3, dtype=dtype_wp),
        d2S: wp.array(ndim=3, dtype=dtype_wp),
        d2edge_attr: wp.array(ndim=3, dtype=dtype_wp),
    ):
        # Col-based iteration: b is source node, idx_j is destination node
        # Computes d2I, d2A, d2S, d2edge_attr - no atomics needed
        b, h = wp.tid()

        d2I_reg = I.dtype(0)
        d2A_reg = vec3(I.dtype(0))
        d2S_reg = vec5(I.dtype(0))

        for i in range(col_indptr[b], col_indptr[b + 1]):
            idx_j = col_indices[i]  # Destination node
            idx_w = col_data[i]

            dweight_I = dedge_attr[idx_w, 0, h]
            dweight_A = dedge_attr[idx_w, 1, h]
            dweight_S = dedge_attr[idx_w, 2, h]

            # d2I[b] = Σ dedge_attr[edge] * doutput_I[dst]
            d2I_reg += doutput_I[idx_j, 0, h] * dweight_I

            # d2edge_attr[edge] = dI[src] * doutput_I[dst]
            d2edge_attr[idx_w, 0, h] = doutput_I[idx_j, 0, h] * dI[b, 0, h]

            # A
            dweight_A_reg = I.dtype(0.0)
            for j in range(3):
                d2A_reg[j] += doutput_A[idx_j, j, h] * dweight_A
                dweight_A_reg += doutput_A[idx_j, j, h] * dA[b, j, h]
            d2edge_attr[idx_w, 1, h] = dweight_A_reg

            # S
            dweight_S_reg = I.dtype(0.0)
            for j in range(5):
                d2S_reg[j] += doutput_S[idx_j, j, h] * dweight_S
                dweight_S_reg += doutput_S[idx_j, j, h] * dS[b, j, h]
            d2edge_attr[idx_w, 2, h] = dweight_S_reg

        d2I[b, 0, h] = d2I_reg

        for j in range(3):
            d2A[b, j, h] = d2A_reg[j]

        for j in range(5):
            d2S[b, j, h] = d2S_reg[j]

    def message_passing_output_bwd_bwd(
        I: wp.array(ndim=3, dtype=dtype_wp),
        A: wp.array(ndim=3, dtype=dtype_wp),
        S: wp.array(ndim=3, dtype=dtype_wp),
        edge_attr: wp.array(ndim=3, dtype=dtype_wp),
        dI: wp.array(ndim=3, dtype=dtype_wp),
        dA: wp.array(ndim=3, dtype=dtype_wp),
        dS: wp.array(ndim=3, dtype=dtype_wp),
        dedge_attr: wp.array(ndim=3, dtype=dtype_wp),
        row_data: wp.array(ndim=1, dtype=wp.int32),
        row_indices: wp.array(ndim=1, dtype=wp.int32),
        row_indptr: wp.array(ndim=1, dtype=wp.int32),
        d2output_I: wp.array(ndim=3, dtype=dtype_wp),
        d2output_A: wp.array(ndim=3, dtype=dtype_wp),
        d2output_S: wp.array(ndim=3, dtype=dtype_wp),
    ):
        # Row-based iteration: b is destination node, idx_j is source node
        # Computes d2output_I, d2output_A, d2output_S - no atomics needed
        b, h = wp.tid()

        d2output_I_reg = I.dtype(0)
        d2output_A_reg = vec3(I.dtype(0))
        d2output_S_reg = vec5(I.dtype(0))

        for i in range(row_indptr[b], row_indptr[b + 1]):
            idx_j = row_indices[i]  # Source node
            idx_w = row_data[i]

            wI = edge_attr[idx_w, 0, h]
            wA = edge_attr[idx_w, 1, h]
            wS = edge_attr[idx_w, 2, h]

            dweight_I = dedge_attr[idx_w, 0, h]
            dweight_A = dedge_attr[idx_w, 1, h]
            dweight_S = dedge_attr[idx_w, 2, h]

            # d2output_I[b] = Σ (dI[src] * edge_attr + I[src] * dedge_attr)
            d2output_I_reg += dI[idx_j, 0, h] * wI
            d2output_I_reg += I[idx_j, 0, h] * dweight_I

            # A
            for j in range(3):
                d2output_A_reg[j] += dA[idx_j, j, h] * wA
                d2output_A_reg[j] += A[idx_j, j, h] * dweight_A

            # S
            for j in range(5):
                d2output_S_reg[j] += dS[idx_j, j, h] * wS
                d2output_S_reg[j] += S[idx_j, j, h] * dweight_S

        d2output_I[b, 0, h] = d2output_I_reg

        for j in range(3):
            d2output_A[b, j, h] = d2output_A_reg[j]

        for j in range(5):
            d2output_S[b, j, h] = d2output_S_reg[j]

    return (
        wp.Kernel(
            message_passing_fwd,
            key=f"message_passing_fwd_{dtype}",
            module=wp.get_module(f"message_passing_fwd_{dtype}"),
        ),
        wp.Kernel(
            message_passing_bwd,
            key=f"message_passing_bwd_{dtype}",
            module=wp.get_module(f"message_passing_bwd_{dtype}"),
        ),
        wp.Kernel(
            message_passing_edge_bwd_bwd,
            key=f"message_passing_edge_bwd_bwd_{dtype}",
            module=wp.get_module(f"message_passing_edge_bwd_bwd_{dtype}"),
        ),
        wp.Kernel(
            message_passing_output_bwd_bwd,
            key=f"message_passing_output_bwd_bwd_{dtype}",
            module=wp.get_module(f"message_passing_output_bwd_bwd_{dtype}"),
        ),
    )


(
    message_passing_fwd_fp64,
    message_passing_bwd_fp64,
    message_passing_edge_bwd_bwd_fp64,
    message_passing_output_bwd_bwd_fp64,
) = generate_message_passing("float64")
(
    message_passing_fwd_fp32,
    message_passing_bwd_fp32,
    message_passing_edge_bwd_bwd_fp32,
    message_passing_output_bwd_bwd_fp32,
) = generate_message_passing("float32")
(
    message_passing_fwd_fp16,
    message_passing_bwd_fp16,
    message_passing_edge_bwd_bwd_fp16,
    message_passing_output_bwd_bwd_fp16,
) = generate_message_passing("float16")

add_module("message_passing_fwd", ["float64"], message_passing_fwd_fp64)
add_module("message_passing_bwd", ["float64"], message_passing_bwd_fp64)
add_module("message_passing_edge_bwd_bwd", ["float64"], message_passing_edge_bwd_bwd_fp64)
add_module("message_passing_output_bwd_bwd", ["float64"], message_passing_output_bwd_bwd_fp64)

add_module("message_passing_fwd", ["float32"], message_passing_fwd_fp32)
add_module("message_passing_bwd", ["float32"], message_passing_bwd_fp32)
add_module("message_passing_edge_bwd_bwd", ["float32"], message_passing_edge_bwd_bwd_fp32)
add_module("message_passing_output_bwd_bwd", ["float32"], message_passing_output_bwd_bwd_fp32)

add_module("message_passing_fwd", ["float16"], message_passing_fwd_fp16)
add_module("message_passing_bwd", ["float16"], message_passing_bwd_fp16)
add_module("message_passing_edge_bwd_bwd", ["float16"], message_passing_edge_bwd_bwd_fp16)
add_module("message_passing_output_bwd_bwd", ["float16"], message_passing_output_bwd_bwd_fp16)
