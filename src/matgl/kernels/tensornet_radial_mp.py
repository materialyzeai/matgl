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


def generate_radial_message_passing(dtype: str):
    dtype_wp = get_wp_fp_dtype(dtype)

    class vec3(wp.types.vector(length=3, dtype=dtype_wp)):
        pass

    class vec5(wp.types.vector(length=5, dtype=dtype_wp)):
        pass

    def radial_message_passing_fwd(
        edge_vec_norm: wp.array(ndim=2, dtype=dtype_wp),
        edge_attr: wp.array(ndim=3, dtype=dtype_wp),
        row_data: wp.array(ndim=1, dtype=wp.int32),
        row_indptr: wp.array(ndim=1, dtype=wp.int32),
        output_I: wp.array(ndim=3, dtype=dtype_wp),
        output_A: wp.array(ndim=3, dtype=dtype_wp),
        output_S: wp.array(ndim=3, dtype=dtype_wp),
    ):
        b, h = wp.tid()

        output_I_reg = output_I.dtype(0)
        output_A_reg = vec3(output_I.dtype(0))
        output_S_reg = vec5(output_I.dtype(0))

        for i in range(row_indptr[b], row_indptr[b + 1]):
            idx_w = row_data[i]

            weight_I_reg = edge_attr[idx_w, 0, h]
            weight_A_reg = edge_attr[idx_w, 1, h]
            weight_S_reg = edge_attr[idx_w, 2, h]

            r_ij = vec3(output_I.dtype(0))
            r_ij[0] = edge_vec_norm[idx_w, 0]
            r_ij[1] = edge_vec_norm[idx_w, 1]
            r_ij[2] = edge_vec_norm[idx_w, 2]

            output_I_reg += weight_I_reg

            output_A_reg[0] += r_ij[2] * weight_A_reg
            output_A_reg[1] += -r_ij[1] * weight_A_reg
            output_A_reg[2] += r_ij[0] * weight_A_reg

            S_reg = vec5()
            mean_r2 = (r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + r_ij[2] * r_ij[2]) / output_I.dtype(3.0)
            S_reg[0] = r_ij[0] * r_ij[0] - mean_r2
            S_reg[1] = r_ij[0] * r_ij[1]
            S_reg[2] = r_ij[0] * r_ij[2]
            S_reg[3] = r_ij[1] * r_ij[1] - mean_r2
            S_reg[4] = r_ij[1] * r_ij[2]

            output_S_reg[0] += S_reg[0] * weight_S_reg
            output_S_reg[1] += S_reg[1] * weight_S_reg
            output_S_reg[2] += S_reg[2] * weight_S_reg
            output_S_reg[3] += S_reg[3] * weight_S_reg
            output_S_reg[4] += S_reg[4] * weight_S_reg

        output_I[b, 0, h] = output_I_reg
        for i in range(3):
            output_A[b, i, h] = output_A_reg[i]

        for i in range(5):
            output_S[b, i, h] = output_S_reg[i]

    def radial_message_passing_bwd(
        edge_vec_norm: wp.array(ndim=2, dtype=dtype_wp),
        edge_attr: wp.array(ndim=3, dtype=dtype_wp),
        row_data: wp.array(ndim=1, dtype=wp.int32),
        row_indptr: wp.array(ndim=1, dtype=wp.int32),
        doutput_I: wp.array(ndim=3, dtype=dtype_wp),
        doutput_A: wp.array(ndim=3, dtype=dtype_wp),
        doutput_S: wp.array(ndim=3, dtype=dtype_wp),
        dedge_vec_norm: wp.array(ndim=2, dtype=dtype_wp),
        dedge_attr: wp.array(ndim=3, dtype=dtype_wp),
    ):
        b, h = wp.tid()

        doutput_I_reg = doutput_I[b, 0, h]
        doutput_A_reg = vec3()
        doutput_A_reg[0] = doutput_A[b, 0, h]
        doutput_A_reg[1] = doutput_A[b, 1, h]
        doutput_A_reg[2] = doutput_A[b, 2, h]

        doutput_S_reg = vec5()
        doutput_S_reg[0] = doutput_S[b, 0, h]
        doutput_S_reg[1] = doutput_S[b, 1, h]
        doutput_S_reg[2] = doutput_S[b, 2, h]
        doutput_S_reg[3] = doutput_S[b, 3, h]
        doutput_S_reg[4] = doutput_S[b, 4, h]

        for i in range(row_indptr[b], row_indptr[b + 1]):
            idx_w = row_data[i]

            edge_attr_A_reg = edge_attr[idx_w, 1, h]
            edge_attr_S_reg = edge_attr[idx_w, 2, h]

            r_ij = vec3(doutput_I.dtype(0))
            dr_ij = vec3(doutput_I.dtype(0))
            r_ij[0] = edge_vec_norm[idx_w, 0]
            r_ij[1] = edge_vec_norm[idx_w, 1]
            r_ij[2] = edge_vec_norm[idx_w, 2]

            dr_ij[2] += doutput_A_reg[0] * edge_attr_A_reg
            dr_ij[1] += -doutput_A_reg[1] * edge_attr_A_reg
            dr_ij[0] += doutput_A_reg[2] * edge_attr_A_reg

            dedge_attr_I = doutput_I_reg

            dedge_attr_A = doutput_A_reg[0] * r_ij[2] - doutput_A_reg[1] * r_ij[1] + doutput_A_reg[2] * r_ij[0]

            S_reg = vec5()
            mean_r2 = (r_ij[0] * r_ij[0] + r_ij[1] * r_ij[1] + r_ij[2] * r_ij[2]) / doutput_I.dtype(3.0)
            S_reg[0] = r_ij[0] * r_ij[0] - mean_r2
            S_reg[1] = r_ij[0] * r_ij[1]
            S_reg[2] = r_ij[0] * r_ij[2]
            S_reg[3] = r_ij[1] * r_ij[1] - mean_r2
            S_reg[4] = r_ij[1] * r_ij[2]

            dedge_attr_S = (S_reg[0]) * doutput_S_reg[0]
            dedge_attr_S += (S_reg[1]) * doutput_S_reg[1]
            dedge_attr_S += (S_reg[2]) * doutput_S_reg[2]
            dedge_attr_S += (S_reg[3]) * doutput_S_reg[3]
            dedge_attr_S += (S_reg[4]) * doutput_S_reg[4]

            dS_reg = vec5()
            dS_reg[0] = edge_attr_S_reg * doutput_S_reg[0]
            dS_reg[1] = edge_attr_S_reg * doutput_S_reg[1]
            dS_reg[2] = edge_attr_S_reg * doutput_S_reg[2]
            dS_reg[3] = edge_attr_S_reg * doutput_S_reg[3]
            dS_reg[4] = edge_attr_S_reg * doutput_S_reg[4]

            dr_ij[0] += (
                dS_reg[0] * (doutput_I.dtype(4.0) / doutput_I.dtype(3.0) * r_ij[0])
                + dS_reg[1] * r_ij[1]
                + dS_reg[2] * r_ij[2]
                + dS_reg[3] * (-doutput_I.dtype(2.0) / doutput_I.dtype(3.0) * r_ij[0])
            )
            dr_ij[1] += (
                dS_reg[0] * (-doutput_I.dtype(2.0) / doutput_I.dtype(3.0) * r_ij[1])
                + dS_reg[1] * r_ij[0]
                + dS_reg[3] * (doutput_I.dtype(4.0) / doutput_I.dtype(3.0) * r_ij[1])
                + dS_reg[4] * r_ij[2]
            )
            dr_ij[2] += (
                dS_reg[0] * (-doutput_I.dtype(2.0) / doutput_I.dtype(3.0) * r_ij[2])
                + dS_reg[2] * r_ij[0]
                + dS_reg[3] * (-doutput_I.dtype(2.0) / doutput_I.dtype(3.0) * r_ij[2])
                + dS_reg[4] * r_ij[1]
            )

            wp.atomic_add(dedge_attr, idx_w, 0, h, dedge_attr_I)
            wp.atomic_add(dedge_attr, idx_w, 1, h, dedge_attr_A)
            wp.atomic_add(dedge_attr, idx_w, 2, h, dedge_attr_S)

            wp.atomic_add(dedge_vec_norm, idx_w, 0, dr_ij[0])
            wp.atomic_add(dedge_vec_norm, idx_w, 1, dr_ij[1])
            wp.atomic_add(dedge_vec_norm, idx_w, 2, dr_ij[2])

    def radial_message_passing_bwd_bwd(
        edge_vec_norm: wp.array(ndim=2, dtype=dtype_wp),
        edge_attr: wp.array(ndim=3, dtype=dtype_wp),
        dedge_vec_norm: wp.array(ndim=2, dtype=dtype_wp),
        dedge_attr: wp.array(ndim=3, dtype=dtype_wp),
        doutput_I: wp.array(ndim=3, dtype=dtype_wp),
        doutput_A: wp.array(ndim=3, dtype=dtype_wp),
        doutput_S: wp.array(ndim=3, dtype=dtype_wp),
        row_data: wp.array(ndim=1, dtype=wp.int32),
        row_indptr: wp.array(ndim=1, dtype=wp.int32),
        d2edge_vec_norm: wp.array(ndim=2, dtype=dtype_wp),
        d2edge_attr: wp.array(ndim=3, dtype=dtype_wp),
        d2output_I: wp.array(ndim=3, dtype=dtype_wp),
        d2output_A: wp.array(ndim=3, dtype=dtype_wp),
        d2output_S: wp.array(ndim=3, dtype=dtype_wp),
    ):
        b, h = wp.tid()

        d2output_I_reg = d2output_I.dtype(0.0)
        d2output_A_reg = vec3()
        d2output_S_reg = vec5()

        for i in range(row_indptr[b], row_indptr[b + 1]):
            idx_w = row_data[i]
            edge_attr_A_reg = edge_attr[idx_w, 1, h]
            edge_attr_S_reg = edge_attr[idx_w, 2, h]

            dedge_attr_I = dedge_attr[idx_w, 0, h]
            dedge_attr_A = dedge_attr[idx_w, 1, h]
            dedge_attr_S = dedge_attr[idx_w, 2, h]

            r_ij = vec3(d2output_I.dtype(0))
            dr_ij = vec3(d2output_I.dtype(0))
            for j in range(3):
                r_ij[j] = edge_vec_norm[idx_w, j]
                dr_ij[j] = dedge_vec_norm[idx_w, j]

            d2output_I_reg += dedge_attr_I

            d2r_ij = vec3(d2output_I.dtype(0))

            # No gradient contribution for edge_attr[*, 0, h] in forward pass
            # d2edge_attr[idx_w, 0, h] = d2output_I.dtype(0.0)

            d2output_A_reg[0] += dr_ij[2] * edge_attr_A_reg
            d2output_A_reg[1] += -dr_ij[1] * edge_attr_A_reg
            d2output_A_reg[2] += dr_ij[0] * edge_attr_A_reg

            d2output_A_reg[0] += dedge_attr_A * r_ij[2]
            d2output_A_reg[1] += -dedge_attr_A * r_ij[1]
            d2output_A_reg[2] += dedge_attr_A * r_ij[0]

            dweight_A = doutput_A[b, 0, h] * dr_ij[2] - doutput_A[b, 1, h] * dr_ij[1] + doutput_A[b, 2, h] * dr_ij[0]

            d2r_ij[2] += dedge_attr_A * doutput_A[b, 0, h]
            d2r_ij[1] += -dedge_attr_A * doutput_A[b, 1, h]
            d2r_ij[0] += dedge_attr_A * doutput_A[b, 2, h]

            wp.atomic_add(d2edge_attr, idx_w, 1, h, dweight_A)

            c0 = doutput_S.dtype(4.0) / doutput_S.dtype(3.0)
            c1 = -doutput_S.dtype(2.0) / doutput_S.dtype(3.0)

            c2 = doutput_S.dtype(2.0) / doutput_S.dtype(3.0)
            c3 = -doutput_S.dtype(1.0) / doutput_S.dtype(3.0)

            d2output_S_reg[0] += edge_attr_S_reg * (
                dedge_vec_norm[idx_w, 0] * c0 * r_ij[0]
                + dedge_vec_norm[idx_w, 1] * c1 * r_ij[1]
                + dedge_vec_norm[idx_w, 2] * c1 * r_ij[2]
            )
            d2output_S_reg[0] += dedge_attr_S * (
                c2 * r_ij[0] * r_ij[0] + c3 * r_ij[1] * r_ij[1] + c3 * r_ij[2] * r_ij[2]
            )

            d2output_S_reg[1] += edge_attr_S_reg * (
                dedge_vec_norm[idx_w, 0] * r_ij[1] + dedge_vec_norm[idx_w, 1] * r_ij[0]
            )
            d2output_S_reg[1] += dedge_attr_S * (r_ij[1] * r_ij[0])

            d2output_S_reg[2] += edge_attr_S_reg * (
                dedge_vec_norm[idx_w, 0] * r_ij[2] + dedge_vec_norm[idx_w, 2] * r_ij[0]
            )
            d2output_S_reg[2] += dedge_attr_S * (r_ij[2] * r_ij[0])

            d2output_S_reg[3] += edge_attr_S_reg * (
                dedge_vec_norm[idx_w, 0] * c1 * r_ij[0]
                + dedge_vec_norm[idx_w, 1] * c0 * r_ij[1]
                + dedge_vec_norm[idx_w, 2] * c1 * r_ij[2]
            )
            d2output_S_reg[3] += dedge_attr_S * (
                c3 * r_ij[0] * r_ij[0] + c2 * r_ij[1] * r_ij[1] + c3 * r_ij[2] * r_ij[2]
            )

            d2output_S_reg[4] += edge_attr_S_reg * (
                dedge_vec_norm[idx_w, 1] * r_ij[2] + dedge_vec_norm[idx_w, 2] * r_ij[1]
            )
            d2output_S_reg[4] += dedge_attr_S * (r_ij[2] * r_ij[1])

            d2r_ij[0] += doutput_S[b, 0, h] * edge_attr_S_reg * (dedge_vec_norm[idx_w, 0] * c0)
            d2r_ij[1] += doutput_S[b, 0, h] * edge_attr_S_reg * (dedge_vec_norm[idx_w, 1] * c1)
            d2r_ij[2] += doutput_S[b, 0, h] * edge_attr_S_reg * (dedge_vec_norm[idx_w, 2] * c1)

            d2r_ij[0] += doutput_S[b, 0, h] * dedge_attr_S * (c0 * r_ij[0])
            d2r_ij[1] += doutput_S[b, 0, h] * dedge_attr_S * (c1 * r_ij[1])
            d2r_ij[2] += doutput_S[b, 0, h] * dedge_attr_S * (c1 * r_ij[2])

            d2r_ij[0] += doutput_S[b, 1, h] * edge_attr_S_reg * (dedge_vec_norm[idx_w, 1])
            d2r_ij[1] += doutput_S[b, 1, h] * edge_attr_S_reg * (dedge_vec_norm[idx_w, 0])

            d2r_ij[0] += doutput_S[b, 1, h] * dedge_attr_S * (r_ij[1])
            d2r_ij[1] += doutput_S[b, 1, h] * dedge_attr_S * (r_ij[0])

            d2r_ij[0] += doutput_S[b, 2, h] * edge_attr_S_reg * (dedge_vec_norm[idx_w, 2])
            d2r_ij[2] += doutput_S[b, 2, h] * edge_attr_S_reg * (dedge_vec_norm[idx_w, 0])

            d2r_ij[0] += doutput_S[b, 2, h] * dedge_attr_S * (r_ij[2])
            d2r_ij[2] += doutput_S[b, 2, h] * dedge_attr_S * (r_ij[0])

            d2r_ij[0] += doutput_S[b, 3, h] * edge_attr_S_reg * (dedge_vec_norm[idx_w, 0] * c1)
            d2r_ij[1] += doutput_S[b, 3, h] * edge_attr_S_reg * (dedge_vec_norm[idx_w, 1] * c0)
            d2r_ij[2] += doutput_S[b, 3, h] * edge_attr_S_reg * (dedge_vec_norm[idx_w, 2] * c1)

            d2r_ij[0] += doutput_S[b, 3, h] * dedge_attr_S * (c1 * r_ij[0])
            d2r_ij[1] += doutput_S[b, 3, h] * dedge_attr_S * (c0 * r_ij[1])
            d2r_ij[2] += doutput_S[b, 3, h] * dedge_attr_S * (c1 * r_ij[2])

            d2r_ij[1] += doutput_S[b, 4, h] * edge_attr_S_reg * (dedge_vec_norm[idx_w, 2])
            d2r_ij[2] += doutput_S[b, 4, h] * edge_attr_S_reg * (dedge_vec_norm[idx_w, 1])

            d2r_ij[1] += doutput_S[b, 4, h] * dedge_attr_S * (r_ij[2])
            d2r_ij[2] += doutput_S[b, 4, h] * dedge_attr_S * (r_ij[1])

            d2weight_S = doutput_S.dtype(0.0)
            d2weight_S += doutput_S[b, 0, h] * (
                c0 * r_ij[0] * dedge_vec_norm[idx_w, 0]
                + c1 * r_ij[1] * dedge_vec_norm[idx_w, 1]
                + c1 * r_ij[2] * dedge_vec_norm[idx_w, 2]
            )

            d2weight_S += doutput_S[b, 1, h] * (r_ij[1] * dedge_vec_norm[idx_w, 0] + r_ij[0] * dedge_vec_norm[idx_w, 1])

            d2weight_S += doutput_S[b, 2, h] * (r_ij[2] * dedge_vec_norm[idx_w, 0] + r_ij[0] * dedge_vec_norm[idx_w, 2])

            d2weight_S += doutput_S[b, 3, h] * (
                c1 * r_ij[0] * dedge_vec_norm[idx_w, 0]
                + c0 * r_ij[1] * dedge_vec_norm[idx_w, 1]
                + c1 * r_ij[2] * dedge_vec_norm[idx_w, 2]
            )

            d2weight_S += doutput_S[b, 4, h] * (r_ij[2] * dedge_vec_norm[idx_w, 1] + r_ij[1] * dedge_vec_norm[idx_w, 2])

            wp.atomic_add(d2edge_attr, idx_w, 2, h, d2weight_S)

            wp.atomic_add(d2edge_vec_norm, idx_w, 0, d2r_ij[0])
            wp.atomic_add(d2edge_vec_norm, idx_w, 1, d2r_ij[1])
            wp.atomic_add(d2edge_vec_norm, idx_w, 2, d2r_ij[2])

        d2output_I[b, 0, h] = d2output_I_reg

        for i in range(3):
            d2output_A[b, i, h] = d2output_A_reg[i]

        for i in range(5):
            d2output_S[b, i, h] = d2output_S_reg[i]

    return (
        wp.Kernel(
            radial_message_passing_fwd,
            key=f"radial_message_passing_fwd_{dtype}",
            module=wp.get_module(f"radial_message_passing_fwd_{dtype}"),
        ),
        wp.Kernel(
            radial_message_passing_bwd,
            key=f"radial_message_passing_bwd_{dtype}",
            module=wp.get_module(f"radial_message_passing_bwd_{dtype}"),
        ),
        wp.Kernel(
            radial_message_passing_bwd_bwd,
            key=f"radial_message_passing_bwd_bwd_{dtype}",
            module=wp.get_module(f"radial_message_passing_bwd_bwd_{dtype}"),
        ),
    )


(
    radial_message_passing_fwd_fp64,
    radial_message_passing_bwd_fp64,
    radial_message_passing_bwd_bwd_fp64,
) = generate_radial_message_passing("float64")
(
    radial_message_passing_fwd_fp32,
    radial_message_passing_bwd_fp32,
    radial_message_passing_bwd_bwd_fp32,
) = generate_radial_message_passing("float32")
(
    radial_message_passing_fwd_fp16,
    radial_message_passing_bwd_fp16,
    radial_message_passing_bwd_bwd_fp16,
) = generate_radial_message_passing("float16")

add_module("radial_message_passing_fwd", ["float64"], radial_message_passing_fwd_fp64)
add_module("radial_message_passing_bwd", ["float64"], radial_message_passing_bwd_fp64)
add_module("radial_message_passing_bwd_bwd", ["float64"], radial_message_passing_bwd_bwd_fp64)

add_module("radial_message_passing_fwd", ["float32"], radial_message_passing_fwd_fp32)
add_module("radial_message_passing_bwd", ["float32"], radial_message_passing_bwd_fp32)
add_module("radial_message_passing_bwd_bwd", ["float32"], radial_message_passing_bwd_bwd_fp32)

add_module("radial_message_passing_fwd", ["float16"], radial_message_passing_fwd_fp16)
add_module("radial_message_passing_bwd", ["float16"], radial_message_passing_bwd_fp16)
add_module("radial_message_passing_bwd_bwd", ["float16"], radial_message_passing_bwd_bwd_fp16)
