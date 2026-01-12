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
"""Warp kernels for graph edge index transformation to sparse CSR format."""

import warp as wp


@wp.kernel
def count_row_col(
    edge_index: wp.array(ndim=2, dtype=wp.int32),
    row_count: wp.array(ndim=1, dtype=wp.int32),
    col_count: wp.array(ndim=1, dtype=wp.int32),
):
    tid = wp.tid()

    shift = edge_index.dtype(1)
    wp.atomic_add(row_count, edge_index[0, tid] + shift, wp.int32(1))
    wp.atomic_add(col_count, edge_index[1, tid] + shift, wp.int32(1))


@wp.kernel
def convert_to_sparse(
    edge_index: wp.array(ndim=2, dtype=wp.int32),
    row_count: wp.array(ndim=1, dtype=wp.int32),
    col_count: wp.array(ndim=1, dtype=wp.int32),
    row_indptr: wp.array(ndim=1, dtype=wp.int32),
    col_indptr: wp.array(ndim=1, dtype=wp.int32),
    row_indices: wp.array(ndim=1, dtype=wp.int32),
    col_indices: wp.array(ndim=1, dtype=wp.int32),
    row_data: wp.array(ndim=1, dtype=wp.int32),
    col_data: wp.array(ndim=1, dtype=wp.int32),
):
    tid = wp.tid()
    shift = edge_index.dtype(1)

    src_id = edge_index[0, tid]
    dst_id = edge_index[1, tid]

    src_cnt = wp.atomic_sub(row_count, src_id + shift, wp.int32(1))
    dst_cnt = wp.atomic_sub(col_count, dst_id + shift, wp.int32(1))

    row_indices[row_indptr[src_id + shift] - src_cnt] = dst_id
    row_data[row_indptr[src_id + shift] - src_cnt] = wp.int32(tid)

    col_indices[col_indptr[dst_id + shift] - dst_cnt] = src_id
    col_data[col_indptr[dst_id + shift] - dst_cnt] = wp.int32(tid)
