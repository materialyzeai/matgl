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
from __future__ import annotations

import torch
import warp as wp
from torch import Tensor

from matgl.kernels import convert_to_sparse, count_row_col, get_stream


@torch.library.custom_op(
    "nvtnet::count_row_col_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(edge_index: Tensor, num_nodes: int) -> tuple[Tensor, Tensor]:
    stream = get_stream(edge_index.device)
    device = wp.device_from_torch(edge_index.device)
    row_count = torch.zeros(num_nodes + 1, dtype=torch.int32, device=edge_index.device)
    col_count = torch.zeros(num_nodes + 1, dtype=torch.int32, device=edge_index.device)

    edge_index_wp = wp.from_torch(edge_index, return_ctype=True)
    row_count_wp = wp.from_torch(row_count, return_ctype=True)
    col_count_wp = wp.from_torch(col_count, return_ctype=True)

    wp.launch(
        count_row_col,
        dim=(edge_index.shape[1]),
        stream=stream,
        device=device,
        inputs=(edge_index_wp, row_count_wp, col_count_wp),
    )

    return row_count, col_count


@torch.library.register_fake("nvtnet::count_row_col_primitive")
def _(edge_index: Tensor, num_nodes: int) -> tuple[Tensor, Tensor]:
    output = torch.zeros(num_nodes + 1, dtype=torch.int32, device=edge_index.device)
    output2 = torch.zeros(num_nodes + 1, dtype=torch.int32, device=edge_index.device)
    return output, output2


@torch.library.custom_op(
    "nvtnet::convert_to_sparse_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(
    edge_index: Tensor,
    row_count: Tensor,
    col_count: Tensor,
    row_indptr: Tensor,
    col_indptr: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    stream = get_stream(edge_index.device)
    device = wp.device_from_torch(edge_index.device)
    edge_index_wp = wp.from_torch(edge_index, return_ctype=True)

    row_count_wp = wp.from_torch(row_count, return_ctype=True)
    col_count_wp = wp.from_torch(col_count, return_ctype=True)

    row_indptr_wp = wp.from_torch(row_indptr, return_ctype=True)
    col_indptr_wp = wp.from_torch(col_indptr, return_ctype=True)

    row_indices = torch.empty(edge_index.shape[1], dtype=torch.int32, device=edge_index.device)
    col_indices = torch.empty(edge_index.shape[1], dtype=torch.int32, device=edge_index.device)

    row_data = torch.empty(edge_index.shape[1], dtype=torch.int32, device=edge_index.device)
    col_data = torch.empty(edge_index.shape[1], dtype=torch.int32, device=edge_index.device)

    row_indices_wp = wp.from_torch(row_indices, return_ctype=True)
    col_indices_wp = wp.from_torch(col_indices, return_ctype=True)

    row_data_wp = wp.from_torch(row_data, return_ctype=True)
    col_data_wp = wp.from_torch(col_data, return_ctype=True)

    wp.launch(
        convert_to_sparse,
        dim=(edge_index.shape[1]),
        stream=stream,
        device=device,
        inputs=(
            edge_index_wp,
            row_count_wp,
            col_count_wp,
            row_indptr_wp,
            col_indptr_wp,
            row_indices_wp,
            col_indices_wp,
            row_data_wp,
            col_data_wp,
        ),
    )

    return row_indices, col_indices, row_data, col_data


@torch.library.register_fake("nvtnet::convert_to_sparse_primitive")
def _(
    edge_index: Tensor,
    row_count: Tensor,
    col_count: Tensor,
    row_indptr: Tensor,
    col_indptr: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    output = torch.empty(edge_index.shape[1], dtype=torch.int32, device=edge_index.device)
    output2 = torch.empty(edge_index.shape[1], dtype=torch.int32, device=edge_index.device)
    output3 = torch.empty(edge_index.shape[1], dtype=torch.int32, device=edge_index.device)
    output4 = torch.empty(edge_index.shape[1], dtype=torch.int32, device=edge_index.device)
    return output, output2, output3, output4


@torch.compiler.allow_in_graph
def graph_transform(edge_index: Tensor, num_nodes: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    row_count, col_count = torch.ops.nvtnet.count_row_col_primitive(edge_index, num_nodes)
    row_indptr, col_indptr = (
        torch.cumsum(row_count, dim=0, dtype=torch.int32),
        torch.cumsum(col_count, dim=0, dtype=torch.int32),
    )
    (
        row_indices,
        col_indices,
        row_data,
        col_data,
    ) = torch.ops.nvtnet.convert_to_sparse_primitive(edge_index, row_count, col_count, row_indptr, col_indptr)
    return row_data, row_indices, row_indptr, col_data, col_indices, col_indptr
