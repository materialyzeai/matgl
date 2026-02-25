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

from matgl.kernels import get_module, get_stream


@torch.library.custom_op(
    "tensornet::message_passing_fwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    edge_attr: Tensor,
    row_data: Tensor,
    row_indices: Tensor,
    row_indptr: Tensor,
    col_data: Tensor,
    col_indices: Tensor,
    col_indptr: Tensor,
) -> list[Tensor]:
    stream = get_stream(x.device)
    device = wp.device_from_torch(x.device)
    output_x = torch.empty_like(x)
    output_y = torch.empty_like(y)
    output_z = torch.empty_like(z)

    x_wp = wp.from_torch(x.detach(), return_ctype=True)
    y_wp = wp.from_torch(y.detach(), return_ctype=True)
    z_wp = wp.from_torch(z.detach(), return_ctype=True)

    output_x_wp = wp.from_torch(output_x.detach(), return_ctype=True)
    output_y_wp = wp.from_torch(output_y.detach(), return_ctype=True)
    output_z_wp = wp.from_torch(output_z.detach(), return_ctype=True)

    edge_attr_wp = wp.from_torch(edge_attr.detach(), return_ctype=True)

    row_data_wp = wp.from_torch(row_data.detach(), return_ctype=True)
    row_indices_wp = wp.from_torch(row_indices.detach(), return_ctype=True)
    row_indptr_wp = wp.from_torch(row_indptr.detach(), return_ctype=True)

    message_passing_fwd = get_module("message_passing_fwd", [str(x.dtype)])
    wp.launch(
        message_passing_fwd,
        dim=(x.shape[0], x.shape[-1]),
        stream=stream,
        device=device,
        inputs=(
            x_wp,
            y_wp,
            z_wp,
            edge_attr_wp,
            row_data_wp,
            row_indices_wp,
            row_indptr_wp,
            output_x_wp,
            output_y_wp,
            output_z_wp,
        ),
    )

    return [output_x, output_y, output_z]


@torch.library.register_fake("tensornet::message_passing_fwd_primitive")
def _(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    edge_attr: Tensor,
    row_data: Tensor,
    row_indices: Tensor,
    row_indptr: Tensor,
    col_data: Tensor,
    col_indices: Tensor,
    col_indptr: Tensor,
) -> list[Tensor]:
    return [torch.empty_like(x), torch.empty_like(y), torch.empty_like(z)]


@torch.library.custom_op(
    "tensornet::message_passing_bwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(
    grad_output_x: Tensor,
    grad_output_y: Tensor,
    grad_output_z: Tensor,
    x: Tensor,
    y: Tensor,
    z: Tensor,
    edge_attr: Tensor,
    row_data: Tensor,
    row_indices: Tensor,
    row_indptr: Tensor,
    col_data: Tensor,
    col_indices: Tensor,
    col_indptr: Tensor,
) -> list[Tensor]:
    stream = get_stream(x.device)
    device = wp.device_from_torch(x.device)
    grad_x = torch.empty_like(x)
    grad_y = torch.empty_like(y)
    grad_z = torch.empty_like(z)

    grad_edge_attr = torch.zeros_like(edge_attr)

    grad_output_x_wp = wp.from_torch(grad_output_x.detach(), return_ctype=True)
    grad_output_y_wp = wp.from_torch(grad_output_y.detach(), return_ctype=True)
    grad_output_z_wp = wp.from_torch(grad_output_z.detach(), return_ctype=True)

    x_wp = wp.from_torch(x.detach(), return_ctype=True)
    y_wp = wp.from_torch(y.detach(), return_ctype=True)
    z_wp = wp.from_torch(z.detach(), return_ctype=True)

    edge_attr_wp = wp.from_torch(edge_attr.detach(), return_ctype=True)

    col_data_wp = wp.from_torch(col_data.detach(), return_ctype=True)
    col_indices_wp = wp.from_torch(col_indices.detach(), return_ctype=True)
    col_indptr_wp = wp.from_torch(col_indptr.detach(), return_ctype=True)

    grad_x_wp = wp.from_torch(grad_x.detach(), return_ctype=True)
    grad_y_wp = wp.from_torch(grad_y.detach(), return_ctype=True)
    grad_z_wp = wp.from_torch(grad_z.detach(), return_ctype=True)
    grad_edge_attr_wp = wp.from_torch(grad_edge_attr.detach(), return_ctype=True)

    message_passing_bwd = get_module("message_passing_bwd", [str(x.dtype)])

    wp.launch(
        message_passing_bwd,
        dim=(x.shape[0], x.shape[-1]),
        stream=stream,
        device=device,
        inputs=(
            x_wp,
            y_wp,
            z_wp,
            edge_attr_wp,
            grad_output_x_wp,
            grad_output_y_wp,
            grad_output_z_wp,
            col_data_wp,
            col_indices_wp,
            col_indptr_wp,
            grad_x_wp,
            grad_y_wp,
            grad_z_wp,
            grad_edge_attr_wp,
        ),
    )

    return [grad_x, grad_y, grad_z, grad_edge_attr]


@torch.library.register_fake("tensornet::message_passing_bwd_primitive")
def _(
    grad_output_x: Tensor,
    grad_output_y: Tensor,
    grad_output_z: Tensor,
    x: Tensor,
    y: Tensor,
    z: Tensor,
    edge_attr: Tensor,
    row_data: Tensor,
    row_indices: Tensor,
    row_indptr: Tensor,
    col_data: Tensor,
    col_indices: Tensor,
    col_indptr: Tensor,
) -> list[Tensor]:
    return [
        torch.empty_like(x),
        torch.empty_like(y),
        torch.empty_like(z),
        torch.empty_like(edge_attr),
    ]


@torch.library.custom_op(
    "tensornet::message_passing_bwd_bwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(
    grad_output_x: Tensor,
    grad_output_y: Tensor,
    grad_output_z: Tensor,
    grad_grad_x: Tensor,
    grad_grad_y: Tensor,
    grad_grad_z: Tensor,
    grad_grad_edge_attr: Tensor,
    x: Tensor,
    y: Tensor,
    z: Tensor,
    edge_attr: Tensor,
    row_data: Tensor,
    row_indices: Tensor,
    row_indptr: Tensor,
    col_data: Tensor,
    col_indices: Tensor,
    col_indptr: Tensor,
) -> list[Tensor]:
    stream = get_stream(x.device)
    device = wp.device_from_torch(x.device)

    # Convert inputs to warp arrays
    x_wp = wp.from_torch(x.detach(), return_ctype=True)
    y_wp = wp.from_torch(y.detach(), return_ctype=True)
    z_wp = wp.from_torch(z.detach(), return_ctype=True)
    edge_attr_wp = wp.from_torch(edge_attr.detach(), return_ctype=True)

    grad_grad_x_wp = wp.from_torch(grad_grad_x.detach(), return_ctype=True)
    grad_grad_y_wp = wp.from_torch(grad_grad_y.detach(), return_ctype=True)
    grad_grad_z_wp = wp.from_torch(grad_grad_z.detach(), return_ctype=True)
    grad_grad_edge_attr_wp = wp.from_torch(grad_grad_edge_attr.detach(), return_ctype=True)

    grad_output_x_wp = wp.from_torch(grad_output_x.detach(), return_ctype=True)
    grad_output_y_wp = wp.from_torch(grad_output_y.detach(), return_ctype=True)
    grad_output_z_wp = wp.from_torch(grad_output_z.detach(), return_ctype=True)

    col_data_wp = wp.from_torch(col_data.detach(), return_ctype=True)
    col_indices_wp = wp.from_torch(col_indices.detach(), return_ctype=True)
    col_indptr_wp = wp.from_torch(col_indptr.detach(), return_ctype=True)

    row_data_wp = wp.from_torch(row_data.detach(), return_ctype=True)
    row_indices_wp = wp.from_torch(row_indices.detach(), return_ctype=True)
    row_indptr_wp = wp.from_torch(row_indptr.detach(), return_ctype=True)

    # Allocate output tensors (no zero-init needed with two-kernel approach)
    dgrad_x = torch.empty_like(x)
    dgrad_y = torch.empty_like(y)
    dgrad_z = torch.empty_like(z)
    dgrad_edge_attr = torch.empty_like(edge_attr)
    dgrad_output_x = torch.empty_like(grad_output_x)
    dgrad_output_y = torch.empty_like(grad_output_y)
    dgrad_output_z = torch.empty_like(grad_output_z)

    dgrad_x_wp = wp.from_torch(dgrad_x.detach(), return_ctype=True)
    dgrad_y_wp = wp.from_torch(dgrad_y.detach(), return_ctype=True)
    dgrad_z_wp = wp.from_torch(dgrad_z.detach(), return_ctype=True)
    dgrad_edge_attr_wp = wp.from_torch(dgrad_edge_attr.detach(), return_ctype=True)
    dgrad_output_x_wp = wp.from_torch(dgrad_output_x.detach(), return_ctype=True)
    dgrad_output_y_wp = wp.from_torch(dgrad_output_y.detach(), return_ctype=True)
    dgrad_output_z_wp = wp.from_torch(dgrad_output_z.detach(), return_ctype=True)

    # Kernel 1: col-based - computes d2I, d2A, d2S, d2edge_attr
    message_passing_edge_bwd_bwd = get_module("message_passing_edge_bwd_bwd", [str(x.dtype)])
    wp.launch(
        message_passing_edge_bwd_bwd,
        dim=(x.shape[0], x.shape[-1]),
        stream=stream,
        device=device,
        inputs=(
            x_wp,
            y_wp,
            z_wp,
            grad_grad_x_wp,
            grad_grad_y_wp,
            grad_grad_z_wp,
            grad_grad_edge_attr_wp,
            grad_output_x_wp,
            grad_output_y_wp,
            grad_output_z_wp,
            col_data_wp,
            col_indices_wp,
            col_indptr_wp,
            dgrad_x_wp,
            dgrad_y_wp,
            dgrad_z_wp,
            dgrad_edge_attr_wp,
        ),
    )

    # Kernel 2: row-based - computes d2output_I, d2output_A, d2output_S
    message_passing_output_bwd_bwd = get_module("message_passing_output_bwd_bwd", [str(x.dtype)])
    wp.launch(
        message_passing_output_bwd_bwd,
        dim=(x.shape[0], x.shape[-1]),
        stream=stream,
        device=device,
        inputs=(
            x_wp,
            y_wp,
            z_wp,
            edge_attr_wp,
            grad_grad_x_wp,
            grad_grad_y_wp,
            grad_grad_z_wp,
            grad_grad_edge_attr_wp,
            row_data_wp,
            row_indices_wp,
            row_indptr_wp,
            dgrad_output_x_wp,
            dgrad_output_y_wp,
            dgrad_output_z_wp,
        ),
    )

    return [
        dgrad_output_x,
        dgrad_output_y,
        dgrad_output_z,
        dgrad_x,
        dgrad_y,
        dgrad_z,
        dgrad_edge_attr,
    ]


@torch.library.register_fake("tensornet::message_passing_bwd_bwd_primitive")
def _(
    grad_output_x: Tensor,
    grad_output_y: Tensor,
    grad_output_z: Tensor,
    grad_grad_x: Tensor,
    grad_grad_y: Tensor,
    grad_grad_z: Tensor,
    grad_grad_edge_attr: Tensor,
    x: Tensor,
    y: Tensor,
    z: Tensor,
    edge_attr: Tensor,
    row_data: Tensor,
    row_indices: Tensor,
    row_indptr: Tensor,
    col_data: Tensor,
    col_indices: Tensor,
    col_indptr: Tensor,
) -> list[Tensor]:
    return [
        torch.empty_like(grad_output_x),
        torch.empty_like(grad_output_y),
        torch.empty_like(grad_output_z),
        torch.empty_like(grad_grad_x),
        torch.empty_like(grad_grad_y),
        torch.empty_like(grad_grad_z),
        torch.empty_like(grad_grad_edge_attr),
    ]


def message_passing_setup_fwd_context(ctx, inputs, output):
    (
        x,
        y,
        z,
        edge_attr,
        row_data,
        row_indices,
        row_indptr,
        col_data,
        col_indices,
        col_indptr,
    ) = inputs
    ctx.save_for_backward(
        x,
        y,
        z,
        edge_attr,
        row_data,
        row_indices,
        row_indptr,
        col_data,
        col_indices,
        col_indptr,
    )


def message_passing_setup_bwd_context(ctx, inputs, output):
    (
        grad_output_x,
        grad_output_y,
        grad_output_z,
        x,
        y,
        z,
        edge_attr,
        row_data,
        row_indices,
        row_indptr,
        col_data,
        col_indices,
        col_indptr,
    ) = inputs
    ctx.save_for_backward(
        grad_output_x,
        grad_output_y,
        grad_output_z,
        x,
        y,
        z,
        edge_attr,
        row_data,
        row_indices,
        row_indptr,
        col_data,
        col_indices,
        col_indptr,
    )


@torch.compiler.allow_in_graph
def message_passing_fwd(*args):
    return torch.ops.tensornet.message_passing_fwd_primitive(*args)


@torch.compiler.allow_in_graph
def message_passing_bwd(ctx, grad_outputs):
    (
        x,
        y,
        z,
        edge_attr,
        row_data,
        row_indices,
        row_indptr,
        col_data,
        col_indices,
        col_indptr,
    ) = ctx.saved_tensors

    result = torch.ops.tensornet.message_passing_bwd_primitive(
        grad_outputs[0],
        grad_outputs[1],
        grad_outputs[2],
        x,
        y,
        z,
        edge_attr,
        row_data,
        row_indices,
        row_indptr,
        col_data,
        col_indices,
        col_indptr,
    )

    grad_x, grad_y, grad_z, grad_edge_attr = result

    return grad_x, grad_y, grad_z, grad_edge_attr, None, None, None, None, None, None


@torch.compiler.allow_in_graph
def message_passing_bwd_bwd(ctx, *grad_outputs):
    grad_grad_x, grad_grad_y, grad_grad_z, grad_grad_edge_attr = grad_outputs[0]

    (
        grad_output_x,
        grad_output_y,
        grad_output_z,
        x,
        y,
        z,
        edge_attr,
        row_data,
        row_indices,
        row_indptr,
        col_data,
        col_indices,
        col_indptr,
    ) = ctx.saved_tensors

    result = torch.ops.tensornet.message_passing_bwd_bwd_primitive(
        grad_output_x,
        grad_output_y,
        grad_output_z,
        grad_grad_x,
        grad_grad_y,
        grad_grad_z,
        grad_grad_edge_attr,
        x,
        y,
        z,
        edge_attr,
        row_data,
        row_indices,
        row_indptr,
        col_data,
        col_indices,
        col_indptr,
    )

    return (
        result[0],
        result[1],
        result[2],
        result[3],
        result[4],
        result[5],
        result[6],
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "tensornet::message_passing_fwd_primitive",
    message_passing_bwd,
    setup_context=message_passing_setup_fwd_context,
)

torch.library.register_autograd(
    "tensornet::message_passing_bwd_primitive",
    message_passing_bwd_bwd,
    setup_context=message_passing_setup_bwd_context,
)


def fn_message_passing(
    x: Tensor,
    y: Tensor,
    z: Tensor,
    edge_attr: Tensor,
    row_data: Tensor,
    row_indices: Tensor,
    row_indptr: Tensor,
    col_data: Tensor,
    col_indices: Tensor,
    col_indptr: Tensor,
) -> list[Tensor]:
    return torch.ops.tensornet.message_passing_fwd_primitive(
        x,
        y,
        z,
        edge_attr,
        row_data,
        row_indices,
        row_indptr,
        col_data,
        col_indices,
        col_indptr,
    )
