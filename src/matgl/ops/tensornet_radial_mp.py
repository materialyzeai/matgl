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
    "tensornet::radial_message_passing_fwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(edge_vec_norm: Tensor, edge_attr: Tensor, row_data: Tensor, row_indptr: Tensor) -> list[Tensor]:
    num_atoms = row_indptr.shape[0] - 1
    stream = get_stream(edge_vec_norm.device)
    device = wp.device_from_torch(edge_vec_norm.device)
    output_I = torch.zeros(
        (num_atoms, 1, edge_attr.shape[-1]),
        dtype=edge_vec_norm.dtype,
        device=edge_vec_norm.device,
    )
    output_A = torch.zeros(
        (num_atoms, 3, edge_attr.shape[-1]),
        dtype=edge_vec_norm.dtype,
        device=edge_vec_norm.device,
    )
    output_S = torch.zeros(
        (num_atoms, 5, edge_attr.shape[-1]),
        dtype=edge_vec_norm.dtype,
        device=edge_vec_norm.device,
    )

    output_I_wp = wp.from_torch(output_I.detach(), return_ctype=True)
    output_A_wp = wp.from_torch(output_A.detach(), return_ctype=True)
    output_S_wp = wp.from_torch(output_S.detach(), return_ctype=True)

    edge_vec_norm_wp = wp.from_torch(edge_vec_norm.detach(), return_ctype=True)
    edge_attr_wp = wp.from_torch(edge_attr.detach(), return_ctype=True)

    row_data_wp = wp.from_torch(row_data.detach(), return_ctype=True)
    row_indptr_wp = wp.from_torch(row_indptr.detach(), return_ctype=True)

    message_passing_fwd = get_module("radial_message_passing_fwd", [str(edge_vec_norm.dtype)])
    wp.launch(
        message_passing_fwd,
        dim=(num_atoms, edge_attr.shape[-1]),
        stream=stream,
        device=device,
        inputs=(
            edge_vec_norm_wp,
            edge_attr_wp,
            row_data_wp,
            row_indptr_wp,
            output_I_wp,
            output_A_wp,
            output_S_wp,
        ),
    )

    return [output_I, output_A, output_S]


@torch.library.register_fake("tensornet::radial_message_passing_fwd_primitive")
def _(edge_vec_norm: Tensor, edge_attr: Tensor, row_data: Tensor, row_indptr: Tensor) -> list[Tensor]:
    num_atoms = row_indptr.shape[0] - 1
    return [
        torch.empty(
            (num_atoms, 1, edge_attr.shape[-1]),
            dtype=edge_vec_norm.dtype,
            device=edge_vec_norm.device,
        ),
        torch.empty(
            (num_atoms, 3, edge_attr.shape[-1]),
            dtype=edge_vec_norm.dtype,
            device=edge_vec_norm.device,
        ),
        torch.empty(
            (num_atoms, 5, edge_attr.shape[-1]),
            dtype=edge_vec_norm.dtype,
            device=edge_vec_norm.device,
        ),
    ]


@torch.library.custom_op(
    "tensornet::radial_message_passing_bwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(
    grad_output_I: Tensor,
    grad_output_A: Tensor,
    grad_output_S: Tensor,
    edge_vec_norm: Tensor,
    edge_attr: Tensor,
    row_data: Tensor,
    row_indptr: Tensor,
) -> list[Tensor]:
    num_atoms = row_indptr.shape[0] - 1
    stream = get_stream(grad_output_I.device)
    device = wp.device_from_torch(grad_output_I.device)

    grad_output_I_wp = wp.from_torch(grad_output_I.detach(), return_ctype=True)
    grad_output_A_wp = wp.from_torch(grad_output_A.detach(), return_ctype=True)
    grad_output_S_wp = wp.from_torch(grad_output_S.detach(), return_ctype=True)

    grad_edge_vec_norm = torch.zeros_like(edge_vec_norm)
    grad_edge_vec_norm_wp = wp.from_torch(grad_edge_vec_norm.detach(), return_ctype=True)

    grad_edge_attr = torch.zeros_like(edge_attr)
    grad_edge_attr_wp = wp.from_torch(grad_edge_attr.detach(), return_ctype=True)

    edge_vec_norm_wp = wp.from_torch(edge_vec_norm.detach(), return_ctype=True)
    edge_attr_wp = wp.from_torch(edge_attr.detach(), return_ctype=True)

    row_data_wp = wp.from_torch(row_data.detach(), return_ctype=True)
    row_indptr_wp = wp.from_torch(row_indptr.detach(), return_ctype=True)

    message_passing_bwd = get_module("radial_message_passing_bwd", [str(edge_vec_norm.dtype)])
    wp.launch(
        message_passing_bwd,
        dim=(num_atoms, edge_attr.shape[-1]),
        stream=stream,
        device=device,
        inputs=(
            edge_vec_norm_wp,
            edge_attr_wp,
            row_data_wp,
            row_indptr_wp,
            grad_output_I_wp,
            grad_output_A_wp,
            grad_output_S_wp,
            grad_edge_vec_norm_wp,
            grad_edge_attr_wp,
        ),
    )

    return [grad_edge_vec_norm, grad_edge_attr]


@torch.library.register_fake("tensornet::radial_message_passing_bwd_primitive")
def _(
    grad_output_I: Tensor,
    grad_output_A: Tensor,
    grad_output_S: Tensor,
    edge_vec_norm: Tensor,
    edge_attr: Tensor,
    row_data: Tensor,
    row_indptr: Tensor,
) -> list[Tensor]:
    return [torch.empty_like(edge_vec_norm), torch.empty_like(edge_attr)]


@torch.library.custom_op(
    "tensornet::radial_message_passing_bwd_bwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(
    grad_output_I: Tensor,
    grad_output_A: Tensor,
    grad_output_S: Tensor,
    grad_grad_edge_vec_norm: Tensor,
    grad_grad_edge_attr: Tensor,
    edge_vec_norm: Tensor,
    edge_attr: Tensor,
    row_data: Tensor,
    row_indptr: Tensor,
) -> list[Tensor]:
    num_atoms = row_indptr.shape[0] - 1
    stream = get_stream(grad_output_I.device)
    device = wp.device_from_torch(grad_output_I.device)

    edge_vec_norm_wp = wp.from_torch(edge_vec_norm.detach(), return_ctype=True)
    edge_attr_wp = wp.from_torch(edge_attr.detach(), return_ctype=True)

    row_data_wp = wp.from_torch(row_data.detach(), return_ctype=True)
    row_indptr_wp = wp.from_torch(row_indptr.detach(), return_ctype=True)

    grad_grad_edge_vec_norm_wp = wp.from_torch(grad_grad_edge_vec_norm.detach(), return_ctype=True)
    grad_grad_edge_attr_wp = wp.from_torch(grad_grad_edge_attr.detach(), return_ctype=True)

    grad_output_I_wp = wp.from_torch(grad_output_I.detach(), return_ctype=True)
    grad_output_A_wp = wp.from_torch(grad_output_A.detach(), return_ctype=True)
    grad_output_S_wp = wp.from_torch(grad_output_S.detach(), return_ctype=True)
    dgrad_output_I = torch.zeros_like(grad_output_I)
    dgrad_output_A = torch.zeros_like(grad_output_A)
    dgrad_output_S = torch.zeros_like(grad_output_S)
    dgrad_output_I_wp = wp.from_torch(dgrad_output_I.detach(), return_ctype=True)
    dgrad_output_A_wp = wp.from_torch(dgrad_output_A.detach(), return_ctype=True)
    dgrad_output_S_wp = wp.from_torch(dgrad_output_S.detach(), return_ctype=True)

    dgrad_grad_edge_vec_norm = torch.zeros_like(grad_grad_edge_vec_norm)
    dgrad_grad_edge_vec_norm_wp = wp.from_torch(dgrad_grad_edge_vec_norm.detach(), return_ctype=True)

    dgrad_grad_edge_attr = torch.zeros_like(grad_grad_edge_attr)
    dgrad_grad_edge_attr_wp = wp.from_torch(dgrad_grad_edge_attr.detach(), return_ctype=True)

    message_passing_bwd_bwd = get_module("radial_message_passing_bwd_bwd", [str(edge_vec_norm.dtype)])
    wp.launch(
        message_passing_bwd_bwd,
        dim=(num_atoms, edge_attr.shape[-1]),
        stream=stream,
        device=device,
        inputs=(
            edge_vec_norm_wp,
            edge_attr_wp,
            grad_grad_edge_vec_norm_wp,
            grad_grad_edge_attr_wp,
            grad_output_I_wp,
            grad_output_A_wp,
            grad_output_S_wp,
            row_data_wp,
            row_indptr_wp,
            dgrad_grad_edge_vec_norm_wp,
            dgrad_grad_edge_attr_wp,
            dgrad_output_I_wp,
            dgrad_output_A_wp,
            dgrad_output_S_wp,
        ),
    )

    return [
        dgrad_output_I,
        dgrad_output_A,
        dgrad_output_S,
        dgrad_grad_edge_vec_norm,
        dgrad_grad_edge_attr,
    ]


@torch.library.register_fake("tensornet::radial_message_passing_bwd_bwd_primitive")
def _(
    grad_output_I: Tensor,
    grad_output_A: Tensor,
    grad_output_S: Tensor,
    grad_grad_edge_vec_norm: Tensor,
    grad_grad_edge_attr: Tensor,
    edge_vec_norm: Tensor,
    edge_attr: Tensor,
    row_data: Tensor,
    row_indptr: Tensor,
) -> list[Tensor]:
    return [
        torch.empty_like(grad_output_I),
        torch.empty_like(grad_output_A),
        torch.empty_like(grad_output_S),
        torch.empty_like(grad_grad_edge_vec_norm),
        torch.empty_like(grad_grad_edge_attr),
    ]


def radial_message_passing_setup_fwd_context(ctx, inputs, output):
    (edge_vec_norm, edge_attr, row_data, row_indptr) = inputs
    ctx.save_for_backward(edge_vec_norm, edge_attr, row_data, row_indptr)


def radial_message_passing_setup_bwd_context(ctx, inputs, output):
    (
        grad_output_I,
        grad_output_A,
        grad_output_S,
        edge_vec_norm,
        edge_attr,
        row_data,
        row_indptr,
    ) = inputs
    ctx.save_for_backward(
        grad_output_I,
        grad_output_A,
        grad_output_S,
        edge_vec_norm,
        edge_attr,
        row_data,
        row_indptr,
    )


@torch.compiler.allow_in_graph
def radial_message_passing_fwd(*args):
    return torch.ops.tensornet.radial_message_passing_fwd_primitive(*args)


@torch.compiler.allow_in_graph
def radial_message_passing_bwd(ctx, grad_outputs):
    edge_vec_norm, edge_attr, row_data, row_indptr = ctx.saved_tensors

    result = torch.ops.tensornet.radial_message_passing_bwd_primitive(
        grad_outputs[0],
        grad_outputs[1],
        grad_outputs[2],
        edge_vec_norm,
        edge_attr,
        row_data,
        row_indptr,
    )

    grad_edge_vec_norm, grad_edge_attr = result

    return grad_edge_vec_norm, grad_edge_attr, None, None


@torch.compiler.allow_in_graph
def radial_message_passing_bwd_bwd(ctx, *grad_outputs):
    grad_grad_edge_vec_norm, grad_grad_edge_attr = grad_outputs[0]
    (
        grad_output_I,
        grad_output_A,
        grad_output_S,
        edge_vec_norm,
        edge_attr,
        row_data,
        row_indptr,
    ) = ctx.saved_tensors

    result = torch.ops.tensornet.radial_message_passing_bwd_bwd_primitive(
        grad_output_I,
        grad_output_A,
        grad_output_S,
        grad_grad_edge_vec_norm,
        grad_grad_edge_attr,
        edge_vec_norm,
        edge_attr,
        row_data,
        row_indptr,
    )

    (
        dgrad_output_I,
        dgrad_output_A,
        dgrad_output_S,
        dgrad_grad_edge_vec_norm,
        dgrad_grad_edge_attr,
    ) = result

    return (
        dgrad_output_I,
        dgrad_output_A,
        dgrad_output_S,
        dgrad_grad_edge_vec_norm,
        dgrad_grad_edge_attr,
        None,
        None,
    )


torch.library.register_autograd(
    "tensornet::radial_message_passing_fwd_primitive",
    radial_message_passing_bwd,
    setup_context=radial_message_passing_setup_fwd_context,
)

torch.library.register_autograd(
    "tensornet::radial_message_passing_bwd_primitive",
    radial_message_passing_bwd_bwd,
    setup_context=radial_message_passing_setup_bwd_context,
)


def fn_radial_message_passing(
    edge_vec_norm: Tensor, edge_attr: Tensor, row_data: Tensor, row_indptr: Tensor
) -> list[Tensor]:
    return torch.ops.tensornet.radial_message_passing_fwd_primitive(edge_vec_norm, edge_attr, row_data, row_indptr)
