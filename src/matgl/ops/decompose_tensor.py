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
    "tensornet::decompose_tensor_fwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(x: Tensor) -> list[Tensor]:
    stream = get_stream(x.device)
    device = wp.device_from_torch(x.device)
    output_i = torch.empty((x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=x.device)
    output_a = torch.empty((x.shape[0], 3, x.shape[-1]), dtype=x.dtype, device=x.device)
    output_s = torch.empty((x.shape[0], 5, x.shape[-1]), dtype=x.dtype, device=x.device)

    x_wp = wp.from_torch(x.detach(), return_ctype=True)
    output_i_wp = wp.from_torch(output_i.detach(), return_ctype=True)
    output_a_wp = wp.from_torch(output_a.detach(), return_ctype=True)
    output_s_wp = wp.from_torch(output_s.detach(), return_ctype=True)

    decompose_tensor_fwd = get_module("decompose_tensor_fwd", [str(x.dtype)])
    wp.launch(
        decompose_tensor_fwd,
        dim=(x.shape[0], x.shape[-1]),
        stream=stream,
        device=device,
        inputs=(x_wp, output_i_wp, output_a_wp, output_s_wp),
    )

    return [output_i, output_a, output_s]


@torch.library.register_fake("tensornet::decompose_tensor_fwd_primitive")
def _(x: Tensor) -> list[Tensor]:
    return [
        torch.empty((x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=x.device),
        torch.empty((x.shape[0], 3, x.shape[-1]), dtype=x.dtype, device=x.device),
        torch.empty((x.shape[0], 5, x.shape[-1]), dtype=x.dtype, device=x.device),
    ]


@torch.library.custom_op(
    "tensornet::decompose_tensor_bwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(grad_output_i: Tensor, grad_output_a: Tensor, grad_output_s: Tensor, x: Tensor) -> list[Tensor]:
    stream = get_stream(x.device)
    device = wp.device_from_torch(x.device)
    grad_x = torch.empty_like(x)

    grad_output_i_wp = wp.from_torch(grad_output_i.detach(), return_ctype=True)
    grad_output_a_wp = wp.from_torch(grad_output_a.detach(), return_ctype=True)
    grad_output_s_wp = wp.from_torch(grad_output_s.detach(), return_ctype=True)

    grad_x_wp = wp.from_torch(grad_x.detach(), return_ctype=True)

    decompose_tensor_bwd = get_module("decompose_tensor_bwd", [str(x.dtype)])
    wp.launch(
        decompose_tensor_bwd,
        dim=(x.shape[0], x.shape[-1]),
        stream=stream,
        device=device,
        inputs=(grad_output_i_wp, grad_output_a_wp, grad_output_s_wp, grad_x_wp),
    )

    return [grad_x]


@torch.library.register_fake("tensornet::decompose_tensor_bwd_primitive")
def _(grad_output_i: Tensor, grad_output_a: Tensor, grad_output_s: Tensor, x: Tensor) -> list[Tensor]:
    return [torch.empty_like(x)]


@torch.library.custom_op(
    "tensornet::decompose_tensor_bwd_bwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(
    grad_output_i: Tensor,
    grad_output_a: Tensor,
    grad_output_s: Tensor,
    grad_grad_x: Tensor,
    x: Tensor,
) -> list[Tensor]:
    stream = get_stream(grad_output_i.device)
    device = wp.device_from_torch(grad_output_i.device)
    grad_x = torch.zeros_like(grad_grad_x)

    grad_grad_output_i = torch.empty_like(grad_output_i)
    grad_grad_output_a = torch.empty_like(grad_output_a)
    grad_grad_output_s = torch.empty_like(grad_output_s)

    grad_grad_output_i_wp = wp.from_torch(grad_grad_output_i.detach(), return_ctype=True)
    grad_grad_output_a_wp = wp.from_torch(grad_grad_output_a.detach(), return_ctype=True)
    grad_grad_output_s_wp = wp.from_torch(grad_grad_output_s.detach(), return_ctype=True)

    grad_grad_x_wp = wp.from_torch(grad_grad_x.detach(), return_ctype=True)

    decompose_tensor_bwd_bwd = get_module("decompose_tensor_bwd_bwd", [str(x.dtype)])
    wp.launch(
        decompose_tensor_bwd_bwd,
        dim=(x.shape[0], x.shape[-1]),
        stream=stream,
        device=device,
        inputs=(
            grad_grad_x_wp,
            grad_grad_output_i_wp,
            grad_grad_output_a_wp,
            grad_grad_output_s_wp,
        ),
    )

    return [grad_grad_output_i, grad_grad_output_a, grad_grad_output_s, grad_x]


@torch.library.register_fake("tensornet::decompose_tensor_bwd_bwd_primitive")
def _(
    grad_output_i: Tensor,
    grad_output_a: Tensor,
    grad_output_s: Tensor,
    grad_grad_x: Tensor,
    x: Tensor,
) -> list[Tensor]:
    return [
        torch.empty_like(grad_output_i),
        torch.empty_like(grad_output_a),
        torch.empty_like(grad_output_s),
        torch.empty_like(grad_grad_x),
    ]


def decompose_tensor_setup_fwd_context(ctx, inputs, output):
    (x,) = inputs  # Unpack the single input tensor
    ctx.save_for_backward(x)


def decompose_tensor_setup_bwd_context(ctx, inputs, output):
    (grad_output_i, grad_output_a, grad_output_s, x) = inputs
    ctx.save_for_backward(grad_output_i, grad_output_a, grad_output_s, x)


@torch.compiler.allow_in_graph
def decompose_tensor_fwd(*args):
    return torch.ops.tensornet.decompose_tensor_fwd_primitive(*args)


@torch.compiler.allow_in_graph
def decompose_tensor_bwd(ctx, *grad_outputs):
    (x,) = ctx.saved_tensors
    grad_output_i, grad_output_a, grad_output_s = grad_outputs[0]
    dx = torch.ops.tensornet.decompose_tensor_bwd_primitive(grad_output_i, grad_output_a, grad_output_s, x)
    return dx[0]


@torch.compiler.allow_in_graph
def decompose_tensor_bwd_bwd(ctx, *grad_outputs):
    (grad_grad_x,) = grad_outputs[0]

    grad_output_i, grad_output_a, grad_output_s, x = ctx.saved_tensors

    if grad_grad_x is None:
        grad_grad_x = torch.zeros_like(x)

    outputs = torch.ops.tensornet.decompose_tensor_bwd_bwd_primitive(
        grad_output_i, grad_output_a, grad_output_s, grad_grad_x, x
    )

    return outputs[0], outputs[1], outputs[2], outputs[3]


torch.library.register_autograd(
    "tensornet::decompose_tensor_fwd_primitive",
    decompose_tensor_bwd,
    setup_context=decompose_tensor_setup_fwd_context,
)

torch.library.register_autograd(
    "tensornet::decompose_tensor_bwd_primitive",
    decompose_tensor_bwd_bwd,
    setup_context=decompose_tensor_setup_bwd_context,
)


def fn_decompose_tensor(x: Tensor) -> list[Tensor]:
    output = torch.ops.tensornet.decompose_tensor_fwd_primitive(x)
    return output
