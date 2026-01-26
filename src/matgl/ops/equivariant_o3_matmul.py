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
    "tensornet::tensor_matmul_o3_3x3_fwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(x: Tensor, y: Tensor) -> Tensor:
    if x.shape[1] != 3 or x.shape[2] != 3 or y.shape[1] != 3 or y.shape[2] != 3:
        raise ValueError("x and y must be 3x3 matrices")
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError("x and y must be 4D tensors")

    stream = get_stream(x.device)
    device = wp.device_from_torch(x.device)
    output = torch.empty_like(x)

    x_wp = wp.from_torch(x.detach(), return_ctype=True)
    y_wp = wp.from_torch(y.detach(), return_ctype=True)
    output_wp = wp.from_torch(output.detach(), return_ctype=True)

    tensor_matmul_o3_3x3_fwd = get_module("tensor_matmul_o3_3x3_fwd", [str(x.dtype)])
    wp.launch(
        tensor_matmul_o3_3x3_fwd,
        dim=(x.shape[0], x.shape[-1]),
        stream=stream,
        device=device,
        inputs=(x_wp, y_wp, output_wp),
    )

    return output


@torch.library.register_fake("tensornet::tensor_matmul_o3_3x3_fwd_primitive")
def _(x: Tensor, y: Tensor) -> Tensor:
    return torch.empty_like(x)


@torch.library.custom_op(
    "tensornet::tensor_matmul_o3_3x3_bwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(grad_output: Tensor, x: Tensor, y: Tensor) -> list[Tensor]:
    if x.shape[1] != 3 or x.shape[2] != 3 or y.shape[1] != 3 or y.shape[2] != 3:
        raise ValueError("x and y must be 3x3 matrices")
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError("x and y must be 4D tensors")

    stream = get_stream(x.device)
    device = wp.device_from_torch(x.device)
    grad_x = torch.empty_like(x)
    grad_y = torch.empty_like(y)

    grad_output_wp = wp.from_torch(grad_output.detach(), return_ctype=True)
    x_wp = wp.from_torch(x.detach(), return_ctype=True)
    y_wp = wp.from_torch(y.detach(), return_ctype=True)
    grad_x_wp = wp.from_torch(grad_x.detach(), return_ctype=True)
    grad_y_wp = wp.from_torch(grad_y.detach(), return_ctype=True)
    tensor_matmul_o3_3x3_bwd = get_module("tensor_matmul_o3_3x3_bwd", [str(x.dtype)])
    wp.launch(
        tensor_matmul_o3_3x3_bwd,
        dim=(x.shape[0], x.shape[-1]),
        stream=stream,
        device=device,
        inputs=(x_wp, y_wp, grad_output_wp, grad_x_wp, grad_y_wp),
    )

    return [grad_x, grad_y]


@torch.library.register_fake("tensornet::tensor_matmul_o3_3x3_bwd_primitive")
def _(grad_output: list[Tensor], x: Tensor, y: Tensor) -> list[Tensor]:
    return [torch.empty_like(x), torch.empty_like(y)]


@torch.library.custom_op(
    "tensornet::tensor_matmul_o3_3x3_bwd_bwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(grad_output: Tensor, grad_grad_x: Tensor, grad_grad_y: Tensor, x: Tensor, y: Tensor) -> list[Tensor]:
    if x.shape[1] != 3 or x.shape[2] != 3 or y.shape[1] != 3 or y.shape[2] != 3:
        raise ValueError("x and y must be 3x3 matrices")
    if x.ndim != 4 or y.ndim != 4:
        raise ValueError("x and y must be 4D tensors")

    stream = get_stream(grad_output.device)
    device = wp.device_from_torch(grad_output.device)
    grad_x = torch.empty_like(grad_output)
    grad_y = torch.empty_like(grad_output)

    grad_grad_output = torch.empty_like(grad_output)

    grad_grad_x_wp = wp.from_torch(grad_grad_x.detach(), return_ctype=True)
    grad_grad_y_wp = wp.from_torch(grad_grad_y.detach(), return_ctype=True)
    grad_output_wp = wp.from_torch(grad_output.detach(), return_ctype=True)
    x_wp = wp.from_torch(x.detach(), return_ctype=True)
    y_wp = wp.from_torch(y.detach(), return_ctype=True)

    grad_x_wp = wp.from_torch(grad_x.detach(), return_ctype=True)
    grad_y_wp = wp.from_torch(grad_y.detach(), return_ctype=True)
    grad_grad_output_wp = wp.from_torch(grad_grad_output.detach(), return_ctype=True)

    tensor_matmul_o3_3x3_bwd_bwd = get_module("tensor_matmul_o3_3x3_bwd_bwd", [str(grad_output.dtype)])
    wp.launch(
        tensor_matmul_o3_3x3_bwd_bwd,
        dim=(grad_output.shape[0], grad_output.shape[-1]),
        stream=stream,
        device=device,
        inputs=(
            x_wp,
            y_wp,
            grad_grad_x_wp,
            grad_grad_y_wp,
            grad_output_wp,
            grad_x_wp,
            grad_y_wp,
            grad_grad_output_wp,
        ),
    )

    return [grad_grad_output, grad_x, grad_y]


@torch.library.register_fake("tensornet::tensor_matmul_o3_3x3_bwd_bwd_primitive")
def _(grad_output: Tensor, grad_grad_x: Tensor, grad_grad_y: Tensor, x: Tensor, y: Tensor) -> list[Tensor]:
    return [
        torch.empty_like(grad_output),
        torch.empty_like(grad_output),
        torch.empty_like(grad_output),
    ]


def tensor_matmul_o3_3x3_setup_fwd_context(ctx, inputs, output):
    (x, y) = inputs
    ctx.save_for_backward(x, y)


def tensor_matmul_o3_3x3_setup_bwd_context(ctx, inputs, output):
    (grad_output, x, y) = inputs
    ctx.save_for_backward(grad_output, x, y)


@torch.compiler.allow_in_graph
def tensor_matmul_o3_3x3_fwd(*args):
    return torch.ops.tensornet.tensor_matmul_o3_3x3_fwd_primitive(*args)


@torch.compiler.allow_in_graph
def tensor_matmul_o3_3x3_bwd(ctx, grad_output):
    x, y = ctx.saved_tensors
    dx, dy = torch.ops.tensornet.tensor_matmul_o3_3x3_bwd_primitive(grad_output, x, y)
    return dx, dy


@torch.compiler.allow_in_graph
def tensor_matmul_o3_3x3_bwd_bwd(ctx, *grad_outputs):
    grad_grad_x = grad_outputs[0][0]
    grad_grad_y = grad_outputs[0][1]

    grad_output_saved, x, y = ctx.saved_tensors

    if grad_grad_x is None:
        grad_grad_x = torch.zeros_like(x)
    if grad_grad_y is None:
        grad_grad_y = torch.zeros_like(y)

    outputs = torch.ops.tensornet.tensor_matmul_o3_3x3_bwd_bwd_primitive(
        grad_output_saved, grad_grad_x, grad_grad_y, x, y
    )
    return outputs[0], outputs[1], outputs[2]


torch.library.register_autograd(
    "tensornet::tensor_matmul_o3_3x3_fwd_primitive",
    tensor_matmul_o3_3x3_bwd,
    setup_context=tensor_matmul_o3_3x3_setup_fwd_context,
)

torch.library.register_autograd(
    "tensornet::tensor_matmul_o3_3x3_bwd_primitive",
    tensor_matmul_o3_3x3_bwd_bwd,
    setup_context=tensor_matmul_o3_3x3_setup_bwd_context,
)


def fn_tensor_matmul_o3_3x3(x: Tensor, y: Tensor) -> Tensor:
    z = torch.ops.tensornet.tensor_matmul_o3_3x3_fwd_primitive(x, y)
    return z
