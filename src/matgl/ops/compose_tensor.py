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
    "tensornet::compose_tensor_fwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    stream = get_stream(x.device)
    device = wp.device_from_torch(x.device)
    output = torch.empty((x.shape[0], 3, 3, x.shape[-1]), dtype=x.dtype, device=x.device)

    x_wp = wp.from_torch(x.detach(), return_ctype=True)
    y_wp = wp.from_torch(y.detach(), return_ctype=True)
    z_wp = wp.from_torch(z.detach(), return_ctype=True)

    output_wp = wp.from_torch(output.detach(), return_ctype=True)

    compose_tensor_fwd = get_module("compose_tensor_fwd", [str(x.dtype)])
    wp.launch(
        compose_tensor_fwd,
        dim=(x.shape[0], x.shape[-1]),
        stream=stream,
        device=device,
        inputs=(x_wp, y_wp, z_wp, output_wp),
    )

    return output


@torch.library.register_fake("tensornet::compose_tensor_fwd_primitive")
def _(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    return torch.empty((z.shape[0], 3, 3, z.shape[-1]), dtype=x.dtype, device=x.device)


@torch.library.custom_op(
    "tensornet::compose_tensor_bwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(grad_output: Tensor, x: Tensor, y: Tensor, z: Tensor) -> list[Tensor]:
    stream = get_stream(x.device)
    device = wp.device_from_torch(x.device)
    grad_x = torch.zeros_like(x)
    grad_y = torch.zeros_like(y)
    grad_z = torch.zeros_like(z)

    grad_output_wp = wp.from_torch(grad_output.detach(), return_ctype=True)

    grad_x_wp = wp.from_torch(grad_x.detach(), return_ctype=True)
    grad_y_wp = wp.from_torch(grad_y.detach(), return_ctype=True)
    grad_z_wp = wp.from_torch(grad_z.detach(), return_ctype=True)

    compose_tensor_bwd = get_module("compose_tensor_bwd", [str(x.dtype)])
    wp.launch(
        compose_tensor_bwd,
        dim=(x.shape[0], x.shape[-1]),
        stream=stream,
        device=device,
        inputs=(grad_output_wp, grad_x_wp, grad_y_wp, grad_z_wp),
    )

    return [grad_x, grad_y, grad_z]


@torch.library.register_fake("tensornet::compose_tensor_bwd_primitive")
def _(grad_output: list[Tensor], x: Tensor, y: Tensor, z: Tensor) -> list[Tensor]:
    return [torch.empty_like(x), torch.empty_like(y), torch.empty_like(z)]


@torch.library.custom_op(
    "tensornet::compose_tensor_bwd_bwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(
    grad_output: Tensor,
    grad_grad_x: Tensor,
    grad_grad_y: Tensor,
    grad_grad_z: Tensor,
    x: Tensor,
    y: Tensor,
    z: Tensor,
) -> list[Tensor]:
    stream = get_stream(grad_output.device)
    device = wp.device_from_torch(grad_output.device)
    grad_x = torch.zeros_like(grad_grad_x)
    grad_y = torch.zeros_like(grad_grad_y)
    grad_z = torch.zeros_like(grad_grad_z)

    grad_grad_output = torch.zeros_like(grad_output)

    grad_grad_output_wp = wp.from_torch(grad_grad_output.detach(), return_ctype=True)

    grad_grad_x_wp = wp.from_torch(grad_grad_x.detach(), return_ctype=True)
    grad_grad_y_wp = wp.from_torch(grad_grad_y.detach(), return_ctype=True)
    grad_grad_z_wp = wp.from_torch(grad_grad_z.detach(), return_ctype=True)

    compose_tensor_bwd_bwd = get_module("compose_tensor_bwd_bwd", [str(x.dtype)])
    wp.launch(
        compose_tensor_bwd_bwd,
        dim=(x.shape[0], x.shape[-1]),
        stream=stream,
        device=device,
        inputs=(grad_grad_x_wp, grad_grad_y_wp, grad_grad_z_wp, grad_grad_output_wp),
    )

    return [grad_grad_output, grad_x, grad_y, grad_z]


@torch.library.register_fake("tensornet::compose_tensor_bwd_bwd_primitive")
def _(
    grad_output: Tensor,
    grad_grad_x: Tensor,
    grad_grad_y: Tensor,
    grad_grad_z: Tensor,
    x: Tensor,
    y: Tensor,
    z: Tensor,
) -> list[Tensor]:
    return [
        torch.empty_like(grad_output),
        torch.empty_like(x),
        torch.empty_like(y),
        torch.empty_like(z),
    ]


def compose_tensor_setup_fwd_context(ctx, inputs, output):
    (x, y, z) = inputs
    ctx.save_for_backward(x, y, z)


def compose_tensor_setup_bwd_context(ctx, inputs, output):
    (grad_output, x, y, z) = inputs
    ctx.save_for_backward(grad_output, x, y, z)


@torch.compiler.allow_in_graph
def compose_tensor_fwd(*args):
    return torch.ops.tensornet.compose_tensor_fwd_primitive(*args)


@torch.compiler.allow_in_graph
def compose_tensor_bwd(ctx, grad_output):
    x, y, z = ctx.saved_tensors
    dx, dy, dz = torch.ops.tensornet.compose_tensor_bwd_primitive(grad_output, x, y, z)
    return dx, dy, dz


@torch.compiler.allow_in_graph
def compose_tensor_bwd_bwd(ctx, *grad_outputs):
    grad_grad_x = grad_outputs[0][0]
    grad_grad_y = grad_outputs[0][1]
    grad_grad_z = grad_outputs[0][2]

    grad_output_saved, x, y, z = ctx.saved_tensors

    if grad_grad_x is None:
        grad_grad_x = torch.zeros_like(x)
    if grad_grad_y is None:
        grad_grad_y = torch.zeros_like(y)
    if grad_grad_z is None:
        grad_grad_z = torch.zeros_like(z)

    outputs = torch.ops.tensornet.compose_tensor_bwd_bwd_primitive(
        grad_output_saved, grad_grad_x, grad_grad_y, grad_grad_z, x, y, z
    )

    return outputs[0], outputs[1], outputs[2], outputs[3]


torch.library.register_autograd(
    "tensornet::compose_tensor_fwd_primitive",
    compose_tensor_bwd,
    setup_context=compose_tensor_setup_fwd_context,
)

torch.library.register_autograd(
    "tensornet::compose_tensor_bwd_primitive",
    compose_tensor_bwd_bwd,
    setup_context=compose_tensor_setup_bwd_context,
)


def fn_compose_tensor(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    output = torch.ops.tensornet.compose_tensor_fwd_primitive(x, y, z)
    return output
