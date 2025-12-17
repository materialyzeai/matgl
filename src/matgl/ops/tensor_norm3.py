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

from typing import List

import torch
from torch import Tensor

import warp as wp

from matgl.kernels import get_module, get_stream


@torch.library.custom_op(
    "nvtensornet::tensor_norm3_fwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(x: Tensor) -> Tensor:

    stream = get_stream(x.device)
    device = wp.device_from_torch(x.device)
    output = torch.empty((x.shape[0], 3 * x.shape[-1]), dtype=x.dtype, device=x.device)

    x_wp = wp.from_torch(x.detach(), return_ctype=True)
    output_wp = wp.from_torch(output.detach(), return_ctype=True)

    tensor_norm3_fwd = get_module("tensor_norm3_fwd", [str(x.dtype)])
    wp.launch(
        tensor_norm3_fwd,
        dim=(x.shape[0], x.shape[-1]),
        stream=stream,
        device=device,
        inputs=(x_wp, output_wp),
    )

    return output


@torch.library.register_fake("nvtensornet::tensor_norm3_fwd_primitive")
def _(x: Tensor) -> Tensor:
    return torch.empty((x.shape[0], 3 * x.shape[-1]), dtype=x.dtype, device=x.device)


@torch.library.custom_op(
    "nvtensornet::tensor_norm3_bwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(
    grad_output: Tensor, x: Tensor
) -> List[Tensor]:

    stream = get_stream(x.device)
    device = wp.device_from_torch(x.device)
    grad_x = torch.empty_like(x)

    grad_output_wp = wp.from_torch(grad_output.detach(), return_ctype=True)
    x_wp = wp.from_torch(x.detach(), return_ctype=True)
    grad_x_wp = wp.from_torch(grad_x.detach(), return_ctype=True)

    tensor_norm3_bwd = get_module("tensor_norm3_bwd", [str(x.dtype)])
    wp.launch(
        tensor_norm3_bwd,
        dim=(x.shape[0], x.shape[-1]),
        stream=stream,
        device=device,
        inputs=(grad_output_wp, x_wp, grad_x_wp),
    )

    return [grad_x]


@torch.library.register_fake("nvtensornet::tensor_norm3_bwd_primitive")
def _(
    grad_output: Tensor, x: Tensor
) -> List[Tensor]:
    return [torch.empty_like(x)]


@torch.library.custom_op(
    "nvtensornet::tensor_norm3_bwd_bwd_primitive",
    mutates_args=(),
    device_types=["cpu", "cuda"],
)
def _(
    grad_grad_x: Tensor,
) -> Tensor:
    stream = get_stream(grad_grad_x.device)
    device = wp.device_from_torch(grad_grad_x.device)
    grad_grad_output = torch.empty((grad_grad_x.shape[0], 3 * grad_grad_x.shape[-1]), dtype=grad_grad_x.dtype, device=grad_grad_x.device)

    grad_grad_x_wp = wp.from_torch(grad_grad_x.detach(), return_ctype=True)
    grad_grad_output_wp = wp.from_torch(grad_grad_output.detach(), return_ctype=True)

    tensor_norm3_bwd_bwd = get_module("tensor_norm3_bwd_bwd", [str(grad_grad_x.dtype)])
    wp.launch(
        tensor_norm3_bwd_bwd,
        dim=(grad_grad_x.shape[0], grad_grad_x.shape[-1]),
        stream=stream,
        device=device,
        inputs=(
            grad_grad_x_wp,
            grad_grad_output_wp,
        ),
    )

    return grad_grad_output


@torch.library.register_fake("nvtensornet::tensor_norm3_bwd_bwd_primitive")
def _(
    grad_grad_x: Tensor,
) -> Tensor:
    return torch.empty((grad_grad_x.shape[0], 3 * grad_grad_x.shape[-1]), dtype=grad_grad_x.dtype, device=grad_grad_x.device)


def tensor_norm3_fwd_setup_context(ctx, inputs, output):
    (x,) = inputs  # Unpack the single input tensor
    ctx.save_for_backward(x)


def tensor_norm3_bwd_setup_context(ctx, inputs, output):
    (grad_output, x) = inputs
    ctx.save_for_backward(x)


@torch.compiler.allow_in_graph
def tensor_norm3_fwd(*args):
    return torch.ops.nvtensornet.tensor_norm3_fwd_primitive(*args)


@torch.compiler.allow_in_graph
def tensor_norm3_bwd(ctx, grad_output):
    (x,) = ctx.saved_tensors
    dx = torch.ops.nvtensornet.tensor_norm3_bwd_primitive(
        grad_output, x
    )
    return dx[0]


@torch.compiler.allow_in_graph
def tensor_norm3_bwd_bwd(ctx, grad_grad_x):
    (x,) = ctx.saved_tensors

    if grad_grad_x is None:
        grad_grad_x = torch.zeros_like(x)

    grad_grad_output = torch.ops.nvtensornet.tensor_norm3_bwd_bwd_primitive(
        grad_grad_x
    )

    return grad_grad_output


torch.library.register_autograd(
    "nvtensornet::tensor_norm3_fwd_primitive",
    tensor_norm3_bwd,
    setup_context=tensor_norm3_fwd_setup_context,
)

torch.library.register_autograd(
    "nvtensornet::tensor_norm3_bwd_primitive",
    tensor_norm3_bwd_bwd,
    setup_context=tensor_norm3_bwd_setup_context,
)


def fn_tensor_norm3(x: Tensor) -> Tensor:
    return torch.ops.nvtensornet.tensor_norm3_fwd_primitive(x)
