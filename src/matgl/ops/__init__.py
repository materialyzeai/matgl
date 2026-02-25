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
"""Custom tensor operations with Warp kernel implementations."""

from __future__ import annotations

import warp as wp

from .compose_tensor import fn_compose_tensor
from .decompose_tensor import fn_decompose_tensor
from .equivariant_o3_matmul import fn_tensor_matmul_o3_3x3
from .equivariant_so3_matmul import fn_tensor_matmul_so3_3x3
from .graph_transform import graph_transform
from .tensor_norm3 import fn_tensor_norm3
from .tensornet_mp import fn_message_passing
from .tensornet_radial_mp import fn_radial_message_passing

wp.init()

__all__ = [
    "fn_compose_tensor",
    "fn_decompose_tensor",
    "fn_message_passing",
    "fn_radial_message_passing",
    "fn_radial_message_passing",
    "fn_tensor_matmul_o3_3x3",
    "fn_tensor_matmul_so3_3x3",
    "fn_tensor_norm3",
    "graph_transform",
]
