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

MODULES = {}


def get_module(name: str, dtype: list[str]):
    """Get the module for the given name and dtype."""
    full_name = name + "_" + "_".join(get_dtype(d) for d in dtype)
    if full_name not in MODULES:
        print(f"Module {full_name} not found in MODULES dictionary")
        print(f"Available modules: {list(MODULES.keys())}")
        raise ValueError(f"Module {full_name} not found")
    return MODULES[full_name]


def add_module(name: str, dtype: list[str], kernel: wp.Kernel):
    """Add the module for the given name and dtype."""
    full_name = name + "_" + "_".join(get_dtype(d) for d in dtype)
    if full_name not in MODULES:
        MODULES[full_name] = kernel
    return MODULES[full_name]


def get_dtype(dtype: str):
    """Get the dtype string representation for the given dtype (WIP)."""
    if dtype.endswith("16"):
        return "fp16"
    if dtype.endswith("32"):
        return "fp32"
    if dtype.endswith("64"):
        return "fp64"
    raise ValueError(f"Unsupported dtype: {dtype}")


def get_wp_fp_dtype(dtype: str):
    """Get the warp dtype for the given dtype (WIP)."""
    if dtype.endswith("16"):
        return wp.float16
    if dtype.endswith("32"):
        return wp.float32
    if dtype.endswith("64"):
        return wp.float64
    raise ValueError(f"Unsupported dtype: {dtype}")


def list_modules():
    """List all modules in the MODULES dictionary."""
    print("Available modules:")
    for name in MODULES:
        print(f"  - {name}")
    return list(MODULES.keys())


def get_stream(device: torch.device):
    """Get the stream for the given device."""
    if device.type == "cuda":
        return wp.stream_from_torch(torch.cuda.current_stream(device))
    return None
