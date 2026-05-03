"""MatGL (Materials Graph Library) is a graph deep learning library for materials science."""

from __future__ import annotations

import typing
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch

import matgl

from .config import clear_cache, ensure_backend
from .utils.io import get_available_pretrained_models, load_model

if typing.TYPE_CHECKING:
    from typing import Literal

try:
    __version__: str = version("matgl")
except PackageNotFoundError:
    __version__ = "unknown"  # package not installed


# Default datatypes definitions

float_np = np.float32
float_th = torch.float32

int_np = np.int32
int_th = torch.int32


def set_default_dtype(type_: str = "float", size: int = 32) -> None:
    """Set the default dtype size (16, 32 or 64) for int or float used throughout matgl.

    Args:
        type_: "float" or "int"
        size: 32 or 64.
    """
    if size in (16, 32, 64):
        globals()[f"{type_}_th"] = getattr(torch, f"{type_}{size}")
        globals()[f"{type_}_np"] = getattr(np, f"{type_}{size}")
        torch.set_default_dtype(getattr(torch, f"float{size}"))
    else:
        raise ValueError("Invalid dtype size")
    if type_ == "float" and size == 16 and not torch.cuda.is_available():
        raise Exception(
            "torch.float16 is not supported for M3GNet because addmm_impl_cpu_ is not implemented"
            " for this floating precision, please use size = 32, 64 or using 'cuda' instead !!"
        )


def set_backend(backend: Literal["DGL", "PYG"] = "PYG") -> None:
    """Sets the computational backend for the application.

    This function allows you to set the backend used for computations, which could
    be either "DGL" (Deep Graph Library) or "PYG" (PyTorch Geometric). The selected
    backend determines how graph-based computations are implemented in the
    application. If an invalid backend is provided, a ValueError is raised.

    Parameters:
    backend: Literal["DGL", "PYG"]
        A string specifying the desired computational backend. Must be either
        "DGL" or "PYG".

    Raises:
    ValueError
        If the input backend is neither "DGL" nor "PYG".

    Returns:
    None
    """
    if backend not in ("DGL", "PYG"):
        raise ValueError("Invalid backend")
    ensure_backend(backend)
    matgl.config.BACKEND = backend


def get_best_device() -> Literal["cpu", "cuda", "mps"]:
    """Get the best device for the current system."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
