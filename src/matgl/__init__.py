"""MatGL (Materials Graph Library) is a graph deep learning library for materials science."""

from __future__ import annotations

import typing
import warnings
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import torch

from .config import clear_cache
from .utils.io import get_available_pretrained_models, load_model

try:
    __version__: str = version("matgl")
except PackageNotFoundError:
    __version__ = "unknown"  # package not installed


# Default datatypes definitions

float_np = np.float32
float_th = torch.float32

int_np = np.int32
int_th = torch.int32

# Training entry points re-exported at the top level for convenience.
# Imported after the dtype defs above because the training module pulls in
# submodules that read ``matgl.float_th`` at import time, which would otherwise
# trip a circular import.
from .utils.training import MGLDatasetLoader, MGLPotentialTrainer  # noqa: E402

__all__ = [
    "MGLDatasetLoader",
    "MGLPotentialTrainer",
    "__version__",
    "clear_cache",
    "float_np",
    "float_th",
    "get_available_pretrained_models",
    "get_best_device",
    "int_np",
    "int_th",
    "load_model",
    "set_backend",
    "set_default_dtype",
]


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


def get_best_device() -> typing.Literal["cpu", "cuda", "mps"]:
    """Get the best device for the current system."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_backend(backend: typing.Literal["DGL", "PYG"] = "PYG") -> None:
    """Deprecated no-op stub retained for backwards compatibility.

    Earlier versions of matgl supported selecting a graph backend -- DGL or PyTorch
    Geometric -- through this function. The DGL backend has since been removed and
    matgl now uses PyTorch Geometric exclusively. This stub is kept so existing code
    that calls ``matgl.set_backend(...)`` continues to run without modification.

    Args:
        backend: Retained only for signature compatibility. Requesting ``"DGL"`` emits
            a ``DeprecationWarning``; the value is otherwise ignored and PyTorch
            Geometric is always used.
    """
    if str(backend).upper() == "DGL":
        warnings.warn(
            "The DGL backend no longer exists in matgl; PyTorch Geometric is now the "
            "only backend and is used regardless of this call. matgl.set_backend() is a "
            "no-op stub preserved for backwards compatibility.",
            DeprecationWarning,
            stacklevel=2,
        )
