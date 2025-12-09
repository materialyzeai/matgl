"""Global configuration variables for matgl."""

from __future__ import annotations

import importlib
import os
import shutil
from pathlib import Path
from typing import Literal

from pymatgen.core.periodic_table import Element

# Coulomb conversion
COULOMB_CONSTANT = 14.399645478425668

# Default set of elements supported by universal matgl models. Excludes radioactive and most artificial elements.
DEFAULT_ELEMENTS = tuple(el.symbol for el in Element if el.symbol not in ["Po", "At", "Rn", "Fr", "Ra"] and el.Z < 95)


# Default location of the cache for matgl, e.g., for storing downloaded models.
MATGL_CACHE = Path(os.path.expanduser("~")) / ".cache/matgl"
os.makedirs(MATGL_CACHE, exist_ok=True)

# Download url for pre-trained models.
PRETRAINED_MODELS_BASE_URL = "http://github.com/materialsvirtuallab/matgl/raw/main/pretrained_models/"

# Set the backend. Note that not all models are available for all backends.
BACKEND: Literal["PYG", "DGL"] = os.environ.get("MATGL_BACKEND", "PYG").upper()  # type: ignore[assignment,return-value]


def ensure_backend(backend: Literal["DGL", "PYG"]) -> None:
    """Validate that the requested backend is installed."""
    if backend == "DGL":
        try:
            importlib.util.find_spec("dgl")  # type: ignore[attr-defined]
        except ImportError as err:
            raise RuntimeError("Please install DGL to use this backend.") from err
    else:
        try:
            importlib.util.find_spec("torch_geometric")  # type: ignore[attr-defined]
        except ImportError as err:
            raise RuntimeError("Please install torch_geometric to use this backend.") from err


ensure_backend(BACKEND)


def clear_cache(confirm: bool = True) -> None:
    """Deletes all files in the matgl.cache. This is used to clean out downloaded models.

    Args:
        confirm: Whether to ask for confirmation. Default is True.
    """
    answer = "" if confirm else "y"
    while answer not in ("y", "n"):
        answer = input(f"Do you really want to delete everything in {MATGL_CACHE} (y|n)? ").lower().strip()
    if answer == "y":
        try:
            shutil.rmtree(MATGL_CACHE)
        except FileNotFoundError:
            print(f"matgl cache dir {MATGL_CACHE!r} not found")
