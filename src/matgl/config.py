"""Global configuration variables for matgl.

Physical and numerical constants (e.g. ``COULOMB_CONSTANT``,
``EV_PER_ANG3_TO_GPA``) live in :mod:`matgl.utils.constants`.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from pymatgen.core.periodic_table import Element

logger = logging.getLogger(__name__)

# Default set of elements supported by universal matgl models. Excludes radioactive and most artificial elements.
DEFAULT_ELEMENTS = tuple(el.symbol for el in Element if el.symbol not in ["Po", "At", "Rn", "Fr", "Ra"] and el.Z < 95)


# Default location of the cache for matgl, e.g., for storing downloaded models.
MATGL_CACHE = Path.home() / ".cache" / "matgl"
MATGL_CACHE.mkdir(parents=True, exist_ok=True)


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
            logger.warning("matgl cache dir %r not found", str(MATGL_CACHE))
