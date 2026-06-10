"""JAX-accelerated inference for matgl TensorNet / QET (PyG backend).

This optional subpackage reimplements the inference path (energy + forces +
stress) of the PyG-backend ``TensorNet`` and ``QET`` models in JAX. A converted
model is JIT-compiled by XLA into a single fused program, giving a portable
(CPU / CUDA / Apple-Silicon) speedup over eager PyTorch.

It requires the optional ``jax`` dependency::

    pip install matgl[jax]

Public entry points:

* :func:`~matgl.ext.jax._convert.convert_potential` -- torch ``Potential`` -> JAX pytree.
* :func:`~matgl.ext.jax._potential.make_potential_fn` -- jitted ``(E, forces, stress)`` fn
  (optionally + QEq charges for QET).
* :class:`~matgl.ext.jax._calculator.JAXPESCalculator` -- ASE calculator, a twin of
  ``matgl.ext.ase.PESCalculator``.
"""

from __future__ import annotations

try:
    import jax
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "matgl.ext.jax requires the optional 'jax' dependency. Install it with: pip install matgl[jax]"
    ) from exc

from ._calculator import JAXPESCalculator
from ._convert import convert_potential
from ._potential import make_potential_fn

__all__ = ["JAXPESCalculator", "convert_potential", "make_potential_fn"]
