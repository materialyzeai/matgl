"""JAX inference-path prototype for matgl TensorNet / QET (PyG backend).

Public entry points:

* :func:`~matgl_jax._convert.convert_potential` -- torch ``Potential`` -> JAX pytree.
* :func:`~matgl_jax._potential.make_potential_fn` -- jitted ``(E, forces, stress)`` fn.
* :class:`~matgl_jax._calculator.JAXPESCalculator` -- ASE calculator twin of
  ``matgl.ext.ase.PESCalculator``.

See ``jax_prototype/README.md`` for usage and the benchmark harness.
"""

from __future__ import annotations

from ._calculator import JAXPESCalculator
from ._convert import build_config, convert_potential, convert_qet, convert_tensornet
from ._potential import make_potential_fn
from ._qet import qet_energy
from ._tensornet import forward_features, tensornet_energy

__all__ = [
    "JAXPESCalculator",
    "build_config",
    "convert_potential",
    "convert_qet",
    "convert_tensornet",
    "forward_features",
    "make_potential_fn",
    "qet_energy",
    "tensornet_energy",
]
