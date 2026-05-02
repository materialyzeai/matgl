"""Package containing model implementations."""

from __future__ import annotations

from matgl.config import BACKEND

from ._core import MatGLModel

if BACKEND == "DGL":
    from ._chgnet import CHGNet
    from ._m3gnet import M3GNet
    from ._megnet import MEGNet
    from ._qet_dgl import QET
    from ._so3net import SO3Net
    from ._tensornet_dgl import TensorNet
else:
    from ._qet_pyg import QET  # type: ignore[assignment]
    from ._tensornet_pyg import TensorNet  # type: ignore[assignment]

from ._wrappers import TransformedTargetModel
