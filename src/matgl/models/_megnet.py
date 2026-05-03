"""Backward-compat shim: ``MEGNet`` moved to ``_megnet_dgl``.

Pretrained checkpoints saved with ``"@module": "matgl.models._megnet"`` rely on
this module path being importable. New code should import from
``matgl.models._megnet_dgl`` (or use the public re-export from
``matgl.models``).
"""

from __future__ import annotations

from matgl.models._megnet_dgl import MEGNet

__all__ = ["MEGNet"]
