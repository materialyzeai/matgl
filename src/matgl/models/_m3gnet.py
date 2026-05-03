"""Backward-compat shim: ``M3GNet`` moved to ``_m3gnet_dgl``.

Pretrained checkpoints saved with ``"@module": "matgl.models._m3gnet"`` rely on
this module path being importable. New code should import from
``matgl.models._m3gnet_dgl`` (or use the public re-export from
``matgl.models``).
"""

from __future__ import annotations

from matgl.models._m3gnet_dgl import M3GNet

__all__ = ["M3GNet"]
