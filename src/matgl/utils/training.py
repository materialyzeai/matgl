"""Public re-exports for the unified MatGL training scaffolding.

The actual implementation lives in :mod:`matgl.utils._training`. That module branches
internally on ``matgl.config.BACKEND`` for the small number of methods that touch
backend-specific graph attributes; everything else is shared.
"""

from __future__ import annotations

from matgl.utils._training import (
    MatglLightningModuleMixin,
    ModelLightningModule,
    PotentialLightningModule,
    xavier_init,
)

__all__ = [
    "MatglLightningModuleMixin",
    "ModelLightningModule",
    "PotentialLightningModule",
    "xavier_init",
]
