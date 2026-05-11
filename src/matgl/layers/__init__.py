"""Reusable building blocks for matgl graph neural networks.

Models in :mod:`matgl.models` (M3GNet, CHGNet, MEGNet, TensorNet, SO3Net,
QET, GRACE) are assembled from the layers exposed here. They fall into
roughly seven categories:

================== =========================================================
Category           Modules / public classes
================== =========================================================
Activations        :class:`ActivationFunction` enum;
                   ``SoftPlus2``, ``SoftExponential``, ``swish`` (see
                   :mod:`matgl.layers._activations`).
Radial / angular   :class:`RadialBesselFunction`, ``SphericalBesselFunction``,
basis              :class:`SphericalBesselWithHarmonics`,
                   :class:`FourierExpansion`, ``GaussianExpansion``,
                   ``ExpNormalFunction`` (see :mod:`matgl.layers._basis`);
                   :class:`BondExpansion` is the convenience wrapper.
Embeddings         :class:`EmbeddingBlock` (backend-agnostic),
                   :class:`TensorEmbedding` (TensorNet),
                   ``NeighborEmbedding`` (TensorNet, DGL only).
Core MLPs          :class:`MLP`, :class:`GatedMLP`, :class:`GatedEquivariantBlock`,
                   :func:`build_gated_equivariant_mlp`, ``MLPNorm``,
                   ``GatedMLPNorm`` (DGL).
Graph convolution  :class:`M3GNetBlock`, :class:`M3GNetGraphConv`,
                   :class:`MEGNetBlock`, :class:`MEGNetGraphConv`,
                   :class:`TensorNetInteraction`,
                   ``CHGNet*`` blocks (DGL only).
Three-body /       :class:`ThreeBodyInteractions` and SO3-coupling helpers
angular            in :mod:`matgl.layers._three_body` and
                   :mod:`matgl.layers._so3`.
Readout            :class:`Set2SetReadOut`, :class:`EdgeSet2Set`,
                   :class:`ReduceReadOut`, :class:`WeightedAtomReadOut`,
                   :class:`WeightedReadOut`, plus DGL-only
                   :class:`GlobalPool`, :class:`AttentiveFPReadout`,
                   ``WeightedReadOutPair``.
Other corrections  :class:`AtomRef` (per-element offsets),
                   :class:`NuclearRepulsion` (ZBL repulsion),
                   :class:`GraphNorm` (graph normalisation).
================== =========================================================

Backend split
-------------

matgl supports two graph backends. The default PyG backend is selected
unless ``MATGL_BACKEND=DGL`` is set in the environment **before** any
``import matgl``; switching after import does not retroactively re-import
backend-specific submodules. Each piece of backend-specific code lives in
a sibling ``_*_dgl.py`` / ``_*_pyg.py`` module, and the public names above
are re-exported from this ``__init__`` so callers can ignore which
implementation is active. As of this writing the PyG path covers
TensorNet, M3GNet, MEGNet, GRACE and SO3Net; CHGNet, QET, and a handful
of readout flavours remain DGL-only.

Backend-agnostic modules (no ``_pyg`` / ``_dgl`` suffix) operate purely on
plain tensors and can be reused from both code paths --
:mod:`matgl.layers._activations`, :mod:`matgl.layers._basis`,
:mod:`matgl.layers._bond`, :mod:`matgl.layers._embedding` (the
``EmbeddingBlock`` base), :mod:`matgl.layers._core`,
:mod:`matgl.layers._three_body`, :mod:`matgl.layers._so3`, and the
:class:`GraphNorm` layer in :mod:`matgl.layers._norm`.

Public-vs-private convention
----------------------------

All ``_`` -prefixed modules are private. Add new public names through
this ``__init__`` rather than importing from the underscored module
directly.
"""

from __future__ import annotations

from matgl.config import BACKEND
from matgl.layers._activations import ActivationFunction
from matgl.layers._basis import FourierExpansion, RadialBesselFunction, SphericalBesselWithHarmonics
from matgl.layers._bond import BondExpansion
from matgl.layers._core import (
    MLP,
    GatedEquivariantBlock,
    GatedMLP,
    build_gated_equivariant_mlp,
)
from matgl.layers._embedding import EmbeddingBlock
from matgl.layers._norm import GraphNorm
from matgl.layers._three_body import ThreeBodyInteractions

if BACKEND == "DGL":
    from matgl.layers._atom_ref_dgl import AtomRef
    from matgl.layers._core_dgl import EdgeSet2Set, GatedMLPNorm, MLPNorm
    from matgl.layers._embedding_dgl import NeighborEmbedding, TensorEmbedding
    from matgl.layers._graph_convolution_dgl import (
        CHGNetAtomGraphBlock,
        CHGNetBondGraphBlock,
        CHGNetGraphConv,
        CHGNetLineGraphConv,
        M3GNetBlock,
        M3GNetGraphConv,
        MEGNetBlock,
        MEGNetGraphConv,
        TensorNetInteraction,
    )
    from matgl.layers._readout_dgl import (
        AttentiveFPReadout,
        GlobalPool,
        ReduceReadOut,
        Set2SetReadOut,
        WeightedAtomReadOut,
        WeightedReadOut,
        WeightedReadOutPair,
    )
    from matgl.layers._zbl_dgl import NuclearRepulsion
else:
    from matgl.layers._atom_ref_pyg import AtomRef  # type: ignore[assignment]
    from matgl.layers._embedding_pyg import TensorEmbedding  # type: ignore[assignment]
    from matgl.layers._graph_convolution_pyg import (  # type: ignore[assignment]
        M3GNetBlock,
        M3GNetGraphConv,
        MEGNetBlock,
        MEGNetGraphConv,
        TensorNetInteraction,
    )
    from matgl.layers._readout_pyg import (  # type: ignore[assignment]
        EdgeSet2Set,
        ReduceReadOut,
        Set2SetReadOut,
        WeightedAtomReadOut,
        WeightedReadOut,
    )
    from matgl.layers._zbl_pyg import NuclearRepulsion  # type: ignore[assignment]
