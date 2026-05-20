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
Embeddings         :class:`EmbeddingBlock`, :class:`TensorEmbedding`
                   (TensorNet).
Core MLPs          :class:`MLP`, :class:`GatedMLP`, :class:`GatedEquivariantBlock`,
                   :func:`build_gated_equivariant_mlp`.
Graph convolution  :class:`M3GNetBlock`, :class:`M3GNetGraphConv`,
                   :class:`MEGNetBlock`, :class:`MEGNetGraphConv`,
                   :class:`TensorNetInteraction`, ``CHGNet*`` blocks.
Three-body /       :class:`ThreeBodyInteractions` and SO3-coupling helpers
angular            in :mod:`matgl.layers._three_body` and
                   :mod:`matgl.layers._so3`.
Readout            :class:`Set2SetReadOut`, :class:`EdgeSet2Set`,
                   :class:`ReduceReadOut`, :class:`WeightedAtomReadOut`,
                   :class:`WeightedReadOut`.
Other corrections  :class:`AtomRef` (per-element offsets),
                   :class:`NuclearRepulsion` (ZBL repulsion).
================== =========================================================

Public-vs-private convention
----------------------------

All ``_`` -prefixed modules are private. Add new public names through
this ``__init__`` rather than importing from the underscored module
directly.
"""

from __future__ import annotations

from matgl.layers._activations import ActivationFunction
from matgl.layers._atom_ref import AtomRef
from matgl.layers._basis import FourierExpansion, RadialBesselFunction, SphericalBesselWithHarmonics
from matgl.layers._bond import BondExpansion
from matgl.layers._core import (
    MLP,
    GatedEquivariantBlock,
    GatedMLP,
    build_gated_equivariant_mlp,
)
from matgl.layers._embedding import EmbeddingBlock, TensorEmbedding
from matgl.layers._graph_convolution import (
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
from matgl.layers._readout import (
    EdgeSet2Set,
    ReduceReadOut,
    Set2SetReadOut,
    WeightedAtomReadOut,
    WeightedReadOut,
)
from matgl.layers._three_body import ThreeBodyInteractions
from matgl.layers._zbl import NuclearRepulsion
