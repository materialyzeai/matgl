"""Layers for graph neural networks."""

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
