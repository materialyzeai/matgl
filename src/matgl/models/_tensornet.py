"""Implementation of TensorNet model.

A Cartesian based equivariant GNN model. For more details on TensorNet,
please refer to::

    G. Simeon, G. de. Fabritiis, _TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular
    Potentials. _arXiv, June 10, 2023, 10.48550/arXiv.2306.06482.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn

import matgl
from matgl.config import DEFAULT_ELEMENTS
from matgl.graph._compute import compute_pair_vector_and_distance
from matgl.layers import (
    MLP,
    ActivationFunction,
    BondExpansion,
)
from matgl.layers._embedding import TensorEmbedding as TensorEmbeddingPyG
from matgl.layers._graph_convolution import TensorNetInteraction as TensorNetInteractionPyG
from matgl.layers._readout_torch import (
    ReduceReadOut,
    WeightedAtomReadOut,
    WeightedReadOut,
)
from matgl.utils.maths import decompose_tensor, scatter_add, tensor_norm

from ._core import MatGLModel

try:
    from matgl.layers._embedding_warp import TensorEmbedding as TensorEmbeddingWarp
    from matgl.layers._graph_convolution_warp import TensorNetInteraction as TensorNetInteractionWarp
    from matgl.ops import fn_tensor_norm3, graph_transform

    _warp_available = True
except ImportError:
    _warp_available = False

if TYPE_CHECKING:
    from matgl.graph._converters import GraphConverter


class TensorNet(MatGLModel):
    """The main TensorNet model. The official implementation can be found in https://github.com/torchmd/torchmd-net.

    When the ``nvalchemi-toolkit-ops`` package is installed, GPU-accelerated warp kernels are used automatically
    for message passing. Pass ``use_warp=False`` to force the plain PyG implementation.
    """

    __version__ = 2
    final_layer: MLP | WeightedReadOut

    def __init__(
        self,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        units: int = 64,
        ntypes_state: int | None = None,
        dim_state_embedding: int = 0,
        dim_state_feats: int | None = None,
        include_state: bool = False,
        nblocks: int = 2,
        num_rbf: int = 32,
        max_n: int = 3,
        max_l: int = 3,
        rbf_type: Literal["Gaussian", "SphericalBessel"] = "Gaussian",
        use_smooth: bool = False,
        activation_type: Literal["swish", "tanh", "sigmoid", "softplus2", "softexp"] = "swish",
        cutoff: float = 5.0,
        equivariance_invariance_group: str = "O(3)",
        dtype: torch.dtype = matgl.float_th,
        width: float = 0.5,
        readout_type: Literal["weighted_atom", "reduce_atom"] = "weighted_atom",
        task_type: Literal["classification", "regression"] = "regression",
        field: Literal["node_feat", "edge_feat"] = "node_feat",
        is_intensive: bool = True,
        ntargets: int = 1,
        use_warp: bool | None = None,
        **kwargs,
    ):
        r"""Initialize the TensorNet (PyG) model.

        Args:
            element_types (tuple): List of elements appearing in the dataset. Default to DEFAULT_ELEMENTS.
            units (int, optional): Hidden embedding size.
                (default: :obj:`64`)
            ntypes_state (int): Number of state labels
            dim_state_embedding (int): Number of hidden neurons in state embedding
            dim_state_feats (int): Number of state features after linear layer
            include_state (bool): Whether to include states features
            nblocks (int, optional): The number of interaction layers.
                (default: :obj:`2`)
            num_rbf (int, optional): The number of radial basis Gaussian functions :math:`\mu`.
                (default: :obj:`32`)
            max_n (int): maximum of n in spherical Bessel functions
            max_l (int): maximum of l in spherical Bessel functions
            rbf_type (str): Radial basis function. choose from 'Gaussian' or 'SphericalBessel'
            use_smooth (bool): Whether to use the smooth version of SphericalBessel functions.
                This is particularly important for the smoothness of PES.
            activation_type (str): Activation type. choose from 'swish', 'tanh', 'sigmoid', 'softplus2', 'softexp'
            cutoff (float): cutoff distance for interatomic interactions.
            equivariance_invariance_group (string, optional): Group under whose action on input
                positions internal tensor features will be equivariant and scalar predictions
                will be invariant. O(3) or SO(3).
                (default :obj:`"O(3)"`)
            dtype (torch.dtype): data type for all variables
            width (float): the width of Gaussian radial basis functions
            readout_type (str): Readout function type, `set2set`, `weighted_atom` (default) or `reduce_atom`.
            task_type (str): `classification` or `regression` (default).
            field (str): Using either "node_feat" or "edge_feat" for Set2Set and Reduced readout
            is_intensive (bool): Whether the prediction is intensive
            ntargets (int): Number of target properties
            use_warp (bool | None): Whether to use warp-accelerated kernels from ``nvalchemi-toolkit-ops``.
                ``None`` (default) auto-detects: warp is used when the package is installed.
                ``True`` raises ``ImportError`` if the package is not available.
                ``False`` forces the plain PyG implementation.
            **kwargs: For future flexibility. Not used at the moment.
        """
        super().__init__()

        self.save_args(locals(), kwargs)

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        if use_warp and not _warp_available:
            raise ImportError(
                "use_warp=True but nvalchemi-toolkit-ops is not installed. "
                "Install it or pass use_warp=False to use the plain PyG backend."
            )
        self._use_warp = _warp_available if use_warp is None else use_warp

        self.element_types = element_types  # type: ignore

        self.bond_expansion = BondExpansion(
            cutoff=cutoff,
            rbf_type=rbf_type,
            final=cutoff + 1.0,
            num_centers=num_rbf,
            width=width,
            smooth=use_smooth,
            max_n=max_n,
            max_l=max_l,
        )

        assert equivariance_invariance_group in ["O(3)", "SO(3)"], "Unknown group representation. Choose O(3) or SO(3)."

        self.units = units
        self.equivariance_invariance_group = equivariance_invariance_group
        self.num_layers = nblocks
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.activation = activation
        self.cutoff = cutoff
        self.dim_state_embedding = dim_state_embedding
        self.dim_state_feats = dim_state_feats
        self.include_state = include_state
        self.ntypes_state = ntypes_state
        self.task_type = task_type

        # The radial-basis width fed to the embedding / interaction layers must match
        # the actual SphericalBessel output: the smooth basis emits ``max_n`` features,
        # the non-smooth basis emits ``max_l * max_n`` (one ``max_n``-wide block per
        # angular order l). Using ``max_n`` for the non-smooth case sized the input
        # Linear layers too small and crashed the forward pass with a shape mismatch.
        if rbf_type == "SphericalBessel":
            num_rbf = max_n if use_smooth else max_l * max_n

        EmbeddingCls = TensorEmbeddingWarp if self._use_warp else TensorEmbeddingPyG  # type: ignore[assignment]
        InteractionCls = TensorNetInteractionWarp if self._use_warp else TensorNetInteractionPyG  # type: ignore[assignment]

        self.tensor_embedding = EmbeddingCls(
            units=units,
            degree_rbf=num_rbf,
            activation=activation,
            ntypes_node=len(element_types),
            cutoff=cutoff,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            InteractionCls(num_rbf, units, activation, cutoff, equivariance_invariance_group, dtype)
            for _ in range(nblocks)
        )

        self.out_norm = nn.LayerNorm(3 * units, dtype=dtype)
        self.linear = nn.Linear(3 * units, units, dtype=dtype)
        self.is_intensive = is_intensive
        self._build_readout(
            units=units,
            ntargets=ntargets,
            readout_type=readout_type,
            field=field,
            activation=activation,
            task_type=task_type,
        )
        self.reset_parameters()

    def _build_readout(
        self,
        units: int,
        ntargets: int,
        readout_type: Literal["weighted_atom", "reduce_atom"],
        field: Literal["node_feat", "edge_feat"],
        activation: nn.Module,
        task_type: Literal["classification", "regression"],
    ) -> None:
        """Build the readout / ``final_layer`` modules.

        Override in a subclass to skip or replace the default readout construction
        (e.g. ``QET`` builds its own atomic-energy head over a wider concatenated
        node feature).
        """
        if not self.is_intensive:
            if task_type == "classification":
                raise ValueError("Classification task cannot be extensive.")
            self.final_layer = WeightedReadOut(
                in_feats=units,
                dims=[units, units],
                num_targets=ntargets,  # type: ignore
            )
            return

        if readout_type == "weighted_atom":
            self.readout = WeightedAtomReadOut(  # type:ignore[assignment]
                in_feats=units, dims=[units, units], activation=activation
            )
        else:
            self.readout = ReduceReadOut("mean", field=field)  # type: ignore
        self.final_layer = MLP([units, units, units, ntargets], activation, activate_last=False)
        if task_type == "classification":
            self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        self.tensor_embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()

    def forward_features(
        self,
        g: Any,
        state_attr: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run TensorNet's feature extraction up through the per-atom readout.

        Returns a dict with intermediate features (``edge_attr``, ``embedding``,
        ``gc_<i>``, ``readout``). Subclasses (e.g. ``QET``) reuse this without
        re-implementing the embedding / interaction stack.

        Args:
            g: PyG Data object or dict with keys 'node_type'/'z', 'pos', 'edge_index',
               and optionally 'pbc_offshift', 'batch', 'num_graphs'.
            state_attr: State attrs for a batch of graphs.
            **kwargs: For future flexibility. Not used at the moment.
        """
        z = getattr(g, "node_type", getattr(g, "z", None))
        pos = g.pos
        edge_index = g.edge_index
        pbc_offshift = getattr(g, "pbc_offshift", None)

        # Bond vectors and distances
        bond_vec, bond_dist = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)

        # Expand distances with radial basis functions
        edge_attr = self.bond_expansion(bond_dist)

        fea_dict: dict[str, Any] = {"edge_attr": edge_attr}

        if self._use_warp:
            (  # type: ignore[name-defined]
                row_data,
                row_indices,
                row_indptr,
                col_data,
                col_indices,
                col_indptr,
            ) = graph_transform(edge_index.int(), z.shape[0])  # type: ignore[union-attr]
            X = self.tensor_embedding(z, edge_index, bond_dist, bond_vec, edge_attr, row_data, row_indptr)
            fea_dict["embedding"] = X
            for i, layer in enumerate(self.layers):
                X = layer(
                    X,
                    edge_index,
                    bond_dist,
                    edge_attr,
                    row_data,
                    row_indices,
                    row_indptr,
                    col_data,
                    col_indices,
                    col_indptr,
                )
                fea_dict[f"gc_{i + 1}"] = X
            x = fn_tensor_norm3(X)  # type: ignore[name-defined]
        else:
            X, _ = self.tensor_embedding(z, edge_index, edge_attr, bond_dist, bond_vec, state_attr)
            fea_dict["embedding"] = X
            for i, layer in enumerate(self.layers):
                X = layer(edge_index, bond_dist, edge_attr, X)
                fea_dict[f"gc_{i + 1}"] = X
            scalars, skew_metrices, traceless_tensors = decompose_tensor(X)
            x = torch.cat((tensor_norm(scalars), tensor_norm(skew_metrices), tensor_norm(traceless_tensors)), dim=-1)

        x = self.out_norm(x)
        x = self.linear(x)
        fea_dict["readout"] = x
        return fea_dict

    def forward(
        self,
        g: Any,
        state_attr: torch.Tensor | None = None,
        return_all_layer_output: bool = False,
        **kwargs,
    ):
        """Forward pass for TensorNet (PyG).

        Args:
            g: PyG Data object or dict with keys 'node_type'/'z', 'pos', 'edge_index',
                and optionally 'pbc_offshift', 'batch', 'num_graphs'.
            state_attr: State attrs for a batch of graphs.
            return_all_layer_output: Whether to return outputs of all intermediate layers.
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            output: Output property for a batch of graphs, or a dict of layer outputs
                when ``return_all_layer_output=True``.
        """
        fea_dict = self.forward_features(g=g, state_attr=state_attr)
        x = fea_dict["readout"]
        batch = getattr(g, "batch", None)
        num_graphs = getattr(g, "num_graphs", None)

        if self.is_intensive:
            node_vec = self.readout(x, batch)
            output = self.final_layer(node_vec)
            if self.task_type == "classification":
                output = self.sigmoid(output)
            output = torch.squeeze(output)
        else:
            atomic_energies = self.final_layer(x).view(-1)
            if batch is None:
                output = atomic_energies.sum()
            else:
                batch_long = batch.to(torch.long)
                if num_graphs is None:
                    num_graphs = int(batch_long.max().item()) + 1
                output = scatter_add(atomic_energies, batch_long, dim_size=num_graphs)  # type: ignore[arg-type]

        fea_dict["final"] = output
        if return_all_layer_output:
            return fea_dict
        return output

    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
        output_layers: list | None = None,
        return_features: bool = False,
    ):
        """Convenience method to directly predict property from structure.

        Args:
            structure: An input crystal/molecule.
            state_feats (torch.tensor): Graph attributes
            graph_converter: Object that implements a get_graph_from_structure.
            output_layers: List of names for the layer of GNN as output.
            return_features (bool): If True, return specified layer outputs. If False, only return final output.

        Returns:
            output (torch.tensor): output property
        """
        allowed_output_layers = ["edge_attr", "embedding", "readout", "final"] + [
            f"gc_{i + 1}" for i in range(self.num_layers)
        ]

        if not return_features:
            output_layers = ["final"]
        elif output_layers is None:
            output_layers = allowed_output_layers
        elif not isinstance(output_layers, list) or set(output_layers).difference(allowed_output_layers):
            raise ValueError(f"Invalid output_layers, it must be a sublist of {allowed_output_layers}.")

        if graph_converter is None:
            from matgl.ext._pymatgen import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)  # type: ignore
        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        g.pos = g.frac_coords @ lat[0]
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)

        if return_features:
            model_output = self(g=g, state_attr=state_feats, return_all_layer_output=True)
            return {
                k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in model_output.items()
                if k in output_layers
            }

        return self(g=g, state_attr=state_feats).detach()
