"""Implementation of TensorNet model.

A Cartesian based equivariant GNN model. For more details on TensorNet,
please refer to::

    G. Simeon, G. de. Fabritiis, _TensorNet: Cartesian Tensor Representations for Efficient Learning of Molecular
    Potentials. _arXiv, June 10, 2023, 10.48550/arXiv.2306.06482.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

import matgl
from matgl.config import DEFAULT_ELEMENTS
from matgl.layers import (
    MLP,
    ActivationFunction,
    BondExpansion,
)
from matgl.layers._readout_torch import (
    ReduceReadOut,
    WeightedAtomReadOut,
    WeightedReadOut,
)
from matgl.ops import (
    fn_compose_tensor,
    fn_decompose_tensor,
    fn_message_passing,
    fn_radial_message_passing,
    fn_tensor_matmul_o3_3x3,
    fn_tensor_matmul_so3_3x3,
    fn_tensor_norm3,
    graph_transform,
)
from matgl.utils.cutoff import cosine_cutoff
from matgl.utils.maths import scatter_add

from ._core import MatGLModel

if TYPE_CHECKING:
    from matgl.graph._converters_pyg import GraphConverter

logger = logging.getLogger(__file__)


def compute_pair_vector_and_distance(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    pbc_offshift: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate bond vectors and distances.

    Args:
        pos: Node positions, shape (num_nodes, 3)
        edge_index: Edge indices, shape (2, num_edges)
        pbc_offshift: Periodic boundary condition offsets, shape (num_edges, 3)

    Returns:
        bond_vec: Bond vectors, shape (num_edges, 3)
        bond_dist: Bond distances, shape (num_edges,)
    """
    src_idx, dst_idx = edge_index[0], edge_index[1]
    src_pos = pos[src_idx]
    dst_pos = pos[dst_idx]

    if pbc_offshift is not None:
        dst_pos = dst_pos + pbc_offshift

    bond_vec = dst_pos - src_pos
    bond_dist = torch.norm(bond_vec, dim=1)

    return bond_vec, bond_dist


def tensor_norm(tensor):
    """Computes Frobenius norm."""
    return (tensor * tensor).sum((-3, -2))


class TensorEmbedding(nn.Module):
    """Pure PyTorch TensorNet embedding layer."""

    def __init__(
        self,
        units: int,
        degree_rbf: int,
        activation: nn.Module,
        ntypes_node: int,
        cutoff: float,
        dtype: torch.dtype = matgl.float_th,
    ):
        super().__init__()
        self.units = units
        self.cutoff = cutoff

        # Create unified distance_proj from 3 temp layers (matches reference RNG pattern).
        self.distance_proj = self._create_distance_proj(degree_rbf, units, dtype=dtype)

        self.emb = nn.Embedding(ntypes_node, units, dtype=dtype)
        self.emb2 = nn.Linear(2 * units, units, dtype=dtype)
        self.act = activation
        self.linears_tensor = nn.ModuleList([nn.Linear(units, units, bias=False, dtype=dtype) for _ in range(3)])
        self.linears_scalar = nn.ModuleList(
            [
                nn.Linear(units, 2 * units, bias=True, dtype=dtype),
                nn.Linear(2 * units, 3 * units, bias=True, dtype=dtype),
            ]
        )
        self.init_norm = nn.LayerNorm(units, dtype=dtype)

        self.reset_parameters()

    def _create_distance_proj(
        self,
        in_features: int,
        units: int,
        dtype: torch.dtype = matgl.float_th,
    ) -> nn.Linear:
        """Create unified distance_proj from 3 separate layers to match reference RNG pattern."""
        d_proj1 = nn.Linear(in_features, units, bias=True, dtype=dtype)
        d_proj2 = nn.Linear(in_features, units, bias=True, dtype=dtype)
        d_proj3 = nn.Linear(in_features, units, bias=True, dtype=dtype)

        layer = torch.nn.utils.skip_init(nn.Linear, in_features, 3 * units, bias=True, dtype=dtype)
        with torch.no_grad():
            layer.weight.copy_(torch.cat([d_proj1.weight, d_proj2.weight, d_proj3.weight], dim=0))
            layer.bias.copy_(torch.cat([d_proj1.bias, d_proj2.bias, d_proj3.bias], dim=0))
        return layer

    def _reset_distance_proj(self) -> None:
        """Reset distance_proj weights using 3 temp layers to match reference RNG pattern."""
        dtype = self.distance_proj.weight.dtype
        d_proj1 = torch.nn.utils.skip_init(
            nn.Linear, self.distance_proj.in_features, self.units, bias=True, dtype=dtype
        )
        d_proj2 = torch.nn.utils.skip_init(
            nn.Linear, self.distance_proj.in_features, self.units, bias=True, dtype=dtype
        )
        d_proj3 = torch.nn.utils.skip_init(
            nn.Linear, self.distance_proj.in_features, self.units, bias=True, dtype=dtype
        )
        d_proj1.reset_parameters()
        d_proj2.reset_parameters()
        d_proj3.reset_parameters()
        with torch.no_grad():
            self.distance_proj.weight.copy_(torch.cat([d_proj1.weight, d_proj2.weight, d_proj3.weight], dim=0))
            self.distance_proj.bias.copy_(torch.cat([d_proj1.bias, d_proj2.bias, d_proj3.bias], dim=0))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Handle legacy checkpoints with separate distance_proj1/2/3 layers."""
        w_keys = [f"{prefix}distance_proj{i}.weight" for i in (1, 2, 3)]
        b_keys = [f"{prefix}distance_proj{i}.bias" for i in (1, 2, 3)]
        new_w = f"{prefix}distance_proj.weight"
        new_b = f"{prefix}distance_proj.bias"

        if all(k in state_dict for k in w_keys + b_keys):
            state_dict[new_w] = torch.cat([state_dict.pop(k) for k in w_keys], dim=0)
            state_dict[new_b] = torch.cat([state_dict.pop(k) for k in b_keys], dim=0)

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def reset_parameters(self):
        """Reinitialize parameters with RNG pattern matching reference implementation."""
        self._reset_distance_proj()
        self.emb.reset_parameters()
        self.emb2.reset_parameters()
        for linear in self.linears_tensor:
            linear.reset_parameters()
        for linear in self.linears_scalar:
            linear.reset_parameters()
        self.init_norm.reset_parameters()

    def forward(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_attr: torch.Tensor,
        col_data: torch.Tensor,
        col_indptr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Node types, shape (num_nodes,)
            edge_index: Edge indices, shape (2, num_edges)
            edge_weight: Edge weights (distances), shape (num_edges,)
            edge_vec: Edge vectors, shape (num_edges, 3)
            edge_attr: Edge attributes (RBF), shape (num_edges, num_rbf)
            col_data: CSR col data for destination aggregation, shape (num_edges,)
            col_indptr: CSR col indptr for destination aggregation, shape (num_nodes+1,)

        Returns:
            X: Tensor representation, shape (num_nodes, 3, 3, units)
        """
        # Node embedding
        x = self.emb(z)  # (num_nodes, units)

        # Edge processing
        C = cosine_cutoff(edge_weight, self.cutoff)
        edge_attr = self.distance_proj(edge_attr).view(-1, 3, self.units)

        # Get atomic number messages
        zij = x.index_select(0, edge_index.t().reshape(-1)).view(-1, self.units * 2)
        Zij = self.emb2(zij)  # (num_edges, units)

        # Create edge attributes with Zij
        edge_attr_processed = edge_attr.view(-1, 3, self.units) * C.view(-1, 1, 1) * Zij.view(-1, 1, self.units)

        # Radial message passing
        edge_vec_norm = edge_vec / torch.norm(edge_vec, dim=1, keepdim=True).clamp(min=1e-6)
        I, A, S = fn_radial_message_passing(edge_vec_norm, edge_attr_processed, col_data, col_indptr)  # noqa: E741

        # Compose initial tensor to get proper shape for norm computation
        X = fn_compose_tensor(I, A, S)  # (num_nodes, 3, 3, units)

        # Normalize and process

        norm = tensor_norm(X)  # (num_nodes, units)
        norm = self.init_norm(norm)  # (num_nodes, units)

        # Process norm through scalar layers
        for linear_scalar in self.linears_scalar:
            norm = self.act(linear_scalar(norm))

        norm = norm.view(-1, self.units, 3)
        norm_I, norm_A, norm_S = norm.unbind(dim=-1)

        # Apply norm to tensors
        I = self.linears_tensor[0](I) * norm_I.unsqueeze(-2)  # noqa: E741
        A = self.linears_tensor[1](A) * norm_A.unsqueeze(-2)
        S = self.linears_tensor[2](S) * norm_S.unsqueeze(-2)

        X = fn_compose_tensor(I, A, S)

        return X


class TensorNetInteraction(nn.Module):
    """Pure PyTorch TensorNet interaction layer."""

    def __init__(
        self,
        num_rbf: int,
        units: int,
        activation: nn.Module,
        cutoff: float,
        equivariance_invariance_group: str,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.num_rbf = num_rbf
        self.units = units
        self.cutoff = cutoff
        self.equivariance_invariance_group = equivariance_invariance_group

        self.linears_scalar = nn.ModuleList(
            [
                nn.Linear(num_rbf, units, bias=True, dtype=dtype),
                nn.Linear(units, 2 * units, bias=True, dtype=dtype),
                nn.Linear(2 * units, 3 * units, bias=True, dtype=dtype),
            ]
        )

        self.linears_tensor = nn.ModuleList([nn.Linear(units, units, bias=False, dtype=dtype) for _ in range(6)])

        self.act = activation
        self.reset_parameters()

    def reset_parameters(self):
        for linear in self.linears_scalar:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        for linear in self.linears_tensor:
            nn.init.xavier_uniform_(linear.weight)

    def forward(
        self,
        X: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
        row_data: torch.Tensor,
        row_indices: torch.Tensor,
        row_indptr: torch.Tensor,
        col_data: torch.Tensor,
        col_indices: torch.Tensor,
        col_indptr: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            X: Node tensor representations, shape (num_nodes, 3, 3, units)
            edge_index: Edge indices, shape (2, num_edges)
            edge_weight: Edge weights (distances), shape (num_edges,)
            edge_attr: Edge attributes (RBF), shape (num_edges, num_rbf)
            row_data: CSR row data indices for message passing.
            row_indices: CSR row indices for message passing.
            row_indptr: CSR row pointers for message passing.
            col_data: CSC column data indices for message passing.
            col_indices: CSC column indices for message passing.
            col_indptr: CSC column pointers for message passing.

        Returns:
            X: Updated tensor representations, shape (num_nodes, 3, 3, units)
        """
        # Process edge attributes
        C = cosine_cutoff(edge_weight, self.cutoff)
        edge_attr_processed = edge_attr
        for linear_scalar in self.linears_scalar:
            edge_attr_processed = self.act(linear_scalar(edge_attr_processed))
        edge_attr_processed = (
            (edge_attr_processed * C.view(-1, 1)).view(edge_attr.shape[0], self.units, 3).mT.contiguous()
        )  # (num_edges, 3, units)

        # Normalize input tensor
        # For X with shape (num_nodes, 3, 3, units), we need to sum over (-3, -2)
        # which are the (3, 3) spatial dimensions to get (num_nodes, units)
        norm_X = (X * X).sum((-3, -2)) + 1  # (num_nodes, units)
        X = X / norm_X.view(-1, 1, 1, X.shape[-1])

        # Decompose input tensor
        I, A, S = fn_decompose_tensor(X)  # noqa: E741

        # Apply tensor linear transformations
        I = self.linears_tensor[0](I)  # noqa: E741
        A = self.linears_tensor[1](A)
        S = self.linears_tensor[2](S)

        # compose back
        Y = fn_compose_tensor(I, A, S)

        # Message passing
        Im, Am, Sm = fn_message_passing(
            I,
            A,
            S,
            edge_attr_processed,
            row_data,
            row_indices,
            row_indptr,
            col_data,
            col_indices,
            col_indptr,
        )
        msg = fn_compose_tensor(Im, Am, Sm)

        # Apply group action
        if self.equivariance_invariance_group == "O(3)":
            C = fn_tensor_matmul_o3_3x3(Y, msg)
        else:  # SO(3)
            C = 2 * fn_tensor_matmul_so3_3x3(Y, msg)
        I, A, S = fn_decompose_tensor(C)  # noqa: E741

        # Normalize
        normp1 = (tensor_norm(C) + 1).unsqueeze(-2)
        I, A, S = I / normp1, A / normp1, S / normp1  # noqa: E741

        # Final tensor transformations
        I = self.linears_tensor[3](I)  # noqa: E741
        A = self.linears_tensor[4](A)
        S = self.linears_tensor[5](S)
        dX = fn_compose_tensor(I, A, S)
        X = X + dX + fn_tensor_matmul_so3_3x3(dX, dX)

        return X


class TensorNet(MatGLModel):
    """The main TensorNet model. The official implementation can be found in https://github.com/torchmd/torchmd-net."""

    __version__ = 1

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
        niters_set2set: int = 3,
        nlayers_set2set: int = 3,
        field: Literal["node_feat", "edge_feat"] = "node_feat",
        is_intensive: bool = True,
        ntargets: int = 1,
        **kwargs,
    ):
        r"""

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
            niters_set2set (int): Number of set2set iterations
            nlayers_set2set (int): Number of set2set layers
            field (str): Using either "node_feat" or "edge_feat" for Set2Set and Reduced readout
            is_intensive (bool): Whether the prediction is intensive
            ntargets (int): Number of target properties
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

        # make sure the number of radial basis functions correct for tensor embedding
        if rbf_type == "SphericalBessel":
            num_rbf = max_n

        self.tensor_embedding = TensorEmbedding(
            units=units,
            degree_rbf=num_rbf,
            activation=activation,
            ntypes_node=len(element_types),
            cutoff=cutoff,
            dtype=dtype,
        )

        self.layers = nn.ModuleList(
            [
                TensorNetInteraction(num_rbf, units, activation, cutoff, equivariance_invariance_group, dtype)
                for _ in range(nblocks)
                if nblocks != 0
            ]
        )

        self.out_norm = nn.LayerNorm(3 * units, dtype=dtype)
        self.linear = nn.Linear(3 * units, units, dtype=dtype)
        if is_intensive:
            input_feats = units
            if readout_type == "weighted_atom":
                self.readout = WeightedAtomReadOut(in_feats=input_feats, dims=[units, units], activation=activation)  # type:ignore[assignment]
                readout_feats = units
            else:
                self.readout = ReduceReadOut("mean", field=field)  # type: ignore
                readout_feats = input_feats  # type: ignore

            dims_final_layer = [readout_feats, units, units, ntargets]
            self.final_layer = MLP(dims_final_layer, activation, activate_last=False)
            if task_type == "classification":
                self.sigmoid = nn.Sigmoid()

        else:
            if task_type == "classification":
                raise ValueError("Classification task cannot be extensive.")
            self.final_layer = WeightedReadOut(
                in_feats=units,
                dims=[units, units],
                num_targets=ntargets,  # type: ignore
            )

        self.is_intensive = is_intensive
        self.reset_parameters()

    def reset_parameters(self):
        self.tensor_embedding.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()

    def forward(
        self,
        g: torch.Tensor | dict[str, torch.Tensor] | object,
        state_attr: torch.Tensor | None = None,
        **kwargs,
    ):
        """
        Args:
            g: Either a PyG Data object or a dict with keys:
                - 'node_type' or 'z': Node types, shape (num_nodes,)
                - 'pos': Node positions, shape (num_nodes, 3)
                - 'edge_index': Edge indices, shape (2, num_edges)
                - 'pbc_offshift': Optional PBC offsets, shape (num_edges, 3)
                - 'batch': Optional batch indices, shape (num_nodes,)
            state_attr: State attrs for a batch of graphs (not used currently).
            **kwargs: For future flexibility. Not used at the moment.

        Returns:
            output: Output property for a batch of graphs
        """
        # Handle both dict and PyG Data object
        if isinstance(g, dict):
            z = g.get("node_type", g.get("z"))
            pos = g["pos"]
            edge_index = g["edge_index"]
            pbc_offshift = g.get("pbc_offshift", None)
            batch = g.get("batch", None)
            num_graphs = g.get("num_graphs", None)
        else:
            # PyG Data object - extract tensors
            z = getattr(g, "node_type", getattr(g, "z", None))
            pos = g.pos  # type: ignore[attr-defined]
            edge_index = g.edge_index  # type: ignore[attr-defined]
            pbc_offshift = getattr(g, "pbc_offshift", None)
            batch = getattr(g, "batch", None)
            num_graphs = getattr(g, "num_graphs", None)

        # Obtain graph, with distances and relative position vectors
        bond_vec, bond_dist = compute_pair_vector_and_distance(pos, edge_index, pbc_offshift)

        # prepare graph indices for message passing
        row_data, row_indices, row_indptr, col_data, col_indices, col_indptr = graph_transform(
            edge_index.int(),
            z.shape[0],  # type: ignore[union-attr]
        )

        # Expand distances with radial basis functions
        edge_attr = self.bond_expansion(bond_dist)

        # Embedding layer
        X = self.tensor_embedding(z, edge_index, bond_dist, bond_vec, edge_attr, col_data, col_indptr)

        # Interaction layers
        for layer in self.layers:
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

        # compute I, A, S norms
        x = fn_tensor_norm3(X)
        # normalize
        x = self.out_norm(x)
        x = self.linear(x)

        if self.is_intensive:
            node_vec = self.readout(x, batch)
            vec = node_vec  # type: ignore
            output = self.final_layer(vec)
            if self.task_type == "classification":
                output = self.sigmoid(output)
            return torch.squeeze(output)

        atomic_energies = self.final_layer(x)
        if batch is not None:
            # edge case, if we do squeeze() directly, we will get torch.size([]) and it will crash in the training.
            if atomic_energies.shape == (1, 1):
                atomic_energies = atomic_energies.squeeze(-1)
            else:
                atomic_energies = atomic_energies.squeeze()
            # Batch case: Use scatter_add with batch tensor
            batch_long = batch.to(torch.long)
            if num_graphs is None:
                num_graphs = int(batch_long.max().item()) + 1
            return scatter_add(atomic_energies, batch_long, dim_size=num_graphs)  # type: ignore[arg-type]
        # Single graph case: Sum all energies (equivalent to scatter_add with all nodes in one graph)
        return torch.sum(atomic_energies, dim=0, keepdim=True).squeeze()

    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
    ):
        """Convenience method to directly predict property from structure.

        Args:
            structure: An input crystal/molecule.
            state_feats (torch.tensor): Graph attributes
            graph_converter: Object that implements a get_graph_from_structure.

        Returns:
            output (torch.tensor): output property
        """
        if graph_converter is None:
            from matgl.ext._pymatgen_pyg import Structure2Graph

            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)  # type: ignore
        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        g.pos = g.frac_coords @ lat[0]
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)
        return self(g=g, state_attr=state_feats).detach()
