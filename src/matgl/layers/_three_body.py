"""Three-Body interaction implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

import matgl
from matgl.utils.maths import _block_repeat, get_segment_indices_from_n, scatter_sum

if TYPE_CHECKING:
    import dgl


class ThreeBodyInteractions(nn.Module):
    """Include 3D interactions to the bond update."""

    def __init__(self, update_network_atom: nn.Module, update_network_bond: nn.Module, **kwargs):
        """Initialize ThreeBodyInteractions.

        Args:
            update_network_atom: MLP for node features in Eq.2
            update_network_bond: Gated-MLP for edge features in Eq.3
            **kwargs: Kwargs pass-through to nn.Module.__init__().
        """
        super().__init__(**kwargs)
        self.update_network_atom = update_network_atom
        self.update_network_bond = update_network_bond

    def forward(
        self,
        graph: dgl.DGLGraph,
        line_graph: dgl.DGLGraph,
        three_basis: torch.Tensor,
        three_cutoff: torch.Tensor,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
    ):
        """Forward function for ThreeBodyInteractions.

        Args:
            graph: dgl graph
            line_graph: line graph
            three_basis: three body basis expansion
            three_cutoff: cutoff radius
            node_feat: node features
            edge_feat: edge features
        """
        # Get the indices of the end atoms for each bond in the line graph
        end_atom_indices = graph.edges()[1][line_graph.edges()[1]].to(matgl.int_th)

        # Update node features using the atom update network
        updated_atoms = self.update_network_atom(node_feat)

        # Gather updated atom features for the end atoms
        end_atom_features = updated_atoms[end_atom_indices]

        # Compute the basis term
        basis = three_basis * end_atom_features

        # Reshape and compute weights based on the three-cutoff tensor
        three_cutoff = three_cutoff.unsqueeze(1)
        edge_indices = torch.stack(list(line_graph.edges()), dim=1)
        weights = three_cutoff[edge_indices].view(-1, 2)
        weights = weights.prod(dim=-1)

        # Compute the weighted basis
        basis = basis * weights[:, None]

        # Aggregate the new bonds using scatter_sum
        segment_ids = get_segment_indices_from_n(line_graph.ndata["n_triple_ij"])
        new_bonds = scatter_sum(
            basis.to(matgl.float_th),
            segment_ids=segment_ids,
            num_segments=graph.num_edges(),
            dim=0,
        )

        # If no new bonds are generated, return the original edge features
        if new_bonds.shape[0] == 0:
            return edge_feat

        # Update edge features using the bond update network
        updated_edge_feat = edge_feat + self.update_network_bond(new_bonds)

        return updated_edge_feat


def combine_sbf_shf(sbf, shf, max_n: int, max_l: int, use_phi: bool):
    """Combine the spherical Bessel function and the spherical Harmonics function.

    For the spherical Bessel function, the column is ordered by
        [n=[0, ..., max_n-1], n=[0, ..., max_n-1], ...], max_l blocks,

    For the spherical Harmonics function, the column is ordered by
        [m=[0], m=[-1, 0, 1], m=[-2, -1, 0, 1, 2], ...] max_l blocks, and each
        block has 2*l + 1
        if use_phi is False, then the columns become
        [m=[0], m=[0], ...] max_l columns

    Args:
        sbf: torch.Tensor spherical bessel function results
        shf: torch.Tensor spherical harmonics function results
        max_n: int, max number of n
        max_l: int, max number of l
        use_phi: whether to use phi
    Returns:
    """
    if sbf.size()[0] == 0:
        return sbf

    if not use_phi:
        repeats_sbf = torch.tensor([1] * max_l * max_n)
        block_size = [1] * max_l
    else:
        # [1, 1, 1, ..., 1, 3, 3, 3, ..., 3, ...]
        repeats_sbf = np.repeat(2 * torch.arange(max_l) + 1, repeats=max_n)  # type:ignore[assignment]
        # tf.repeat(2 * tf.range(max_l) + 1, repeats=max_n)
        block_size = 2 * torch.arange(max_l) + 1  # type: ignore
        # 2 * tf.range(max_l) + 1
    repeats_sbf = repeats_sbf.to(sbf.device)
    expanded_sbf = torch.repeat_interleave(sbf, repeats_sbf, 1)
    expanded_shf = _block_repeat(shf, block_size=block_size, repeats=[max_n] * max_l)
    shape = max_n * max_l
    if use_phi:
        shape *= max_l
    return torch.reshape(expanded_sbf * expanded_shf, [-1, shape])
